import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2

from lime import lime_image
from skimage.segmentation import mark_boundaries
from keras.preprocessing.image import img_to_array

import shap
from keras.applications.vgg16 import preprocess_input

class Lime:
    def explain(self, image, model, class_idx, class_names=None, num_samples=1000, num_features=8):
        img_array = np.array(image) / 255

        # Create a prediction wrapper that handles both TFSMLayer and regular Keras models
        def predict_fn(images):
            if isinstance(model, tf.keras.layers.TFSMLayer):
                # TFSMLayer returns a dict, extract the predictions
                preds_dict = model(images)
                preds = list(preds_dict.values())[0].numpy()
            else:
                # Regular Keras model
                preds = model.predict(images)
            return preds

        # Explaining the prediction
        explainer = lime_image.LimeImageExplainer()
        explanation = explainer.explain_instance(
            img_array.astype('float64'), 
            predict_fn, 
            hide_color=0, 
            num_samples=num_samples
        )
        
        temp, mask = explanation.get_image_and_mask(
            class_idx, 
            positive_only=False, 
            num_features=num_features, 
            hide_rest=True
        )

        class_label = (
            class_names[class_idx]
            if class_names and class_idx < len(class_names)
            else f"class {class_idx}"
        )

        fig, axs = plt.subplots(1, 2, figsize=(8, 4))
        axs[0].imshow(image)
        axs[0].set_title("Original image")

        axs[1].imshow(mark_boundaries(temp, mask))
        axs[1].set_title(f"LIME - Predicted class: {class_label}")
        plt.tight_layout()

        return(fig)


class GradCAM:
    def explain(self, image, model, class_idx, class_names=None):
        img_array = img_to_array(image)

        x = np.expand_dims(img_array,axis=0)
        x = tf.keras.applications.vgg16.preprocess_input(x)

        model = tf.keras.applications.VGG16(weights='imagenet', include_top=True)
        last_conv_layer = model.get_layer('block5_conv3')
        grad_model = tf.keras.models.Model([model.inputs], [last_conv_layer.output, model.output])

        with tf.GradientTape() as tape:
            last_conv_layer_output, preds = grad_model(x)
            class_output = preds[:, class_idx]

        grads = tape.gradient(class_output, last_conv_layer_output)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        last_conv_layer_output = last_conv_layer_output[0]
        heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)

        heatmap = cv2.resize(np.float32(heatmap), (x.shape[2], x.shape[1]))

        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        heatmap = heatmap.astype(np.float32)
        superimposed_img = cv2.addWeighted(x[0], 0.6, heatmap, 0.4, 0, dtype = cv2.CV_32F)

        class_label = (
            class_names[class_idx]
            if class_names and class_idx < len(class_names)
            else f"class {class_idx}"
        )

        # Showing the original image and the explanation
        fig, axs = plt.subplots(1, 2, figsize=(8, 4))
        axs[0].imshow(image)
        axs[0].set_title("Original image")

        axs[1].imshow(superimposed_img)
        axs[1].set_title(f"Grad-CAM - Predicted class: {class_label}")
        plt.tight_layout()

        return(fig)
    

class SHAP_Gradient:
    def explain(self, image, model, class_idx, background_images, class_names=None, max_background=10):
        """
        SHAP implementation using GradientExplainer.
        """
        img = img_to_array(image) / 255.0
        img = np.expand_dims(img, axis=0)

        # Preparing the background
        bg = []
        for bg_img in background_images[:max_background]:
            bg_arr = img_to_array(bg_img) / 255.0
            bg.append(bg_arr)

        bg = np.array(bg)
        
        # Ensure background has proper shape for GradientExplainer
        # Should be (n_samples, height, width, channels)
        if len(bg.shape) == 3:
            bg = np.expand_dims(bg, axis=0)

        # Handle different model output formats
        def create_wrapper_model(base_model):
            """Create a Keras model wrapper that handles TFSMLayer and dict outputs"""
            # Get the input shape from the image
            input_shape = img.shape[1:]  # (height, width, channels)
            
            # Check if it's a TFSMLayer
            if isinstance(base_model, tf.keras.layers.TFSMLayer):
                # Create a proper Keras model that wraps the TFSMLayer
                input_layer = tf.keras.Input(shape=input_shape, dtype=tf.float32)
                
                # Define the output processing
                def call_model(x):
                    preds = base_model(x)
                    if isinstance(preds, dict):
                        return list(preds.values())[0]
                    return preds
                
                output = tf.keras.layers.Lambda(call_model)(input_layer)
                wrapper_model = tf.keras.Model(inputs=input_layer, outputs=output)
                # Build the model with concrete input shape
                wrapper_model.build(input_shape=(None,) + input_shape)
                return wrapper_model
            else:
                # Regular Keras model - try to get logits
                try:
                    logit_model = tf.keras.Model(
                        inputs=base_model.input,
                        outputs=base_model.layers[-2].output
                    )
                    # Build the model with concrete input shape
                    logit_model.build(input_shape=(None,) + input_shape)
                    return logit_model
                except Exception:
                    # Return the model as-is if we can't extract logits
                    return base_model

        logit_model = create_wrapper_model(model)
        
        # Ensure data types match
        img = img.astype(np.float32)
        bg = bg.astype(np.float32)
        
        # Build the model explicitly with sample data to ensure concrete shapes
        _ = logit_model(bg[:1])

        # Use SHAP GradientExplainer with local smoothing disabled
        explainer = shap.GradientExplainer(logit_model, bg, local_smoothing=0)
        print("Computing SHAP values...")
        shap_values = explainer.shap_values(img, nsamples=200)
        
        # Process SHAP values
        if isinstance(shap_values, list) and len(shap_values) > class_idx:
            shap_map = shap_values[class_idx][0]
        else:
            shap_map = shap_values[0]
            
        shap_map = np.mean(shap_map, axis=-1)
        shap_map = (shap_map - shap_map.min()) / (shap_map.max() - shap_map.min() + 1e-8)

        class_label = (
            class_names[class_idx]
            if class_names and class_idx < len(class_names)
            else f"class {class_idx}"
        )

        # Create visualization
        fig, axs = plt.subplots(1, 2, figsize=(8, 4))
        axs[0].imshow(image)
        axs[0].set_title("Original image")

        axs[1].imshow(shap_map)
        axs[1].set_title(f"SHAP - Predicted class: {class_label}")
        plt.tight_layout()

        return fig


class SHAP_Kernel:
    def explain(self, image, model, class_idx, background_images, class_names=None, max_background=10, nsamples=100):
        """
        SHAP implementation using KernelExplainer (model-agnostic).
        
        Args:
            image: Input image to explain
            model: Keras model or TFSMLayer
            class_idx: Index of the class to explain
            background_images: List of background images for baseline
            class_names: Optional list of class names
            max_background: Maximum number of background images to use
            nsamples: Number of samples for KernelExplainer (higher = more accurate but slower)
        """
        # Prepare input image
        img = img_to_array(image) / 255.0
        img = np.expand_dims(img, axis=0)

        # Prepare background dataset
        bg = []
        for bg_img in background_images[:max_background]:
            bg_arr = img_to_array(bg_img) / 255.0
            bg.append(bg_arr)

        bg = np.array(bg)

        # Flatten images for KernelExplainer (it works with tabular data)
        img_shape = img.shape
        img_flat = img.reshape(1, -1)
        bg_flat = bg.reshape(bg.shape[0], -1)

        # Create prediction function wrapper
        def predict_fn(x):
            x_reshaped = x.reshape(-1, img_shape[1], img_shape[2], img_shape[3])
            if isinstance(model, tf.keras.layers.TFSMLayer):
                preds_dict = model(x_reshaped)
                preds = tf.nn.softmax(list(preds_dict.values())[0]).numpy()
            else:
                preds = tf.nn.softmax(model.predict(x_reshaped, verbose=0)).numpy()
            return preds

        # Create KernelExplainer
        print(f"Initializing SHAP KernelExplainer with {max_background} background samples...")
        explainer = shap.KernelExplainer(predict_fn, bg_flat)
        
        # Compute SHAP values
        print(f"Computing SHAP values with {nsamples} samples (this may take a moment)...")
        shap_values = explainer.shap_values(img_flat, nsamples=nsamples)
        

        shap_map = shap_values[class_idx][:, 0]
        
        
        shap_map = shap_map.reshape(img_shape[1], img_shape[2], img_shape[3])
        
        # Average across color channels
        shap_map = np.mean(shap_map, axis=-1)
        
        print("SHAP map range:", shap_map.min(), shap_map.max())
        
        # Normalize for visualization
        shap_map = (shap_map - shap_map.min()) / (shap_map.max() - shap_map.min() + 1e-8)
        # shap_map = np.abs(shap_map)
        
        class_label = (
            class_names[class_idx]
            if class_names and class_idx < len(class_names)
            else f"class {class_idx}"
        )

        # Create visualization
        fig, axs = plt.subplots(1, 2, figsize=(8, 4))
        axs[0].imshow(image)
        axs[0].set_title("Original image")

        axs[1].imshow(shap_map, cmap='RdBu')
        axs[1].set_title(f"SHAP Kernel - Predicted class: {class_label}")
        plt.tight_layout()

        return fig


class SHAP_DEEP_EXPLAINER:
    def explain(self, image, model, class_idx=None, background=None, num_samples=100):
        """
        Generate a SHAP explanation for a spectrogram/image using GradientExplainer (Keras-compatible).

        Args:
            image: Input spectrogram/image (np.array, HxW or CxHxW).
            model: Keras model (must output logits).
            class_idx: Target class index. If None, explains the predicted class.
            background: Background dataset for SHAP (np.array, NxHxWxC or NxCxHxW).
            num_samples: Number of samples for SHAP approximation.
        Returns:
            matplotlib.figure.Figure: Figure with original and explanation.
        """
        
        image = np.array(image, dtype=np.float32)
        
        # Preprocess image
        if image.ndim == 2:
            image = np.expand_dims(image, axis=(0, -1))  # Add batch and channel dims
        elif image.ndim == 3:
            image = np.expand_dims(image, axis=0)  # Add batch dim

        # Wrap model for SHAP
        def predict_fn(x):
            return model(x)

        # Initialize explainer
        if background is None:
            background = image  # Use input as background if none provided
        explainer = shap.GradientExplainer(predict_fn, background)

        # Compute SHAP values
        shap_values = explainer.shap_values(image, ranked_outputs=num_samples)

        # Select class (predicted if not specified)
        if class_idx is None:
            logits = model.predict(image)
            class_idx = np.argmax(logits)

        # Plot
        fig, axs = plt.subplots(1, 2, figsize=(8, 4))
        axs[0].imshow(image.squeeze(), cmap='viridis' if image.shape[-1] == 1 else None)
        axs[0].set_title("Original")
        axs[0].axis('off')

        shap.image_plot(shap_values[class_idx], image.squeeze(), show=False, ax=axs[1])
        axs[1].set_title(f"SHAP - Class {class_idx}")
        plt.tight_layout()

        return fig