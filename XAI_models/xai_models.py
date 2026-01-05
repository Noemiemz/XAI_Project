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

        # Find the last convolutional layer in the model
        last_conv_layer = None
        for layer in reversed(model.layers):
            if 'conv' in layer.name.lower():
                last_conv_layer = layer
                break
        
        if last_conv_layer is None:
            raise ValueError("Could not find a convolutional layer in the model")
        
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
    

class SHAP_GRADIENT:
    def explain(self, image, model, class_idx, background=None, num_samples=100):
        image = np.array(image, dtype=np.float32)
        
        # Preprocess image
        if image.ndim == 2:
            image = np.expand_dims(image, axis=(0, -1))  # Add batch and channel dims
        elif image.ndim == 3:
            image = np.expand_dims(image, axis=0)  # Add batch dim

        # create a fixed model to fix input shape issues
        input_shape = image.shape[1:]  # (height, width, channels)
        input_layer = tf.keras.Input(shape=input_shape, dtype=tf.float32)
        output = model(input_layer)
        fixed_model = tf.keras.Model(inputs=input_layer, outputs=output)
        _ = fixed_model(image)
        
        if background is None:
            print("\n3. Creating background dataset...")
            num_background = 5
            background_images = []
            
            for _ in range(num_background):
                noisy_image = image + np.random.randn(*image.shape) * 0.1
                noisy_image = np.clip(noisy_image, 0, 1)
                if image.ndim == 2:
                    background_images.append(np.expand_dims(noisy_image, axis=-1))
                else:
                    background_images.append(noisy_image)
            
            background = np.array(background_images)
            print(f"   [OK] Background dataset created with shape: {background.shape}")

        background = np.array(background, dtype=np.float32)
        
        # Ensure background has the right shape
        if background.ndim == 3:
            background = np.expand_dims(background, axis=0)
        
        
        explainer = shap.GradientExplainer(fixed_model, background)
        
        # Try different parameter combinations
        try:
            shap_values = explainer.shap_values(image)
        except TypeError as e:
            print("ERROOORRRRR, \nCaught TypeError, trying with nsamples parameter:", e)
            if "ranked_outputs" in str(e):
                shap_values = explainer.shap_values(image, nsamples=num_samples)
            else:
                raise e
        
        # Ensure class_idx is within bounds
        class_idx = min(class_idx, len(shap_values) - 1)

        # Plot
        fig, axs = plt.subplots(1, 2, figsize=(8, 4))
        axs[0].imshow(image.squeeze(), cmap='viridis' if image.shape[-1] == 1 else None)
        axs[0].set_title("Original")
        axs[0].axis('off')

        shap_map = shap_values[class_idx]
        
        if shap_map.ndim == 4:
            shap_map = shap_map.squeeze(axis=-1)
        
        if shap_map.ndim == 3 and shap_map.shape[-1] == 3:
            shap_map = np.mean(shap_map, axis=-1)
        
        # Normalize for visualization
        shap_map = (shap_map - shap_map.min()) / (shap_map.max() - shap_map.min() + 1e-8)
        
        axs[1].imshow(shap_map, cmap='RdBu')
        axs[1].set_title(f"SHAP - Class {class_idx}")
        axs[1].axis('off')
        plt.tight_layout()

        return fig