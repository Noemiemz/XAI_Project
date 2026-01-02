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
    

class SHAP:
    def explain(self, image, model, class_idx, background_images, class_names=None, max_background=20):
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
        if len(bg.shape) == 3:
            bg = np.expand_dims(bg, axis=0)

        # Handle different model output formats
        def create_wrapper_model(base_model):
            """Create a Keras model wrapper that handles TFSMLayer and dict outputs"""
            # Check if it's a TFSMLayer
            if isinstance(base_model, tf.keras.layers.TFSMLayer):
                # Create a proper Keras model that wraps the TFSMLayer
                input_layer = tf.keras.Input(shape=(224, 224, 3))
                
                # Define the output processing
                def call_model(x):
                    preds = base_model(x)
                    if isinstance(preds, dict):
                        return list(preds.values())[0]
                    return preds
                
                output = tf.keras.layers.Lambda(call_model)(input_layer)
                wrapper_model = tf.keras.Model(inputs=input_layer, outputs=output)
                return wrapper_model
            else:
                # Regular Keras model - try to get logits
                try:
                    logit_model = tf.keras.Model(
                        inputs=base_model.input,
                        outputs=base_model.layers[-2].output
                    )
                    return logit_model
                except Exception:
                    # Return the model as-is if we can't extract logits
                    return base_model

        logit_model = create_wrapper_model(model)

        # Use SHAP GradientExplainer
        explainer = shap.GradientExplainer(logit_model, bg)
        print("Computing SHAP values...")
        shap_values = explainer.shap_values(img)
        
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

