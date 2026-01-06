import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2

from lime import lime_image
from skimage.segmentation import mark_boundaries
from keras.preprocessing.image import img_to_array

import shap
from keras.applications.vgg16 import preprocess_input

import torch
import torch.nn.functional as F
from PIL import Image

class Lime:
    def explain(self, image, model, class_idx, class_names=None, num_samples=1000, num_features=8):
        # Check if this is a PyTorch model
        if isinstance(model, torch.nn.Module):
            return self._explain_pytorch(image, model, class_idx, class_names, num_samples, num_features)
        else:
            # TensorFlow/Keras model
            return self._explain_keras(image, model, class_idx, class_names, num_samples, num_features)
    
    def _explain_keras(self, image, model, class_idx, class_names=None, num_samples=1000, num_features=8):
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
        axs[0].imshow(image, cmap='gray')
        axs[0].set_title("Original image")

        axs[1].imshow(mark_boundaries(temp, mask))
        axs[1].set_title(f"LIME - Predicted class: {class_label}")
        plt.tight_layout()

        return(fig)
    
    def _explain_pytorch(self, image, model, class_idx, class_names=None, num_samples=1000, num_features=8):
        """LIME explanation for PyTorch models"""
        device = next(model.parameters()).device
        
        # Convert PIL Image to numpy array
        img_array = np.array(image)
        
        # Normalize to [0, 1]
        if img_array.dtype == np.uint8:
            img_normalized = img_array.astype(np.float32) / 255.0
        else:
            img_normalized = img_array.astype(np.float32)
        
        # Ensure 3 channels
        if img_normalized.ndim == 2:
            img_normalized = np.stack([img_normalized] * 3, axis=-1)
        elif img_normalized.shape[2] == 4:  # RGBA
            img_normalized = img_normalized[:, :, :3]
        
        # Create a prediction wrapper for PyTorch model
        def predict_fn(images):
            batch_preds = []
            model.eval()
            with torch.no_grad():
                for img in images:
                    # Convert numpy to PIL Image
                    if img.max() <= 1.0:
                        img_display = (img * 255).astype(np.uint8)
                    else:
                        img_display = img.astype(np.uint8)
                    
                    img_pil = Image.fromarray(img_display)
                    
                    # Convert to tensor and normalize
                    img_tensor = torch.from_numpy(np.array(img_pil).astype(np.float32).transpose(2, 0, 1) / 255.0).unsqueeze(0).to(device)
                    
                    # Normalize using ImageNet statistics
                    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
                    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
                    img_tensor = (img_tensor - mean) / std
                    
                    # Get predictions
                    output = model(img_tensor)
                    if isinstance(output, dict):
                        output = list(output.values())[0]
                    
                    batch_preds.append(output[0].cpu().numpy())
            
            return np.array(batch_preds)
        
        # Explaining the prediction
        explainer = lime_image.LimeImageExplainer()
        explanation = explainer.explain_instance(
            img_normalized.astype('float64'),
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
        
        # Visualization
        fig, axs = plt.subplots(1, 2, figsize=(8, 4))
        axs[0].imshow(image, cmap='gray')
        axs[0].set_title("Original image")
        axs[0].axis('off')
        
        axs[1].imshow(mark_boundaries(temp, mask))
        axs[1].set_title(f"LIME - Predicted class: {class_label}")
        axs[1].axis('off')
        plt.tight_layout()
        
        return fig

class GradCAM:
    def explain(self, image, model, class_idx, class_names=None):
        # Check if this is a PyTorch model
        if isinstance(model, torch.nn.Module):
            return self._explain_pytorch(image, model, class_idx, class_names)
        else:
            # TensorFlow/Keras model
            return self._explain_keras(image, model, class_idx, class_names)
    
    def _explain_keras(self, image, model, class_idx, class_names=None):
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
        axs[0].imshow(image, cmap='gray')
        axs[0].set_title("Original image")

        axs[1].imshow(superimposed_img)
        axs[1].set_title(f"Grad-CAM - Predicted class: {class_label}")
        plt.tight_layout()

        return(fig)
    
    def _explain_pytorch(self, image, model, class_idx, class_names=None):
        """Grad-CAM explanation for PyTorch models"""
        device = next(model.parameters()).device
        
        # Convert PIL Image to tensor
        if isinstance(image, Image.Image):
            img_array = np.array(image).astype(np.float32) / 255.0
        else:
            img_array = np.array(image).astype(np.float32) / 255.0
        
        # Ensure 3 channels
        if img_array.ndim == 2:
            img_array = np.stack([img_array] * 3, axis=-1)
        elif img_array.shape[2] == 4:  # RGBA
            img_array = img_array[:, :, :3]
        
        # Convert to tensor and normalize
        img_tensor = torch.from_numpy(img_array.transpose(2, 0, 1)).unsqueeze(0).to(device)
        
        # Normalize using ImageNet statistics
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
        img_tensor = (img_tensor - mean) / std
        
        # Enable gradient computation
        img_tensor.requires_grad = True
        
        # Register hook to get the feature map
        feature_maps = None
        def get_feature_maps(module, input, output):
            nonlocal feature_maps
            feature_maps = output.detach()
        
        # Find the last convolutional layer
        last_conv_layer = None
        for module in reversed(list(model.modules())):
            if isinstance(module, torch.nn.Conv2d):
                last_conv_layer = module
                break
        
        if last_conv_layer is None:
            raise ValueError("Could not find a convolutional layer in the model")
        
        # Register hook
        hook = last_conv_layer.register_forward_hook(get_feature_maps)
        
        # Forward pass - need gradients for Grad-CAM
        model.eval()
        output = model(img_tensor)
        
        # Get the scores
        if isinstance(output, dict):
            output = list(output.values())[0]
        
        scores = output[0].detach().cpu().numpy()
        
        # Backward pass to get gradients
        img_tensor.requires_grad = True
        output = model(img_tensor)
        if isinstance(output, dict):
            output = list(output.values())[0]
        
        # Get the target class score
        target_score = output[0, class_idx]
        model.zero_grad()
        target_score.backward()
        
        # Get gradients of the feature map - this is the correct approach
        # We need to get gradients of the feature maps, not the input
        
        # Clear previous hooks
        hook.remove()
        
        # Register new hooks for proper Grad-CAM computation
        feature_maps = None
        feature_grads = None
        
        def get_feature_maps(module, input, output):
            nonlocal feature_maps
            feature_maps = output.detach()
        
        def get_feature_grads(module, grad_input, grad_output):
            nonlocal feature_grads
            feature_grads = grad_output[0].detach()
        
        # Register hooks
        hook_fwd = last_conv_layer.register_forward_hook(get_feature_maps)
        hook_bwd = last_conv_layer.register_backward_hook(get_feature_grads)
        
        # Forward pass - need gradients for Grad-CAM
        model.eval()
        output = model(img_tensor)
        
        # Backward pass to get gradients
        if isinstance(output, dict):
            output = list(output.values())[0]
        target_score = output[0, class_idx]
        model.zero_grad()
        target_score.backward()
        
        # Remove hooks
        hook_fwd.remove()
        hook_bwd.remove()
        
        # Compute pooled gradients
        if feature_grads is not None:
            pooled_grads = torch.mean(feature_grads, dim=[0, 2, 3])
        else:
            # Fallback: this shouldn't happen if the model has conv layers
            raise ValueError("Could not get feature map gradients")
        
        # Weighted combination
        batch_size, num_channels, h, w = feature_maps.shape
        heatmap = torch.zeros(h, w, device=device)
        
        for i in range(num_channels):
            heatmap += pooled_grads[i] * feature_maps[0, i, :, :]
        
        heatmap = torch.relu(heatmap)
        heatmap = heatmap / (torch.max(heatmap) + 1e-8)
        
        # Resize heatmap to match input image size
        heatmap_np = heatmap.cpu().numpy()
        heatmap_np = cv2.resize(heatmap_np, (img_array.shape[1], img_array.shape[0]))
        
        # Apply colormap
        heatmap_np = np.uint8(255 * heatmap_np)
        heatmap_colored = cv2.applyColorMap(heatmap_np, cv2.COLORMAP_JET)
        
        # Denormalize image for visualization
        img_display = img_array.copy()
        img_display = np.uint8(255 * img_display)
        if img_display.shape[2] == 3:
            img_display = cv2.cvtColor(img_display, cv2.COLOR_RGB2BGR)
        
        # Superimpose
        superimposed_img = cv2.addWeighted(img_display, 0.6, heatmap_colored, 0.4, 0)
        
        class_label = (
            class_names[class_idx]
            if class_names and class_idx < len(class_names)
            else f"class {class_idx}"
        )
        
        # Visualization
        fig, axs = plt.subplots(1, 2, figsize=(8, 4))
        axs[0].imshow(image, cmap='gray')
        axs[0].set_title("Original image")
        axs[0].axis('off')
        
        axs[1].imshow(cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB))
        axs[1].set_title(f"Grad-CAM - Predicted class: {class_label}")
        axs[1].axis('off')
        plt.tight_layout()
        
        hook.remove()
        
        return fig
    
class SHAP_GRADIENT:
    def explain(self, image, model, class_idx, class_names=None, background=None, num_samples=100):
        # Check if this is a PyTorch model
        if isinstance(model, torch.nn.Module):
            return self._explain_pytorch(image, model, class_idx, class_names, background, num_samples)
        else:
            # TensorFlow/Keras model
            return self._explain_keras(image, model, class_idx, class_names, background, num_samples)
    
    def _explain_keras(self, image, model, class_idx, class_names=None, background=None, num_samples=100):
        image = np.array(image, dtype=np.float32) / 255.0
        
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
        axs[0].imshow(image.squeeze(), cmap='gray')
        axs[0].set_title("Original")
        axs[0].axis('off')

        shap_map = shap_values[class_idx]
        
        if shap_map.ndim == 4:
            shap_map = shap_map.squeeze(axis=-1)
        
        if shap_map.ndim == 3 and shap_map.shape[-1] == 3:
            shap_map = np.mean(shap_map, axis=-1)
        
        # Normalize for visualization
        shap_map = (shap_map - shap_map.min()) / (shap_map.max() - shap_map.min() + 1e-8)
        
        class_label = (
            class_names[class_idx]
            if class_names and class_idx < len(class_names)
            else f"class {class_idx}"
        )
        
        axs[1].imshow(shap_map, cmap='jet')
        axs[1].set_title(f"SHAP - Class {class_idx}")
        axs[1].axis('off')
        plt.tight_layout()

        return fig
    
    def _explain_pytorch(self, image, model, class_idx, class_names=None, background=None, num_samples=100):
        """SHAP Gradient Explainer for PyTorch models"""
        device = next(model.parameters()).device
        
        # Convert PIL Image to numpy array
        img_array = np.array(image).astype(np.float32) / 255.0
        
        # Ensure 3 channels
        if img_array.ndim == 2:
            img_array = np.stack([img_array] * 3, axis=-1)
        elif img_array.shape[2] == 4:  # RGBA
            img_array = img_array[:, :, :3]
        
        # Convert to tensor and normalize
        img_tensor = torch.from_numpy(img_array.transpose(2, 0, 1)).unsqueeze(0).to(device)
        
        # Normalize using ImageNet statistics
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
        img_tensor = (img_tensor - mean) / std
        
        # Create background if not provided
        if background is None:
            print("\n3. Creating background dataset...")
            num_background = 5
            background_images = []
            
            for _ in range(num_background):
                noisy_image = img_array + np.random.randn(*img_array.shape) * 0.1
                noisy_image = np.clip(noisy_image, 0, 1)
                background_images.append(noisy_image)
            
            background = np.array(background_images)
            print(f"   [OK] Background dataset created with shape: {background.shape}")
        
        # For PyTorch models, SHAP integration is complex, so we'll use a simpler gradient-based approach
        # This provides similar functionality to SHAP gradient explainer
        return self._basic_gradient_attribution(image, model, class_idx, class_names, device)
    
    def _basic_gradient_attribution(self, image, model, class_idx, class_names=None, device=None):
        """Fallback method: Basic gradient attribution for PyTorch models"""
        # Convert PIL Image to tensor
        img_array = np.array(image).astype(np.float32) / 255.0
        
        # Ensure 3 channels
        if img_array.ndim == 2:
            img_array = np.stack([img_array] * 3, axis=-1)
        elif img_array.shape[2] == 4:  # RGBA
            img_array = img_array[:, :, :3]
        
        # Convert to tensor and normalize
        img_tensor = torch.from_numpy(img_array.transpose(2, 0, 1)).unsqueeze(0).to(device)
        
        # Normalize using ImageNet statistics
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
        img_tensor = (img_tensor - mean) / std
        
        # Enable gradient computation
        img_tensor.requires_grad = True
        
        # Forward pass
        model.eval()
        output = model(img_tensor)
        
        if isinstance(output, dict):
            output = list(output.values())[0]
        
        # Get the target class score
        target_score = output[0, class_idx]
        
        # Backward pass to get gradients
        model.zero_grad()
        target_score.backward()
        
        # Get gradients
        gradients = img_tensor.grad.data.cpu().numpy()
        gradients = np.abs(gradients)  # Take absolute values
        gradients = np.mean(gradients, axis=1)  # Average over channels
        gradients = gradients.squeeze()  # Remove batch dimension
        
        # Normalize
        gradients = (gradients - gradients.min()) / (gradients.max() - gradients.min() + 1e-8)
        
        class_label = (
            class_names[class_idx]
            if class_names and class_idx < len(class_names)
            else f"class {class_idx}"
        )
        
        # Visualization
        fig, axs = plt.subplots(1, 2, figsize=(8, 4))
        axs[0].imshow(image)
        axs[0].set_title("Original image")
        axs[0].axis('off')
        
        axs[1].imshow(gradients, cmap='jet')
        axs[1].set_title(f"Gradient Attribution - Predicted class: {class_label}")
        axs[1].axis('off')
        plt.tight_layout()
        
        return fig

class OcclusionSensitivity:
    def explain(self, image, model, class_idx, class_names=None, patch_size=32, stride=8, occlusion_value=0.0):
        # Check if this is a PyTorch model
        if isinstance(model, torch.nn.Module):
            return self._explain_pytorch(image, model, class_idx, class_names, patch_size, stride, occlusion_value)
        else:
            # TensorFlow/Keras model
            return self._explain_keras(image, model, class_idx, class_names, patch_size, stride, occlusion_value)
    
    def _explain_keras(self, image, model, class_idx, class_names=None, patch_size=32, stride=8, occlusion_value=0.0):
        img = np.array(image) / 255.0
        h, w, c = img.shape

        # Prediction wrapper
        def predict_fn(x):
            if isinstance(model, tf.keras.layers.TFSMLayer):
                preds = list(model(x).values())[0].numpy()
            else:
                preds = model.predict(x)
            return preds

        # Original prediction
        original_pred = predict_fn(img[np.newaxis, ...])[0, class_idx]

        heatmap = np.zeros((h, w))

        for y in range(0, h - patch_size, stride):
            for x in range(0, w - patch_size, stride):
                occluded = img.copy()
                occluded[y:y+patch_size, x:x+patch_size, :] = occlusion_value

                pred = predict_fn(occluded[np.newaxis, ...])[0, class_idx]
                delta = original_pred - pred

                heatmap[y:y+patch_size, x:x+patch_size] += delta

        heatmap = np.maximum(heatmap, 0)
        heatmap /= (heatmap.max() + 1e-8)
        heatmap_smooth = cv2.GaussianBlur(heatmap, (15, 15), 0)

        class_label = (
            class_names[class_idx]
            if class_names and class_idx < len(class_names)
            else f"class {class_idx}"
        )

        fig, axs = plt.subplots(1, 2, figsize=(8, 4))

        axs[0].imshow(image, cmap='gray')
        axs[0].set_title("Original image")
        axs[0].axis('off')

        axs[1].imshow(image, cmap='gray')
        axs[1].imshow(heatmap_smooth, cmap='jet', alpha=0.5)
        axs[1].set_title(f"Occlusion - Predicted class: {class_label}")
        axs[1].axis('off')

        plt.tight_layout()
        return fig
    
    def _explain_pytorch(self, image, model, class_idx, class_names=None, patch_size=32, stride=8, occlusion_value=0.0):
        """Occlusion Sensitivity for PyTorch models"""
        device = next(model.parameters()).device
        
        # Convert PIL Image to numpy array
        img_array = np.array(image).astype(np.float32) / 255.0
        
        # Ensure 3 channels
        if img_array.ndim == 2:
            img_array = np.stack([img_array] * 3, axis=-1)
        elif img_array.shape[2] == 4:  # RGBA
            img_array = img_array[:, :, :3]
        
        h, w, c = img_array.shape
        
        # Normalize using ImageNet statistics
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
        
        # Prediction wrapper for PyTorch
        def predict_fn(x):
            """Convert numpy array to tensor and get predictions"""
            # x should be in format (batch, height, width, channels)
            x_tensor = torch.from_numpy(x).to(device).float()
            x_tensor = x_tensor.permute(0, 3, 1, 2)  # Convert to (batch, channels, height, width)
            x_tensor = (x_tensor - mean) / std
            
            model.eval()
            with torch.no_grad():
                output = model(x_tensor)
                if isinstance(output, dict):
                    output = list(output.values())[0]
            
            return output.cpu().numpy()
        
        # Original prediction
        original_pred = predict_fn(img_array[np.newaxis, ...])[0, class_idx]

        heatmap = np.zeros((h, w))

        for y in range(0, h - patch_size, stride):
            for x in range(0, w - patch_size, stride):
                occluded = img_array.copy()
                occluded[y:y+patch_size, x:x+patch_size, :] = occlusion_value

                pred = predict_fn(occluded[np.newaxis, ...])[0, class_idx]
                delta = original_pred - pred

                heatmap[y:y+patch_size, x:x+patch_size] += delta

        heatmap = np.maximum(heatmap, 0)
        heatmap /= (heatmap.max() + 1e-8)
        heatmap_smooth = cv2.GaussianBlur(heatmap, (15, 15), 0)

        class_label = (
            class_names[class_idx]
            if class_names and class_idx < len(class_names)
            else f"class {class_idx}"
        )

        fig, axs = plt.subplots(1, 2, figsize=(8, 4))

        axs[0].imshow(image, cmap='gray')
        axs[0].set_title("Original image")
        axs[0].axis('off')

        axs[1].imshow(image, cmap='gray')
        axs[1].imshow(heatmap_smooth, cmap='jet', alpha=0.5)
        axs[1].set_title(f"Occlusion Sensitivity - Predicted class: {class_label}")
        axs[1].axis('off')

        plt.tight_layout()
        return fig

class IntegratedGradients:
    def explain(self, image, model, class_idx, class_names=None, baseline=None, steps=50):
        # Check if this is a PyTorch model
        if isinstance(model, torch.nn.Module):
            return self._explain_pytorch(image, model, class_idx, class_names, baseline, steps)
        else:
            # TensorFlow/Keras model
            return self._explain_keras(image, model, class_idx, class_names, baseline, steps)
    
    def _explain_keras(self, image, model, class_idx, class_names=None, baseline=None, steps=50):
        img = np.array(image) / 255.0
        img = tf.convert_to_tensor(img, dtype=tf.float32)

        if baseline is None:
            baseline = tf.zeros_like(img)
        else:
            baseline = tf.convert_to_tensor(baseline, dtype=tf.float32)

        img = tf.expand_dims(img, axis=0)
        baseline = tf.expand_dims(baseline, axis=0)

        alphas = tf.linspace(0.0, 1.0, steps)

        def predict(x):
            if isinstance(model, tf.keras.layers.TFSMLayer):
                return list(model(x).values())[0]
            return model(x)

        integrated_grads = tf.zeros_like(img)

        for alpha in alphas:
            interpolated = baseline + alpha * (img - baseline)

            with tf.GradientTape() as tape:
                tape.watch(interpolated)
                preds = predict(interpolated)
                target = preds[:, class_idx]

            grads = tape.gradient(target, interpolated)
            integrated_grads += grads

        integrated_grads /= steps
        attributions = (img - baseline) * integrated_grads
        attributions = tf.reduce_mean(attributions, axis=-1)[0].numpy()

        attributions = np.maximum(attributions, 0)
        attributions /= (attributions.max() + 1e-8)

        class_label = (
            class_names[class_idx]
            if class_names and class_idx < len(class_names)
            else f"class {class_idx}"
        )

        fig, axs = plt.subplots(1, 2, figsize=(8, 4))

        axs[0].imshow(image, cmap='gray')
        axs[0].set_title("Original image")
        axs[0].axis('off')

        axs[1].imshow(np.zeros_like(attributions), cmap='gray')
        axs[1].imshow(attributions, cmap="jet", vmin=0, vmax=1)
        axs[1].set_title(f"Integrated Gradients - Predicted class: {class_label}")
        axs[1].axis('off')

        plt.tight_layout()
        return fig
    
    def _explain_pytorch(self, image, model, class_idx, class_names=None, baseline=None, steps=50):
        """Integrated Gradients for PyTorch models"""
        device = next(model.parameters()).device
        
        # Convert PIL Image to numpy array
        img_array = np.array(image).astype(np.float32) / 255.0
        
        # Ensure 3 channels
        if img_array.ndim == 2:
            img_array = np.stack([img_array] * 3, axis=-1)
        elif img_array.shape[2] == 4:  # RGBA
            img_array = img_array[:, :, :3]
        
        # Convert to tensor and normalize
        img_tensor = torch.from_numpy(img_array.transpose(2, 0, 1)).unsqueeze(0).to(device)
        
        # Normalize using ImageNet statistics
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
        img_tensor = (img_tensor - mean) / std
        
        # Create baseline (black image)
        if baseline is None:
            baseline = torch.zeros_like(img_tensor)
        else:
            baseline_array = np.array(baseline).astype(np.float32) / 255.0
            if baseline_array.ndim == 2:
                baseline_array = np.stack([baseline_array] * 3, axis=-1)
            elif baseline_array.shape[2] == 4:  # RGBA
                baseline_array = baseline_array[:, :, :3]
            baseline = torch.from_numpy(baseline_array.transpose(2, 0, 1)).unsqueeze(0).to(device)
            baseline = (baseline - mean) / std
        
        # Create alphas for interpolation
        alphas = torch.linspace(0.0, 1.0, steps, device=device)
        
        # Compute integrated gradients
        integrated_grads = torch.zeros_like(img_tensor)
        
        model.eval()
        for alpha in alphas:
            # Interpolate between baseline and input
            interpolated = baseline + alpha * (img_tensor - baseline)
            interpolated.requires_grad = True
            
            # Forward pass
            output = model(interpolated)
            if isinstance(output, dict):
                output = list(output.values())[0]
            
            # Get the target class score
            target_score = output[0, class_idx]
            
            # Backward pass to get gradients
            model.zero_grad()
            target_score.backward()
            
            # Accumulate gradients
            integrated_grads += interpolated.grad.data
        
        # Average gradients
        integrated_grads /= steps
        
        # Compute attributions
        attributions = (img_tensor - baseline) * integrated_grads
        attributions = attributions.mean(dim=1).squeeze().cpu().numpy()  # Average over channels
        
        # Take absolute values and normalize
        attributions = np.abs(attributions)
        attributions = np.maximum(attributions, 0)
        attributions /= (attributions.max() + 1e-8)
        
        class_label = (
            class_names[class_idx]
            if class_names and class_idx < len(class_names)
            else f"class {class_idx}"
        )
        
        # Visualization
        fig, axs = plt.subplots(1, 2, figsize=(8, 4))
        axs[0].imshow(image, cmap='gray')
        axs[0].set_title("Original image")
        axs[0].axis('off')
        
        axs[1].imshow(np.zeros_like(attributions), cmap='gray')
        axs[1].imshow(attributions, cmap="jet", vmin=0, vmax=1)
        axs[1].set_title(f"Integrated Gradients - Predicted class: {class_label}")
        axs[1].axis('off')
        
        plt.tight_layout()
        return fig