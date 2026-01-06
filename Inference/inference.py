import numpy as np
import tensorflow as tf
import torch
from torchvision import transforms
from PIL import Image

def predict_image(model, image_data, normalize=True):
    img = np.array(image_data)
    if normalize:
        img = img / 255.0

    img = np.expand_dims(img, axis=0)
    
    # Check if model is TFSMLayer (SavedModel format)
    if isinstance(model, tf.keras.layers.TFSMLayer):
        # TFSMLayer is called directly, not with .predict()
        preds_dict = model(img)
        # Extract the output tensor (it's usually a dict with one key)
        preds = list(preds_dict.values())[0].numpy()
    else:
        # Regular Keras model
        preds = model.predict(img)
    
    class_idx = int(np.argmax(preds, axis=1)[0])

    return {
        "predictions": preds,
        "class_idx": class_idx
    }

def predict_image_pytorch(model, image_data, device="cpu"):
    """Predict using PyTorch model for lung cancer detection"""
    # Convert PIL Image to tensor
    if isinstance(image_data, Image.Image):
        img = image_data
    else:
        img = Image.fromarray(np.uint8(image_data))
    
    # Convert to RGB if needed (handles L, RGBA, P, etc.)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    # Apply transforms (same as training)
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    img_tensor = transform(img).unsqueeze(0).to(device)
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.softmax(outputs, dim=1)
        class_idx = int(torch.argmax(probs, dim=1)[0])
        
    # Convert to numpy for consistency with Keras predictions
    preds = probs.cpu().numpy()
    
    return {
        "predictions": preds,
        "class_idx": class_idx
    }
