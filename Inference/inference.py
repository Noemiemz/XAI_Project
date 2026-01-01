import numpy as np
import tensorflow as tf

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
