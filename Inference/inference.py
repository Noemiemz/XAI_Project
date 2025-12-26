import numpy as np

def predict_image(model, image_data, normalize=True):
    img = np.array(image_data)
    if normalize:
        img = img / 255.0

    img = np.expand_dims(img, axis=0)
    preds = model.predict(img)
    class_idx = int(np.argmax(preds, axis=1)[0])

    return {
        "predictions": preds,
        "class_idx": class_idx
    }
