"""
Simple test script for SHAP DeepExplainer
Tests the SHAP_DEEP_EXPLAINER class with a sample spectrogram/image
"""

import librosa
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.models import load_model
from keras.preprocessing.image import load_img
import sys
import os


from XAI_models.xai_models import SHAP_DEEP_EXPLAINER


def create_sample_spectrogram(height=224, width=224):
    """
    Create a synthetic spectrogram for testing
    """
    # Create a sample spectrogram with some patterns
    x = np.linspace(0, 10, width)
    y = np.linspace(0, 10, height)
    X, Y = np.meshgrid(x, y)
    
    # Create some wave-like patterns
    spectrogram = np.sin(X) * np.cos(Y) + np.random.randn(height, width) * 0.1
    
    # Normalize to [0, 1]
    spectrogram = (spectrogram - spectrogram.min()) / (spectrogram.max() - spectrogram.min())
    
    return spectrogram

def load_background_spectrograms(folder="audio_files/specs", max_images=20):
    images = []
    for file in os.listdir(folder):
        if file.endswith(".png"):
            img = load_img(
                os.path.join(folder, file),
                target_size=(224, 224)
            )
            images.append(img)
        if len(images) >= max_images:
            break
    return images



def create_spectrogram(sound):
    audio_file = os.path.join('audio_files/', sound)

    spec_dir = os.path.join("audio_files", "specs")
    os.makedirs(spec_dir, exist_ok=True)

    base_name = os.path.splitext(sound)[0]
    spec_path = os.path.join(spec_dir, f"{base_name}_aaaaaaaaaaaaaaaaaa_spec.png")

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

    y, sr = librosa.load(audio_file)
    ms = librosa.feature.melspectrogram(y=y, sr=sr)
    log_ms = librosa.power_to_db(ms, ref=np.max)
    librosa.display.specshow(log_ms, sr=sr)
    plt.savefig(spec_path)
    image_data = load_img(spec_path ,target_size=(224,224))
    return image_data


def test_deep_explainer():
    """
    Test the SHAP_DEEP_EXPLAINER with a sample model and spectrogram
    """
    print("=" * 60)
    print("Testing SHAP DeepExplainer")
    print("=" * 60)
    
    # 1. Load a pre-trained model
    print("\n1. Loading model...")
    model_path = "Streamlit/saved_models/MobileNet_audio_classifier.h5"
    
    try:
        model = load_model(model_path)
        print(f"   [OK] Model loaded successfully from {model_path}")
        print(f"   Model input shape: {model.input_shape}")
        print(f"   Model output shape: {model.output_shape}")
    except Exception as e:
        print(f"   [ERROR] Error loading model: {e}")
        return
    
    # 2. Load sound and create spectrogram
    print("\n2. Creating sample spectrogram...")
    sound_file = "audio_1.wav" 
    try:
        sample_image = create_spectrogram(sound_file)
        sample_image = np.array(sample_image) / 255.0  # Normalize to [0, 1]
        # plt.imshow(sample_image)
        # plt.axis('off')
        print(f"   [OK] Spectrogram created with shape: {sample_image.shape}")
    except Exception as e:
        print(f"   [ERROR] Error creating spectrogram: {e}")
        return
    
    # 3. Create background data (simplified - using the same image with noise)
    # print("\n3. Creating background dataset...")
    # num_background = 5
    # background_images = []
    
    # for i in range(num_background):
    #     noisy_image = sample_image + np.random.randn(*sample_image.shape) * 0.1
    #     noisy_image = np.clip(noisy_image, 0, 1)
    #     if sample_image.ndim == 2:
    #         background_images.append(np.expand_dims(noisy_image, axis=-1))
    #     else:
    #         background_images.append(noisy_image)
    
    # background = np.array(background_images)
    
    print("\n3. Loading background dataset from folder...")
    background_images = load_background_spectrograms(folder="audio_files/specs", max_images=20)
    background = np.array([np.array(img) / 255.0 for img in background_images]) 
    print(f"   [OK] Background dataset created with shape: {background.shape}")
    
    # 4. Get model prediction
    print("\n4. Getting model prediction...")
    img_batch = np.expand_dims(sample_image, axis=0)
    predictions = model.predict(img_batch, verbose=0)
    predicted_class = np.argmax(predictions[0])
    confidence = predictions[0][predicted_class]
    
    print(f"   [OK] Predicted class: {predicted_class}")
    print(f"   [OK] Confidence: {confidence:.4f}")
    print(f"   Predictions: {predictions[0]}")
    
    # 5. Initialize DeepExplainer
    print("\n5. Initializing SHAP DeepExplainer...")
    explainer = SHAP_DEEP_EXPLAINER()
    print("   [OK] DeepExplainer initialized")
    
    # 6. Generate explanation
    print("\n6. Generating SHAP explanation...")
    print("   (This may take a moment...)")
    
    try:
        fig = explainer.explain(
            image=sample_image,
            model=model,
            class_idx=predicted_class,
            background=background,
            num_samples=50  # Reduced for faster testing
        )
        print("   [OK] Explanation generated successfully!")
        
        # Save the figure
        output_path = "test_deep_explainer_output.png"
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\n   [OK] Visualization saved to: {output_path}")
        
        # Show the plot
        plt.title("SHAP DeepExplainer Explanation")
        plt.show()
        
    except Exception as e:
        print(f"   [ERROR] Error generating explanation: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("\n" + "=" * 60)
    print("Test completed successfully! [OK]")
    print("=" * 60)


if __name__ == "__main__":
    test_deep_explainer()
