import sys
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from pyparsing import col
import streamlit as st
import numpy as np
import librosa
import librosa.display, os
import matplotlib.pyplot as plt
from keras.preprocessing.image import load_img
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image
import torch
from torchvision import models
import torch.nn as nn

from XAI_models.xai_models import Lime, GradCAM, SHAP_GRADIENT, OcclusionSensitivity, IntegratedGradients
from Inference.inference import predict_image, predict_image_pytorch

ALLOWED_FILE_TYPES = {
    "audio": [".wav", ".mp3"],
    "image": [".png", ".jpg", ".jpeg"]
}

st.set_page_config(
    page_title="XAI Platform",
    page_icon="img/robot.gif",
    layout="wide"
)

class_names_audio_deepfakes = ['real','fake']
class_names_lung_cancer = ['benign', 'malignant']

# Available models
AVAILABLE_MODELS_AUDIO = {
    "MobileNet": "models/xai_audioclassifiers/saved_models/MobileNet_audio_classifier.h5",
    "InceptionV3": "models/xai_audioclassifiers/saved_models/InceptionV3_audio_classifier.h5",
    "VGG16": "models/xai_audioclassifiers/saved_models/VGG16_audio_classifier.h5",
    "ResNet50": "models/xai_audioclassifiers/saved_models/ResNet50_audio_classifier.h5",
}

AVAILABLE_MODELS_LUNG = {
    "AlexNet": "models/Lung_Cancer_Detection/AlexNet_weights.pth",
    "DenseNet": "models/Lung_Cancer_Detection/DenseNet_weights.pth",
}

XAI_METHODS_AUTHORIZED_TYPE_OF_MODELS = {
    "LIME": ["ALL"],
    "GradCAM": ["CNN"],
    "SHAP_GRADIENT": ["ALL"],
    "OcclusionSensitivity": ["CNN"],
    "IntegratedGradients": ["ALL"],
}


MODELS_TYPE = { 
    "MobileNet": "CNN",
    "InceptionV3": "CNN",
    "VGG16": "CNN",
    "ResNet50": "CNN",
    "AlexNet": "CNN",
    "DenseNet": "CNN",
}


if "prediction_done" not in st.session_state:
    st.session_state.prediction_done = False

if "lung_prediction_done" not in st.session_state:
    st.session_state.lung_prediction_done = False

if "audio_selected_model" not in st.session_state:
    st.session_state.audio_selected_model = list(AVAILABLE_MODELS_AUDIO.keys())[0]

if "lung_selected_model" not in st.session_state:
    st.session_state.lung_selected_model = list(AVAILABLE_MODELS_LUNG.keys())[0]


def get_available_xai_methods(model_name):
    """Return list of XAI methods available for the selected model"""
    model_type = MODELS_TYPE.get(model_name, "CNN")
    available_methods = {}
    
    for method, authorized_types in XAI_METHODS_AUTHORIZED_TYPE_OF_MODELS.items():
        if "ALL" in authorized_types or model_type in authorized_types:
            available_methods[method] = True
        else:
            available_methods[method] = False
    
    return available_methods

def is_file_allowed(filename, allowed_types):
    file_type= os.path.splitext(filename)[1]
    
    all_types = {ext for ext_list in allowed_types.values() for ext in ext_list}

    return file_type in all_types


def get_file_category(filename, allowed_types):
    ext = os.path.splitext(filename)[1].lower()
    for category, exts in allowed_types.items():
        if ext in exts:
            return category
    return None

def cleanup_old_files(filename):
    """Delete old audio file and its spectrogram"""
    if filename:
        # Delete audio file
        audio_path = os.path.join('audio_files/', filename)
        if os.path.exists(audio_path):
            try:
                os.remove(audio_path)
            except Exception:
                pass
        
        # Delete spectrogram
        base_name = os.path.splitext(filename)[0]
        spec_path = os.path.join("audio_files", "specs", f"{base_name}_spec.png")
        if os.path.exists(spec_path):
            try:
                os.remove(spec_path)
            except Exception:
                pass

def load_pytorch_model(model_path, device="cpu"):
    """Load a PyTorch model for lung cancer detection"""
    if model_path.endswith("AlexNet_weights.pth"):
        model = models.alexnet(pretrained=False)
        num_features = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_features, 2)
    elif model_path.endswith("DenseNet_weights.pth"):
        model = models.densenet121(pretrained=False)
        num_features = model.classifier.in_features
        model.classifier = nn.Linear(num_features, 2)
    else:
        raise ValueError(f"Unknown model type from path: {model_path}")
    
    # Load weights
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def save_audio_file(sound_file):
    # save your sound file in the right folder by following the path
    with open(os.path.join('audio_files/', sound_file.name),'wb') as f:
         f.write(sound_file.getbuffer())
    return sound_file.name

def create_spectrogram(sound):
    audio_file = os.path.join('audio_files/', sound)

    spec_dir = os.path.join("audio_files", "specs")
    os.makedirs(spec_dir, exist_ok=True)

    base_name = os.path.splitext(sound)[0]
    spec_path = os.path.join(spec_dir, f"{base_name}_spec.png")

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

    y, sr = librosa.load(audio_file)
    ms = librosa.feature.melspectrogram(y=y, sr=sr)
    log_ms = librosa.power_to_db(ms, ref=np.max)
    librosa.display.specshow(log_ms, sr=sr)
    plt.savefig(spec_path)
    image_data = load_img(spec_path ,target_size=(224,224))
    st.image(image_data)
    st.caption("Mel-Spectrogram (used as CNN input)")
    return image_data



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


def main():
    st.title('Explainable AI Platform')
    tab_audio_deepfake, tab_lung_cancer, tab_about = st.tabs(
        ["Audio Deepfake Detection", "Lung Cancer Detection", "About"]
    )

    with tab_audio_deepfake:
        audio_pipeline()

    with tab_lung_cancer:
        lung_cancer_pipeline()

    with tab_about:
        about()


def about():
    st.title("About the Project")




def audio_pipeline():

    st.title("Audio Deepfake Detection with XAI")
    st.markdown("Upload an audio file to detect if it's **real** or **fake** and understand the model's decision with Explainability AI (XAI).")

    st.divider()

    with st.expander("Model Settings", expanded=True):
        selected_audio_model = st.selectbox(
            "Choose a model to classify your audio:",
            options=list(AVAILABLE_MODELS_AUDIO.keys()),
            index=list(AVAILABLE_MODELS_AUDIO.keys()).index(st.session_state.audio_selected_model)
        )
        
        # Reset prediction if model changes
        if selected_audio_model != st.session_state.audio_selected_model:
            st.session_state.audio_selected_model = selected_audio_model
            st.session_state.prediction_done = False

    st.divider()

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("### Upload Audio")
        with st.container(border=True):
            uploaded_file = st.file_uploader(
                "Choose an audio file",
                type=['wav','mp3'],
                help="Only .wav and .mp3 files are supported"
            )

        if not uploaded_file:
            # Clean up old file if user removed it
            if st.session_state.get("last_file"):
                # cleanup_old_files(st.session_state.last_file)
                st.session_state.last_file = None
                st.session_state.prediction_done = False
            st.info("Please upload a file to begin.")
            return

        if uploaded_file:
            if not is_file_allowed(uploaded_file.name, {"audio": ALLOWED_FILE_TYPES["audio"]}):
                st.error("This file type is not allowed. Please upload a .wav or .mp3 file.")
                return
            
            audio_bytes = uploaded_file.read()
            st.audio(audio_bytes, format="audio/wav")

        if uploaded_file is not None:
            if st.session_state.get("last_file") != uploaded_file.name:
                # Clean up old file when a new file is uploaded
                # if st.session_state.get("last_file"):
                #     cleanup_old_files(st.session_state.last_file)
                st.session_state.last_file = uploaded_file.name
                st.session_state.prediction_done = False

    if uploaded_file:
        with col2:
            st.markdown("### Spectrogram")

            with st.spinner("Generating Spectrogram..."):
                save_audio_file(uploaded_file)
                spec = create_spectrogram(uploaded_file.name)
            # Prediction
            if not st.session_state.prediction_done:
                with st.spinner("Analyzing audio..."):
                    model_path = AVAILABLE_MODELS_AUDIO[st.session_state.audio_selected_model]
                    if model_path.endswith('.h5'):
                        model = tf.keras.models.load_model(model_path)
                    else:
                        model = tf.keras.layers.TFSMLayer(model_path, call_endpoint='serving_default')
                    output = predict_image(model, spec)

                    st.session_state.model = model
                    st.session_state.output = output
                    st.session_state.class_label = output["class_idx"]
                    st.session_state.prediction = output["predictions"]
                    st.session_state.prediction_done = True

            class_label = st.session_state.class_label
            prediction = st.session_state.prediction
            
            # For sigmoid output: prediction[0][0] is probability of "fake"
            fake_probability = float(prediction[0][0])

            st.subheader("Prediction Result")

            col_a, col_b = st.columns([2, 1])

            with col_a:
                if class_label == 1:  # fake
                    confidence = fake_probability
                    st.error("The audio is **Fake**")
                else:  # real
                    confidence = 1 - fake_probability
                    st.success("The audio is **Real**")
            
            with col_b:
                st.metric(
                    label="Confidence",
                    value=f"{confidence:.1%}"
                )

        # XAI
        st.divider()

        if not st.session_state.prediction_done:
            st.info("Run a prediction to enable Explainability.")
            return
        
        st.markdown("## Explainability")

        st.markdown("**Select the XAI methods you want to try:**")
        available_methods = get_available_xai_methods(st.session_state.audio_selected_model)
        
        col_lime, col_gradcam, col_shap, col_occlu, col_int_grad  = st.columns(5, width=800, border=True)
        
        with col_lime:
            use_lime = st.checkbox("LIME", value=True, disabled=not available_methods["LIME"])
        with col_gradcam:
            use_gradcam = st.checkbox("Grad-CAM", value=False, disabled=not available_methods["GradCAM"])
        with col_shap:
            use_shap = st.checkbox("SHAP Gradient", value=False, disabled=not available_methods["SHAP_GRADIENT"])
        with col_occlu:
            use_occlu = st.checkbox("Occlusion Sensitivity", value=False, disabled=not available_methods["OcclusionSensitivity"])
        with col_int_grad:
            use_int_grad = st.checkbox("Integrated Gradients", value=False, disabled=not available_methods["IntegratedGradients"])
        
        if (use_lime or use_gradcam or use_shap) and st.button("Run Explainability"):

            if use_lime:
                st.markdown("### LIME Explanation")
                with st.spinner("Generating LIME results..."):
                    try:
                        fig_lime = Lime().explain(
                            image=spec,
                            model=st.session_state.model,
                            class_idx=st.session_state.class_label,
                            class_names=class_names_audio_deepfakes
                        )
                        with st.expander("LIME Results"):
                            st.pyplot(fig_lime, width='content')
                    except Exception as e:
                        st.error(f"⚠️ LIME Error: {str(e)[:100]}")

            if use_gradcam:
                st.markdown("### Grad-CAM Explanation")
                with st.spinner("Generating Grad-CAM results..."):
                    try:
                        fig_grad = GradCAM().explain(
                            image=spec,
                            model=st.session_state.model,
                            class_idx=st.session_state.class_label,
                            class_names=class_names_audio_deepfakes
                        )
                        with st.expander("Grad-CAM Results"):
                            st.pyplot(fig_grad, width='content')
                    except Exception as e:
                        st.error(f"⚠️ Grad-CAM Error: {str(e)}")

            if use_shap:
                st.markdown("### SHAP Gradient Explanation")
                with st.spinner("Generating SHAP Gradient results..."):
                        # Load background images for SHAP (sample from training data)
                        background_dir = "audio_files/specs"
                        background_images = load_background_spectrograms(folder=background_dir, max_images=10)
                        
                        if len(background_images) == 0:
                            st.warning("No background images found. Using spectrogram as background.")
                            background_images = [spec]
                        
                        fig_shap = SHAP_GRADIENT().explain(
                            image=spec,
                            model=st.session_state.model,
                            class_idx=st.session_state.class_label,
                            background=background_images,
                            num_samples=100
                        )
                        with st.expander("SHAP Gradient Results"):
                            st.pyplot(fig_shap, width='content')
            
            if use_occlu:
                st.markdown("### Occlusion Sensitivity Explanation")
                with st.spinner("Generating Occlusion Sensitivity results..."):
                    try:
                        fig_occlu = OcclusionSensitivity().explain(
                            image=spec,
                            model=st.session_state.model,
                            class_idx=st.session_state.class_label,
                            class_names=class_names_audio_deepfakes,
                            patch_size=30,
                            stride=15,
                            occlusion_value=0
                        )
                        with st.expander("Occlusion Sensitivity Results"):
                            st.pyplot(fig_occlu, width='content')
                    except Exception as e:
                        st.error(f"⚠️ Occlusion Sensitivity Error: {str(e)}")
            
            if use_int_grad:
                st.markdown("### Integrated Gradients Explanation")
                with st.spinner("Generating Integrated Gradients results..."):
                    try:
                        fig_int_grad = IntegratedGradients().explain(
                            image=spec,
                            model=st.session_state.model,
                            class_idx=st.session_state.class_label,
                            class_names=class_names_audio_deepfakes,
                            baseline=None,
                            steps=50
                        )
                        with st.expander("Integrated Gradients Results"):
                            st.pyplot(fig_int_grad, width='content')
                    except Exception as e:
                        st.error(f"⚠️ Integrated Gradients Error: {str(e)}")

    
def lung_cancer_pipeline():
    st.title("Lung Cancer Detection")
    st.markdown("Upload a chest X-ray image for lung cancer analysis and understand the model's decision with Explainability AI (XAI).")

    st.divider()

    with st.expander("Model Settings", expanded=True):
        selected_lung_model = st.selectbox(
            "Choose a model to classify chest X-ray image:",
            options=list(AVAILABLE_MODELS_LUNG.keys()),
            index=list(AVAILABLE_MODELS_LUNG.keys()).index(st.session_state.lung_selected_model)
        )
        
        # Reset prediction if model changes
        if selected_lung_model != st.session_state.lung_selected_model:
            st.session_state.lung_selected_model = selected_lung_model
            st.session_state.lung_prediction_done = False

    st.divider()

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("### Upload Image")
        with st.container(border=True):
            uploaded_file = st.file_uploader(
                "Choose an image file",
                type=['png','jpg','jpeg'],
                help="Only .png, .jpg and .jpeg files are supported",
                key="lung_file_uploader"
            )

        if not uploaded_file:
            st.info("Please upload a file to begin.")
            return

        if uploaded_file:
            if not is_file_allowed(uploaded_file.name, {"image": ALLOWED_FILE_TYPES["image"]}):
                st.error("This file type is not allowed. Please upload a .png, .jpg or .jpeg file.")
                return
            

    if uploaded_file:
        with col2:
            # Display the image
            st.markdown("### X-ray Image")
            image = Image.open(uploaded_file)
            st.image(image, width=400)

            # Prediction
            if not st.session_state.get("lung_prediction_done", False):
                with st.spinner("Analyzing image..."):
                    device = "cuda" if torch.cuda.is_available() else "cpu"
                    model_path = AVAILABLE_MODELS_LUNG[st.session_state.lung_selected_model]
                    
                    # Load PyTorch model
                    model = load_pytorch_model(model_path, device=device)
                    
                    # Make prediction
                    image = Image.open(uploaded_file)
                    output = predict_image_pytorch(model, image, device=device)

                    st.session_state.lung_model = model
                    st.session_state.lung_output = output
                    st.session_state.lung_class_label = output["class_idx"]
                    st.session_state.lung_prediction = output["predictions"]
                    st.session_state.lung_prediction_done = True
                    st.session_state.lung_image = image

            class_label = st.session_state.lung_class_label
            prediction = st.session_state.lung_prediction
            
            # prediction[0][0] is probability of benign, prediction[0][1] is probability of malignant
            benign_prob = float(prediction[0][0])
            malignant_prob = float(prediction[0][1])
            
            st.subheader("Prediction Result")

            col_a, col_b = st.columns([2, 1])

            with col_a:
                if class_label == 1:  # malignant
                    confidence = malignant_prob
                    st.error("The image shows **Malignant** (Lung Cancer Detected)")
                else:  # benign
                    confidence = benign_prob
                    st.success("The image shows **Benign** (No Cancer Detected)")

            with col_b:
                st.metric(
                    label="Confidence",
                    value=f"{confidence:.1%}"
                )

        # XAI
        st.divider()
        if not st.session_state.lung_prediction_done:
            st.info("Run a prediction to enable Explainability.")
            return
        
        st.markdown("## Explainability")

        st.markdown("**Select the XAI methods you want to try:**")
        available_methods = get_available_xai_methods(st.session_state.lung_selected_model)
        
        col_lime, col_gradcam, col_shap, col_occlu, col_int_grad = st.columns(5, width=800, border=True)
        
        with col_lime:
            use_lime = st.checkbox("LIME", value=False, disabled=not available_methods["LIME"], key="lung_lime")
        with col_gradcam:
            use_gradcam = st.checkbox("Grad-CAM", value=True, disabled=not available_methods["GradCAM"], key="lung_gradcam")
        with col_shap:
            use_shap = st.checkbox("SHAP Gradient", value=False, disabled=not available_methods["SHAP_GRADIENT"], key="lung_shap")
        with col_occlu:
            use_occlu = st.checkbox("Occlusion Sensitivity", value=False, disabled=not available_methods["OcclusionSensitivity"], key="lung_occlu")
        with col_int_grad:
            use_int_grad = st.checkbox("Integrated Gradients", value=False, disabled=not available_methods["IntegratedGradients"], key="lung_int_grad")
        
        if (use_lime or use_gradcam or use_shap or use_occlu or use_int_grad) and st.button("Run Explainability", key="lung_xai_button"):
            
            # Convert PyTorch model to TensorFlow-compatible format for XAI methods
            # For now, we'll use LIME and GradCAM which can work with PyTorch
            
            if use_lime:
                st.markdown("### LIME Explanation")
                with st.spinner("Generating LIME results..."):
                    try:
                        fig_lime = Lime().explain(
                            image=st.session_state.lung_image,
                            model=st.session_state.lung_model,
                            class_idx=st.session_state.lung_class_label,
                            class_names=class_names_lung_cancer
                        )
                        with st.expander("LIME Results"):
                            st.pyplot(fig_lime, width='content')
                    except Exception as e:
                        st.error(f"⚠️ LIME Error: {str(e)[:100]}")

            if use_gradcam:
                st.markdown("### Grad-CAM Explanation")
                with st.spinner("Generating Grad-CAM results..."):
                    # try:
                        fig_grad = GradCAM().explain(
                            image=st.session_state.lung_image,
                            model=st.session_state.lung_model,
                            class_idx=st.session_state.lung_class_label,
                            class_names=class_names_lung_cancer
                        )
                        with st.expander("Grad-CAM Results"):
                            st.pyplot(fig_grad, width='content')
                    # except Exception as e:
                    #     st.error(f"⚠️ Grad-CAM Error: {str(e)}")

            if use_shap:
                st.markdown("### SHAP Gradient Explanation")
                with st.spinner("Generating SHAP Gradient results..."):
                    fig_shap = SHAP_GRADIENT().explain(
                        image=st.session_state.lung_image,
                        model=st.session_state.lung_model,
                        class_idx=st.session_state.lung_class_label,
                        background=[st.session_state.lung_image],
                        num_samples=100
                    )
                    with st.expander("SHAP Gradient Results"):
                        st.pyplot(fig_shap, width='content')
            
            if use_occlu:
                st.markdown("### Occlusion Sensitivity Explanation")
                with st.spinner("Generating Occlusion Sensitivity results..."):
                    try:
                        fig_occlu = OcclusionSensitivity().explain(
                            image=st.session_state.lung_image,
                            model=st.session_state.lung_model,
                            class_idx=st.session_state.lung_class_label,
                            class_names=class_names_lung_cancer,
                            patch_size=30,
                            stride=15,
                            occlusion_value=0
                        )
                        with st.expander("Occlusion Sensitivity Results"):
                            st.pyplot(fig_occlu, width='content')
                    except Exception as e:
                        st.error(f"⚠️ Occlusion Sensitivity Error: {str(e)}")

            if use_int_grad:
                st.markdown("### Integrated Gradients Explanation")
                with st.spinner("Generating Integrated Gradients results..."):
                    try:
                        fig_int_grad = IntegratedGradients().explain(
                            image=st.session_state.lung_image,
                            model=st.session_state.lung_model,
                            class_idx=st.session_state.lung_class_label,
                            class_names=class_names_lung_cancer,
                            baseline=None,
                            steps=50
                        )
                        with st.expander("Integrated Gradients Results"):
                            st.pyplot(fig_int_grad, width='content')
                    except Exception as e:
                        st.error(f"⚠️ Integrated Gradients Error: {str(e)}")



if __name__ == "__main__":
    main()