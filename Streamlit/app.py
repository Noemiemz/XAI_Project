import sys
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

import streamlit as st
import numpy as np
import librosa
import librosa.display, os
import matplotlib.pyplot as plt
from keras.preprocessing.image import load_img
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image

from XAI_models.xai_models import Lime, GradCAM, SHAP_Gradient, SHAP_Kernel, SHAP_DEEP_EXPLAINER
from Inference.inference import predict_image

ALLOWED_FILE_TYPES = {
    "audio": [".wav", ".mp3"],
    "image": [".png", ".jpg", ".jpeg"]
}

st.set_page_config(
    page_title="XAI Audio Detection",
    page_icon="🐈",
    layout="wide"
)

class_names_audio_deepfakes = ['real','fake']

# Available models
AVAILABLE_MODELS = {
    # "MobileNet": "Streamlit/saved_model/model",
    "MobileNet": "Streamlit/saved_models/MobileNet_audio_classifier.h5",
    "InceptionV3": "Streamlit/saved_models/InceptionV3_audio_classifier.h5",
    "VGG16": "Streamlit/saved_models/VGG16_audio_classifier.h5",
    "ResNet50": "Streamlit/saved_models/ResNet50_audio_classifier.h5"
}

if "prediction_done" not in st.session_state:
    st.session_state.prediction_done = False
if "selected_model" not in st.session_state:
    st.session_state.selected_model = list(AVAILABLE_MODELS.keys())[0]

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
    st.sidebar.title('XAI Platform')
    pipeline = st.sidebar.radio ("What type of detection do you want to do ?:", ["Audio Deepfake Detection", "Lung Cancer Detection"])
    
    # Model selection (only for Audio Deepfake Detection)
    if pipeline == "Audio Deepfake Detection":
        st.sidebar.markdown("---")
        st.sidebar.markdown("### Model Selection")
        selected_model = st.sidebar.selectbox(
            "Choose a model:",
            options=list(AVAILABLE_MODELS.keys()),
            index=list(AVAILABLE_MODELS.keys()).index(st.session_state.selected_model)
        )
        
        # Reset prediction if model changes
        if selected_model != st.session_state.selected_model:
            st.session_state.selected_model = selected_model
            st.session_state.prediction_done = False
    
    if pipeline == "Audio Deepfake Detection":
        audio_pipeline()
    elif pipeline == "Lung Cancer Detection":
        lung_cancer_pipeline()
    else:
        st.error("Unknown pipeline selected!")

# def about():
#     st.title("About present work")
#     st.markdown("**Deepfake audio refers to synthetically created audio by digital or manual means. An emerging field, it is used to not only create legal digital hoaxes, but also fool humans into believing it is a human speaking to them. Through this project, we create our own deep faked audio using Generative Adversarial Neural Networks (GANs) and objectively evaluate generator quality using Fréchet Audio Distance (FAD) metric. We augment a pre-existing dataset of real audio samples with our fake generated samples and classify data as real or fake using MobileNet, Inception, VGG and custom CNN models. MobileNet is the best performing model with an accuracy of 91.5% and precision of 0.507. We further convert our black box deep learning models into white box models, by using explainable AI (XAI) models. We quantitatively evaluate the classification of a MEL Spectrogram through LIME, SHAP and GradCAM models. We compare the features of a spectrogram that an XAI model focuses on to provide a qualitative analysis of frequency distribution in spectrograms.**")
#     st.markdown("**The goal of this project is to study features of audio and bridge the gap of explain ability in deep fake audio detection, through our novel system pipeline. The findings of this study are applicable to the fields of phishing audio calls and digital mimicry detection on video streaming platforms. The use of XAI will provide end-users a clear picture of frequencies in audio that are flagged as fake, enabling them to make better decisions in generation of fake samples through GANs.**")

def audio_pipeline():
    # with st.expander("What does this app do?", expanded=True):
    #     st.write(
    #         """
    #         This application detects **deepfake audio** using a deep learning model.
    #         It then explains the prediction using **Explainable AI (XAI)** techniques
    #         like **LIME** and **Grad-CAM**.
    #         """
    #     )

    st.title("Audio Deepfake Detection with XAI")
    st.markdown("Upload an audio file to detect if it's **real** or **fake** and understand the model's decision with Explainability AI (XAI).")

    st.divider()

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("### Upload Audio")
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
                print('ccccccccccccccccccc', spec.size)
            # Prediction
            if not st.session_state.prediction_done:
                with st.spinner("Analyzing audio..."):
                    model_path = AVAILABLE_MODELS[st.session_state.selected_model]
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
            print(f"Fake probability: {fake_probability}")
            print(f"Class label: {class_label}")
            if class_label == 1:  # fake
                confidence = fake_probability
                st.error("The audio is **Fake**")
            else:  # real
                confidence = 1 - fake_probability
                st.success("The audio is **Real**")

            st.progress(confidence)
            st.caption(f"Confidence: {confidence:.2%}")

        # XAI
        st.divider()
        st.markdown("## Explainability")

        st.markdown("**Select XAI methods:**")
        col_lime, col_gradcam, col_shap, col_shap_kernel, col_shap_deep = st.columns(5, width=500, border=True)
        
        with col_lime:
            use_lime = st.checkbox("LIME", value=True)
        with col_gradcam:
            use_gradcam = st.checkbox("Grad-CAM", value=False)
        with col_shap:
            use_shap = st.checkbox("SHAP Gradient", value=False)
        with col_shap_kernel:
            use_shap_kernel = st.checkbox("SHAP Kernel", value=False)
        with col_shap_deep:
            use_shap_deep = st.checkbox("SHAP Deep", value=False)

        if (use_lime or use_gradcam or use_shap or use_shap_kernel or use_shap_deep) and st.button("Run Explainability"):

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
                        st.error(f"⚠️ Grad-CAM Error: {str(e)[:100]}")

            if use_shap:
                st.markdown("### SHAP Gradient Explanation")
                with st.spinner("Generating SHAP Gradient results..."):
                        # Load background images for SHAP (sample from training data)
                        background_dir = "audio_files/specs"
                        background_images = load_background_spectrograms(folder=background_dir, max_images=10)
                        
                        if len(background_images) == 0:
                            st.warning("No background images found. Using spectrogram as background.")
                            background_images = [spec]
                        
                        fig_shap = SHAP_Gradient().explain(
                            image=spec,
                            model=st.session_state.model,
                            class_idx=st.session_state.class_label,
                            background_images=background_images,
                            class_names=class_names_audio_deepfakes
                        )
                        with st.expander("SHAP Gradient Results"):
                            st.pyplot(fig_shap, width='content')

            if use_shap_kernel:
                st.markdown("### SHAP Kernel Explanation")
                with st.spinner("Generating SHAP Kernel results (this may take longer)..."):
                        # Load background images for SHAP (sample from training data)
                        background_dir = "audio_files/specs"
                        background_images = load_background_spectrograms(folder=background_dir, max_images=10)
                        
                        if len(background_images) == 0:
                            st.warning("No background images found. Using spectrogram as background.")
                            background_images = [spec]
                        
                        fig_shap_kernel = SHAP_Kernel().explain(
                            image=spec,
                            model=st.session_state.model,
                            class_idx=st.session_state.class_label,
                            background_images=background_images,
                            class_names=class_names_audio_deepfakes,
                            max_background=10,
                            nsamples=100
                        )
                        with st.expander("SHAP Kernel Results"):
                            st.pyplot(fig_shap_kernel, width='content')
            if use_shap_deep:
                st.markdown("### SHAP Deep Explanation")
                with st.spinner("Generating SHAP Deep results..."):
                        # Load background images for SHAP (sample from training data)
                        background_dir = "audio_files/specs"
                        background_images = load_background_spectrograms(folder=background_dir, max_images=10)
                        
                        if len(background_images) == 0:
                            st.warning("No background images found. Using spectrogram as background.")
                            background_images = [spec]
                        
                        fig_shap_deep = SHAP_DEEP_EXPLAINER().explain(
                            image=spec,
                            model=st.session_state.model,
                            class_idx=st.session_state.class_label
                        )
                        with st.expander("SHAP Deep Results"):
                            st.pyplot(fig_shap_deep, width='content')
            

def lung_cancer_pipeline():
    st.title("Lung Cancer Detection")
    st.markdown("Upload an x-ray image for lung cancer analysis and understand the model's decision with Explainability AI (XAI).")

    st.divider()

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("### Upload Image")
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=['png','jpg','jpeg'],
            help="Only .png, .jpg and .jpeg files are supported"
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
            st.image(uploaded_file)



if __name__ == "__main__":
    main()