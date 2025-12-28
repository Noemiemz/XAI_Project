import sys
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

import streamlit as st
import pandas as pd
import numpy as np
import librosa
from tensorflow.keras.models import load_model
import librosa.display, os
import matplotlib.pyplot as plt
from keras.preprocessing.image import load_img,img_to_array
from keras.models import load_model
import tensorflow as tf
from keras.applications.mobilenet import preprocess_input
import matplotlib.pyplot as plt
from skimage.util import img_as_float

from XAI_models.xai_models import Lime, GradCAM
from Inference.inference import predict_image

st.set_page_config(
    page_title="XAI Audio Detection",
    page_icon="🐈",
    layout="wide"
)

class_names = ['real','fake']

def save_file(sound_file):
    # save your sound file in the right folder by following the path
    with open(os.path.join('audio_files/', sound_file.name),'wb') as f:
         f.write(sound_file.getbuffer())
    return sound_file.name

def create_spectrogram(sound):
    audio_file = os.path.join('audio_files/', sound)

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

    y, sr = librosa.load(audio_file)
    ms = librosa.feature.melspectrogram(y=y, sr=sr)
    log_ms = librosa.power_to_db(ms, ref=np.max)
    librosa.display.specshow(log_ms, sr=sr)
    plt.savefig('melspectrogram.png')
    image_data = load_img('melspectrogram.png',target_size=(224,224))
    st.image(image_data)
    return(image_data)


def main():
    with st.sidebar:
        st.markdown('## XAI Platform')
        page= st.radio("Navigation", ["Home", "Explanability", "About"], label_visibility="collapsed")
    if page == "Home":
        homepage()
    elif page=="Explanability":
        homepage()
    elif page == "About":
        about()

def about():
    st.title("About present work")
    st.markdown("**Deepfake audio refers to synthetically created audio by digital or manual means. An emerging field, it is used to not only create legal digital hoaxes, but also fool humans into believing it is a human speaking to them. Through this project, we create our own deep faked audio using Generative Adversarial Neural Networks (GANs) and objectively evaluate generator quality using Fréchet Audio Distance (FAD) metric. We augment a pre-existing dataset of real audio samples with our fake generated samples and classify data as real or fake using MobileNet, Inception, VGG and custom CNN models. MobileNet is the best performing model with an accuracy of 91.5% and precision of 0.507. We further convert our black box deep learning models into white box models, by using explainable AI (XAI) models. We quantitatively evaluate the classification of a MEL Spectrogram through LIME, SHAP and GradCAM models. We compare the features of a spectrogram that an XAI model focuses on to provide a qualitative analysis of frequency distribution in spectrograms.**")
    st.markdown("**The goal of this project is to study features of audio and bridge the gap of explain ability in deep fake audio detection, through our novel system pipeline. The findings of this study are applicable to the fields of phishing audio calls and digital mimicry detection on video streaming platforms. The use of XAI will provide end-users a clear picture of frequencies in audio that are flagged as fake, enabling them to make better decisions in generation of fake samples through GANs.**")

def homepage():
    with st.expander("What does this app do?", expanded=True):
        st.write(
            """
            This application detects **deepfake audio** using a deep learning model.
            It then explains the prediction using **Explainable AI (XAI)** techniques
            like **LIME** and **Grad-CAM**.
            """
        )


    st.markdown("## Deepfake Audio Detection with XAI")
    st.markdown("Upload an audio file to detect if it's **real** or **fake** and understand the model's decision with Explainability AI (XAI).")

    st.divider()

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("### Upload Audio")
        uploaded_file = st.file_uploader(
            "Choose a WAV file",
            type="wav",
            help="Only .wav audio files are supported"
        )

        if not uploaded_file:
            st.info("Please upload a WAV file to begin.")
            return

        if uploaded_file:
            audio_bytes = uploaded_file.read()
            st.audio(audio_bytes, format="audio/wav")

    if uploaded_file:
        with col2:
            st.markdown("### Spectrogram")

            with st.spinner("Generating Spectrogram..."):
                save_file(uploaded_file)
                spec = create_spectrogram(uploaded_file.name)

    
            # Prediction
            st.markdown("### Prediction Result")

            with st.spinner("Analyzing audio..."):
                model = tf.keras.models.load_model('Streamlit/saved_model/model')
                output = predict_image(model, spec)
                class_label = output["class_idx"]
                prediction = output["predictions"]


            if class_names[class_label] == "fake":
                st.error("The audio is **Fake**")
            else:
                st.success("The audio is **Real**")
            confidence = float(prediction[0][class_label])
            st.progress(confidence)
            st.caption(f"Confidence: {confidence:.2%}")

        # XAI
        st.divider()
        st.markdown("## Explainability")

        xai_methods = st.multiselect(
            "Select XAI methods",
            ["LIME", "Grad-CAM"],
            default=["LIME"]
        )

        if st.button("Run Explainability"):
            if "LIME" in xai_methods:
                st.markdown("### LIME Explanation")
                with st.spinner("Generating XAI results..."):
                    fig_lime = Lime().explain(
                        image=spec,
                        model=model,
                        class_idx=class_label,
                        class_names=class_names
                    )
                    st.pyplot(fig_lime)

            if "Grad-CAM" in xai_methods:
                st.markdown("### Grad-CAM Explanation")
                with st.spinner("Generating XAI results..."):
                    fig_grad = GradCAM().explain(
                        image=spec,
                        model=model,
                        class_idx=class_label,
                        class_names=class_names
                    )
                    st.pyplot(fig_grad)



if __name__ == "__main__":
    main()