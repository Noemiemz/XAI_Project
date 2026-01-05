# **Unified Explainable AI Interface**
### *XAI Project*

---
## Group Members
- Lorrain MORLET
- Noémie MAZEPA
- Auriane MARCELINO

Class: **DIA2**

---

## Project Overview
This project aims to create a unified explainable (XAI) interface, allowing users to classify audio and image files with Deep Learning models but most importantly, analyze the results via XAI techniques to gain insights into the model's decisions.

The platform gives the possibility for users to either detect **audio deepfakes** or **detect lung cancer**:
- **Audio Deepfake Detection**: detects if an audio is real or not, using models like VGG16, MobileNet, ResNet and a custom CNN model. Audio files are first processed into spectrograms for a better performance.
- **Lung Cancer Detection**: detects malignant tumors in chest X-rays, using the fine-tuned AlexNet and DenseNet models.

The platform integrates XAI methods such as LIME, Grad-CAM, SHAP, Occlusion Sensitivity and Integrated Gradients to better understand the different models' predictions and have a better transparency in the models' decisions.

## Key Features
- Multi-modal support: audio (.wav and .mp3) and images (chest X-rays in .jpg, .jpeg and .png).
- Choice between several classification models for both tasks.
- Application of different XAI methods: LIME, Grad-CAM, SHAP, Occlusion Sensitivity, Integrated Gradients. 
- Automatic filtering of the proposed XAI techniques based on the input type.
- Web interface via Streamlit.

## Project Structure
```

```

## Audio Deepfake Detection
Users can upload an audio file, select a model, apply one or several XAI methods and look at the predictions with explanations.

- **Input**: Audio files (.wav, .mp3)

- **Models**: VGG16, MobileNet, ResNet, Custom CNN

- **XAI**: LIME, SHAP, Grad-CAM (on spectrograms), Occlusion Sensitivity, Integrated Gradients



## Lung Cancer Detection
Users can upload chest X-rays in the form of an image, select a model, apply one or several XAI methods and inspect the areas of interest highlighted by each XAI model.

- **Input**: Chest X-ray images (.png, .jpg)

- **Models**: AlexNet, DenseNet

- **XAI**: Grad-CAM, LIME, SHAP, Occlusion Sensitivity, Integrated Gradients


## Technologies Used

- **Deep Learning Models**: 
    - *Audio Deepfake Detection*: VGG16, MobileNet, ResNet, Custom CNN
    - *Lung Cancer Detection*: AlexNet, DenseNet
- **Explainable AI (XAI) Techniques**: LIME, Grad-CAM, SHAP
- **Programming Languages and Libraries**: Python, TensorFlow, Keras, Matplotlib, NumPy, Librosa
- **Development Tools**: Jupyter Notebooks
- **Web Application Framework**: Streamlit

## Streamlit Web Application
The Streamlit interface allows users to interact with the tasks and models in a simple way:

- Choose the task (audie deepfake detection or Lung Cancer Detection)
- Upload an audio or image file based on the task
- Select the classification model
- Apply XAI methods
- Visualize results and explanations
- Compare multiple explainability outputs in a side-by-side comparison tab


## Setup and Installation

**1. Clone the repository**
````
git clone https://github.com/Noemiemz/XAI_Project.git
cd XAI_Project
````

**2. Set up a virtual environment**
For Windows:
````
python -m venv venv
source venv\Scripts\activate
````

**3. Install dependencies**
````
pip install --upgrade pip
pip install -r requirements.txt
````
**Note**: Python 3.11 or below is recommended. Python 3.12 can cause compatibility issues.

**4. Download the pretrained classifier models weights**
This project uses several classifier models to make predictions. The weigths of those pretrained models are hosted in 2 Hugging Face repositories.

To ensure that the models load correctly when using the Streamlit application, these repositories need to be cloned locally and placed in the appropriate folders.

- **Lung Cancer Detection Models**
    Clone the repository containing the pretrained AlexNet and DenseNet weights:
    ````
    git clone https://huggingface.co/nomiemzp/Lung_Cancer_Detection
    ````
    
    After cloning, place the downloaded folder in the following directory:
    ````

    ````

- **Audio Deepfake Detection Models**
    Clone the repository containing the pretrained AlexNet and DenseNet weights:
    ````
    git clone https://huggingface.co/Nasotro/xai_audioclassifiers
    ````
    
    After cloning, place the downloaded folder in the following directory:
    ````

    ````

**Note**: the folder structure and location must match the paths expected in the Streamlit application (`Streamlit/app.py` file) or need to be updated in the code.


**5. Run the Streamlit application**
````
streamlit run Streamlit/app.py
````
Open the URL shown in the terminal in your web browser (ex: http://localhost:8503)

---
## Generative AI Usage Statement

