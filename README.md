# 🥹 Emotion-detection-analysis
- This repository contains the code and resources for an Emotion Detection project, developed to identify human emotions from various inputs.
- Leveraging deep learning techniques, it also aims to provide insights into emotional states, with a strong emphasis on being a responsible and ethical application.

## ✨ Project Objectives
* The core objective is to develop a tool that can be used for various beneficial applications, such as improving human-computer interaction, enhancing mental health support systems, or personalizing user experiences.
* This project focuses on building a robust model capable of recognizing and classifying emotions.
* The approach involves using a neural network trained on a relevant dataset to learn the subtle patterns associated with different expressions.
* In the process, we are committed to ensure that the development and deployment of this technology are handled with care and consideration for individual privacy and well-being.
* Note: All the images used for training are obtained from FER-2013 dataset from kaggle.

## 🛠 Technology Stack
- This project is built using the following key technologies and libraries:

- Python: The core programming language for the entire project.
- PyTorch: A leading open-source machine learning framework used for building and training neural networks.
- Torchvision: A PyTorch library that provides popular datasets, model architectures, and image transformations for computer vision(CV).
- NumPy: Essential for numerical operations and efficient array manipulation.
- Matplotlib: For data visualization such as plotting training and validation metrics.
- Google Colab: Cloud GPU environment used for development, experimentation, and running the training scripts.
- Streamlit: Used for intercative User Interface to upload images and showcase the detected emotions.
- dlib: A library used for face detection and landmark prediction.
- cv2: OpenCV

## Features
* Deep Learning Model: Utilizes a pre-trained neural network architecture (Convolutional Neural Network - CNN) for emotion classification.
* Training Loop: Includes a comprehensive training script with validation phases to monitor model performance and prevent overfitting.
* Performance Tracking: Records training and validation loss/accuracy per epoch to visualize learning progress.
* Best Model Saving: Implements a mechanism to save the model weights that achieves the highest validation accuracy, ensuring optimal performance is preserved.

## 🚀 Getting Started
- Environment setup: Use Virtual Environment so the arithmetics used for other projects are left untouched.
- Type the following commands one after another in the terminal.
- python -m venv env # to create virtual environment(Note: env at the end is environment name - it can be any name provided by the developer).
- .\env\Scripts\activate
- pip install streamlit dlib torch torchvision opencv-python matplotlib scikit-learn
- Prepare your dataset:
  - Ensure your dataset is organized and preprocessed according to the expectations of the DataLoader in the training script.(Note: Place your dataset in a folder in GDrive if Google Collab is being used).
- Run the training script (copyemotiondetectiontrain.ipynb), save the emotion_cnn_model.pth to the folder where the streamlit app python file is placed.
- Download 68-shape-face-detector.dat file and place it in the same folder where the streamlit app python file is placed.(68-shape-face-detector.dat is part of dlib (Histogram-oriented Gradient(HOG) based face detector)to extract face features and landmarks).
- Run the streamlit app Emotiondetection.ipynb
- Upload the image and the emotions will get detected.

## 📞 Support & Contribution
* Contributions to this project, including suggestions for improvements, bug fixes, and new features are Welcomed!
* Please feel free to open an issue or submit a pull request.


