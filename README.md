# Indian Bovine Breeds Classification
[![Ask DeepWiki](https://devin.ai/assets/askdeepwiki.png)](https://deepwiki.com/SezarTheGreat/Indian-Bovine-Breeds-Classification)

This repository contains a deep learning project for classifying various Indian bovine breeds. The system uses a pre-trained ResNet-50 model, fine-tuned to recognize ten different breeds. It includes scripts for training the model, performing real-time classification via a webcam, and a web application to stream the classification results to a browser.

## Features

- **Transfer Learning:** Utilizes a ResNet-50 model pre-trained on ImageNet for robust feature extraction.
- **Multi-Class Classification:** Capable of classifying ten distinct Indian bovine breeds.
- **Real-time Webcam Classifier:** A script to use your webcam for live breed identification.
- **Web Application:** A Flask-based web app that streams the webcam feed with classification overlays directly to your browser.
- **Pre-trained Model:** Includes a pre-trained model file, ready for immediate use.

## Model Details

The classification model is built on top of the ResNet-50 architecture.

- **Base Model:** ResNet-50 with weights pre-trained on ImageNet. The convolutional base is frozen (`trainable=False`) to leverage its learned features without altering them.
- **Custom Head:** On top of the base, the following layers are added:
    1.  `GlobalAveragePooling2D()`: To flatten the feature maps.
    2.  `Dense(1024, activation='relu')`: A fully connected layer for learning high-level patterns.
    3.  `Dense(10, activation='softmax')`: The final output layer for classification among the 10 breeds.
- **Optimizer:** Adam with a learning rate of `0.001`.
- **Loss Function:** `sparse_categorical_crossentropy`.

The model is trained on the CPU to ensure accessibility for users without a dedicated GPU.

## Breeds Classified

The model is trained to recognize the following 10 breeds:
- Amritmahal
- Deoni
- Gir
- Hallikar
- Hariana
- Kankrej
- Khillar
- Ongole
- Red Kandhari
- Sahiwal

## Project Structure

```
.
├── ResNet-50.py                    # Script to train the ResNet-50 model from scratch.
├── ResNet-50BovineClassify.py      # Script for real-time classification using a local webcam.
├── app.py                          # Flask web server for the browser-based application.
├── best_resnet50_bovine_classifier.pkl # The pre-trained model file (requires Git LFS).
├── index.html                      # Frontend for the web application.
├── requirements.txt                # Python dependencies.
└── style.css                       # CSS for the web application.
```

## Setup and Installation

### Prerequisites
- Python 3.8+
- [Git LFS](https://git-lfs.github.com/) (for downloading the model file)

### Instructions
1.  **Clone the repository:**
    Make sure you have Git LFS installed before cloning.
    ```bash
    git lfs install
    git clone https://github.com/SezarTheGreat/Indian-Bovine-Breeds-Classification.git
    cd Indian-Bovine-Breeds-Classification
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    This will install libraries including TensorFlow, OpenCV, and Flask.

## Usage

You can either use the pre-trained model for classification or train your own.

### 1. Real-time Webcam Classifier (Console)
This script uses OpenCV to capture your webcam feed and display the classification result in a window.

-   **Run the script:**
    ```bash
    python ResNet-50BovineClassify.py
    ```
-   A window will open showing your webcam feed with the predicted breed and confidence score overlaid.
-   Press `q` to quit.

### 2. Web Application Classifier
This runs a Flask server that you can access from your web browser.

-   **Run the Flask application:**
    ```bash
    python app.py
    ```
-   **Open your browser** and navigate to:
    ```
    http://127.0.0.1:5000
    ```
-   You will see the live webcam stream with the classification results.

### 3. Training a New Model
To train the model yourself, you will need to provide the dataset.

-   **Prepare the dataset:**
    -   Create a directory named `Indian_bovine_breeds` in the root of the project.
    -   Inside this directory, create subdirectories for each breed class (e.g., `Gir`, `Sahiwal`, etc.).
    -   Place the corresponding images for each breed into their respective folders.

-   **Run the training script:**
    ```bash
    python ResNet-50.py
    ```
-   The script will split the data, train the model for 20 epochs, and save the model with the best validation accuracy as `best_resnet50_bovine_classifier.pkl`.
