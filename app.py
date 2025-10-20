import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import preprocess_input
from flask import Flask, render_template, request, jsonify
import base64
import io
from PIL import Image

# --- 1. Initialize App and Paths ---
app = Flask(__name__)

# Paths (assuming files are in the same directory as app.py)
MODEL_FILENAME = 'indian_bovine_vgg16_model_cpu.h5'
INPUT_FOLDER = 'Indian_bovine_breeds'
IMAGE_SIZE = (224, 224)

# --- 2. Load Model and Class Names (Done once at startup) ---
print("Loading model and class names...")

# Check for model file
if not os.path.exists(MODEL_FILENAME):
    raise FileNotFoundError(f"Model file not found at '{MODEL_FILENAME}'.")
model = tf.keras.models.load_model(MODEL_FILENAME)
print(f"✅ Model '{MODEL_FILENAME}' loaded.")

# Check for dataset directory to infer class names
if not os.path.exists(INPUT_FOLDER):
    raise FileNotFoundError(f"Dataset folder '{INPUT_FOLDER}' not found. It is needed to get class names.")
class_names = sorted([d for d in os.listdir(INPUT_FOLDER) if os.path.isdir(os.path.join(INPUT_FOLDER, d))])
print(f"✅ Found {len(class_names)} classes: {class_names}")

# --- 3. Preprocessing Function ---
def preprocess_image(image_data):
    """Decodes base64 image, preprocesses it for VGG16, and returns a numpy array."""
    # Decode the base64 string
    img_data = base64.b64decode(image_data.split(',')[1])
    # Convert to a PIL Image
    img = Image.open(io.BytesIO(img_data))
    # Convert to numpy array (and from RGB to BGR for OpenCV)
    frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    
    # Resize and preprocess for the model
    img_resized = cv2.resize(frame, IMAGE_SIZE)
    img_array = np.expand_dims(img_resized, axis=0)
    img_preprocessed = preprocess_input(img_array)
    return img_preprocessed

# --- 4. Define Flask Routes ---
@app.route('/')
def index():
    """Render the main HTML page."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Receive image data, predict, and return JSON response."""
    try:
        data = request.get_json()
        image_data = data['image']
        
        # Preprocess the image and make a prediction
        processed_image = preprocess_image(image_data)
        predictions = model.predict(processed_image)
        
        # Get top prediction
        predicted_class_index = np.argmax(predictions[0])
        confidence = np.max(predictions[0]) * 100
        predicted_class_name = class_names[predicted_class_index]
        
        # Return the result as JSON
        return jsonify({
            'breed': predicted_class_name,
            'confidence': f"{confidence:.2f}"
        })
    except Exception as e:
        print(f"Error during prediction: {e}")
        return jsonify({'error': str(e)}), 500

# --- 5. Run the App ---
if __name__ == '__main__':
    # Use host='0.0.0.0' to make it accessible on your local network
    app.run(debug=True, host='0.0.0.0')