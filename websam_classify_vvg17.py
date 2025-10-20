import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import preprocess_input
import os

# --- 1. Define Paths and Constants ---
# This should be the same model file saved by your training script
MODEL_FILENAME = 'indian_bovine_vgg16_model_cpu.h5'
IMAGE_SIZE = (224, 224)

# We need to get the class names in the same order the model was trained on.
# The most reliable way is to read the sub-directory names from the dataset folder.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# The dataset folder is in the same directory as the script.
INPUT_FOLDER = os.path.join(SCRIPT_DIR, 'Indian_bovine_breeds')

# --- 2. Load the Trained Model and Class Names ---
print("Loading model and class names...")

# Check for model file
if not os.path.exists(MODEL_FILENAME):
    raise FileNotFoundError(f"Model file not found at '{MODEL_FILENAME}'. Please run the training script first to generate it.")

# Load the Keras model
model = tf.keras.models.load_model(MODEL_FILENAME)
print(f"✅ Model '{MODEL_FILENAME}' loaded successfully.")

# Check for dataset directory to infer class names
if not os.path.exists(INPUT_FOLDER):
    raise FileNotFoundError(f"Dataset folder not found at '{os.path.abspath(INPUT_FOLDER)}'. It is needed to get the class names.")

# Get class names by listing the subdirectories, sorted alphabetically
class_names = sorted([d for d in os.listdir(INPUT_FOLDER) if os.path.isdir(os.path.join(INPUT_FOLDER, d))])
print(f"✅ Found {len(class_names)} classes: {class_names}")


# --- 3. Initialize Webcam and Start Classification Loop ---
print("\nStarting webcam feed... Press 'q' to exit.")
# Use 0 for the default webcam. CAP_DSHOW is often more stable on Windows.
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    raise IOError("Cannot open webcam. Please check if it is connected and not in use by another application.")

while True:
    # Read a frame from the webcam
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    # --- 4. Preprocess the Frame for the Model ---
    # Resize the frame to the size expected by the model (224x224)
    img_resized = cv2.resize(frame, IMAGE_SIZE)

    # Convert the image to a numpy array and add a batch dimension (1, 224, 224, 3)
    img_array = np.expand_dims(img_resized, axis=0)

    # Use the VGG16 preprocess_input function to prepare the image
    img_preprocessed = preprocess_input(img_array)

    # --- 5. Make a Prediction ---
    predictions = model.predict(img_preprocessed)
    
    # Get the index of the highest probability and the probability value
    predicted_class_index = np.argmax(predictions[0])
    confidence = np.max(predictions[0]) * 100  # Convert to percentage

    # Get the corresponding class name
    predicted_class_name = class_names[predicted_class_index]

    # --- 6. Display the Result on the Frame ---
    # Create the text to display
    label = f"{predicted_class_name}: {confidence:.2f}%"

    # Put the text on the frame
    cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('Bovine Breed Classification', frame)

    # Check for the 'q' key to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# --- 7. Clean Up ---
print("Closing webcam feed.")
cap.release()
cv2.destroyAllWindows()