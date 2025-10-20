import cv2
import pickle
import numpy as np
import tensorflow as tf

# Force TensorFlow to use CPU
tf.config.set_visible_devices([], 'GPU')
print("TensorFlow is configured to use CPU.")

# --- Configuration ---
MODEL_PATH = 'best_resnet50_bovine_classifier.pkl'
IMG_HEIGHT, IMG_WIDTH = 224, 224

# IMPORTANT: Update this list to match your breed folder names in the correct order.
# You can find the order from the output of the training script.
# Example: ['Amritmahal', 'Gir', 'Hallikar', 'Hariana', 'Kankrej', 'Sahiwal']
CLASS_NAMES = [
    'Amritmahal', 'Deoni', 'Gir', 'Hallikar', 'Hariana', 'Kankrej', 
    'Khillar', 'Ongole', 'Red Kandhari', 'Sahiwal'
] 
# This is an example list based on common Indian breeds, please verify and correct it.

# --- Load the Trained Model ---
print(f"Loading model from {MODEL_PATH}...")
try:
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    print("Model loaded successfully.")
except FileNotFoundError:
    print(f"Error: Model file not found at '{MODEL_PATH}'")
    print("Please make sure the model file is in the same directory as this script.")
    exit()
except Exception as e:
    print(f"An error occurred while loading the model: {e}")
    exit()

# --- Initialize Webcam ---
cap = cv2.VideoCapture(0) # Use 0 for the default webcam
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("\nStarting webcam feed. Press 'q' to quit.")

while True:
    # 1. Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    # 2. Preprocess the frame for the model
    # Resize the frame to the model's expected input size
    input_image = cv2.resize(frame, (IMG_WIDTH, IMG_HEIGHT))
    
    # Convert the image from BGR (OpenCV's default) to RGB
    input_image_rgb = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
    
    # Convert to a numpy array and expand dimensions to create a batch of 1
    image_array = np.expand_dims(input_image_rgb, axis=0)
    
    # Apply the same preprocessing as in training
    preprocessed_image = tf.keras.applications.resnet50.preprocess_input(image_array)

    # 3. Make a prediction
    predictions = model.predict(preprocessed_image)
    
    # 4. Interpret the prediction
    score = np.max(predictions[0])
    predicted_class_index = np.argmax(predictions[0])
    
    if predicted_class_index < len(CLASS_NAMES):
        predicted_class_name = CLASS_NAMES[predicted_class_index]
    else:
        predicted_class_name = "Unknown Class"

    # 5. Display the result on the frame
    display_text = f"{predicted_class_name}: {score:.2f}"
    
    # Put text on the original, larger frame
    cv2.putText(
        frame, 
        display_text, 
        (10, 30),                 # Position
        cv2.FONT_HERSHEY_SIMPLEX, # Font
        1,                        # Font scale
        (0, 255, 0),              # Color (Green)
        2,                        # Thickness
        cv2.LINE_AA
    )

    # 6. Show the frame
    cv2.imshow('Bovine Classifier', frame)

    # 7. Check for exit key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# --- Cleanup ---
print("Closing application.")
cap.release()
cv2.destroyAllWindows()