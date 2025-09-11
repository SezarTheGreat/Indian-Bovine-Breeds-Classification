import cv2
import pickle
import numpy as np

# --- Configuration ---
MODEL_PATH = 'bovine_classifier.pkl'
IMG_WIDTH, IMG_HEIGHT = 224, 224

# --- Load the trained model ---
try:
    with open(MODEL_PATH, 'rb') as file:
        model_data = pickle.load(file)
        model = model_data['model']
        class_names = model_data['class_names']
    print("Model loaded successfully.")
except FileNotFoundError:
    print(f"Error: Model file '{MODEL_PATH}' not found. Please run 'train_model.py' first.")
    exit()

# --- Initialize the camera ---
cap = cv2.VideoCapture(0) # 0 for the default camera

if not cap.isOpened():
    print("Error: Could not open video stream.")
    exit()

print("--- Starting real-time classification. Press 'q' to quit. ---")

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        break

    # --- Preprocess the frame for the model ---
    resized_frame = cv2.resize(frame, (IMG_WIDTH, IMG_HEIGHT))
    normalized_frame = resized_frame / 255.0
    input_frame = np.expand_dims(normalized_frame, axis=0) # Add a batch dimension

    # --- Real-time Inference ---
    predictions = model.predict(input_frame)
    probabilities = predictions[0]

    # Get the top predictions
    top_n = 3
    top_indices = np.argsort(probabilities)[::-1][:top_n]

    # --- Display predictions on the frame ---
    y_offset = 30
    for i in range(top_n):
        idx = top_indices[i]
        breed = class_names[idx]
        score = probabilities[idx] * 100  # Convert to percentage
        text = f"{breed}: {score:.2f}%"

        # Display the text on the video feed
        cv2.putText(
            frame, 
            text, 
            (10, y_offset + i * 25), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.7, 
            (0, 255, 0), # Green color
            2
        )

    # Display the resulting frame
    cv2.imshow('Bovine Breed Classifier', frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close all windows
cap.release()
cv2.destroyAllWindows()