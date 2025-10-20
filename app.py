from flask import Flask, Response, make_response
import cv2
import pickle
import numpy as np
import tensorflow as tf

# --- Flask App Initialization ---
# We specify template_folder and static_folder to be the same directory
# although we will serve them manually for clarity.
app = Flask(__name__)

# --- Configuration ---
MODEL_PATH = 'best_resnet50_bovine_classifier.pkl'
IMG_HEIGHT, IMG_WIDTH = 224, 224

# IMPORTANT: Update this list to match your breed folder names in the correct order.
CLASS_NAMES = [
    'Amritmahal', 'Deoni', 'Gir', 'Hallikar', 'Hariana', 'Kankrej', 
    'Khillar', 'Ongole', 'Red Kandhari', 'Sahiwal'
] 
# This is an example list, please verify and correct it.

# --- Load the Trained Model ---
print(f"Loading model from {MODEL_PATH}...")
try:
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    print("Model loaded successfully.")
except FileNotFoundError:
    print(f"Error: Model file not found at '{MODEL_PATH}'")
    exit()
except Exception as e:
    print(f"An error occurred while loading the model: {e}")
    exit()

# --- Initialize Webcam ---
camera = cv2.VideoCapture(0)

def generate_frames():
    """Captures frames, processes them, and yields them for the web page."""
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            input_image = cv2.resize(frame, (IMG_WIDTH, IMG_HEIGHT))
            input_image_rgb = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
            image_array = np.expand_dims(input_image_rgb, axis=0)
            preprocessed_image = tf.keras.applications.resnet50.preprocess_input(image_array)

            predictions = model.predict(preprocessed_image)
            score = np.max(predictions[0])
            predicted_class_index = np.argmax(predictions[0])
            
            predicted_class_name = CLASS_NAMES[predicted_class_index] if predicted_class_index < len(CLASS_NAMES) else "Unknown"

            display_text = f"{predicted_class_name}: {score:.2f}"
            cv2.putText(frame, display_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

            ret, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    """Serves the index.html file."""
    with open('index.html', 'r') as f:
        html_content = f.read()
    return Response(html_content, mimetype='text/html')

@app.route('/style.css')
def style():
    """Serves the style.css file."""
    with open('style.css', 'r') as f:
        css_content = f.read()
    response = make_response(css_content)
    response.headers['Content-Type'] = 'text/css'
    return response

@app.route('/video_feed')
def video_feed():
    """Video streaming route."""
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)