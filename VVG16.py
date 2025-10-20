# ==============================================================================
#      VGG16 IMPLEMENTATION - FORCED CPU TRAINING
# ==============================================================================
# This single cell performs all steps without creating new data folders and
# is configured to run exclusively on the CPU.

# --- 1. Import Necessary Libraries ---
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Flatten, RandomFlip, RandomRotation, RandomZoom
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
import os
import matplotlib.pyplot as plt

# --- 2. Setup: Force CPU Usage ---
print("Forcing TensorFlow to use the CPU...")
# Hide all GPUs from TensorFlow's perspective.
# This must be done at the very beginning of the script.
try:
    tf.config.set_visible_devices([], 'GPU')
    logical_devices = tf.config.list_logical_devices('CPU')
    print(f"✅ Successfully configured to use CPU: {len(logical_devices)} logical CPU device(s)")
except RuntimeError as e:
    # Visible devices must be set at program startup
    print(f"Error: {e}")
    print("Could not modify visible devices. This must be done before TensorFlow initializes the GPU.")


# --- 3. Define Paths and Constants ---
# IMPORTANT: Update this path if you are NOT using Google Drive.
INPUT_FOLDER = "Indian_bovine_breeds"
MODEL_FILENAME = 'indian_bovine_vgg16_model_cpu.h5'

# Image and batching parameters
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
VALIDATION_SPLIT = 0.2
TEST_SPLIT = 0.5

# Check if the input folder exists
if not os.path.exists(INPUT_FOLDER) or not os.listdir(INPUT_FOLDER):
    raise FileNotFoundError(f"The specified input folder does not exist or is empty: {INPUT_FOLDER}\nPlease update the INPUT_FOLDER variable with the correct path to your dataset.")

# --- 4. Create Datasets Directly from Directory ---
print("\nCreating datasets directly from source folder...")

# Create the 80% training dataset
training_set = tf.keras.utils.image_dataset_from_directory(
    INPUT_FOLDER,
    validation_split=VALIDATION_SPLIT,
    subset="training",
    seed=42,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE
)

# Create the remaining 20% as a temporary dataset
val_test_temp_set = tf.keras.utils.image_dataset_from_directory(
    INPUT_FOLDER,
    validation_split=VALIDATION_SPLIT,
    subset="validation",
    seed=42,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False  # Disable shuffle to ensure a consistent split
)

class_names = training_set.class_names
num_classes = len(class_names)
print(f"✅ Datasets loaded. Found {num_classes} classes: {class_names}")

# Split the temporary 20% set into validation (10%) and testing (10%)
# Correctly calculate the number of batches for test and validation sets
val_test_cardinality = tf.data.experimental.cardinality(val_test_temp_set)

# Convert the cardinality tensor to a Python integer before doing math
if val_test_cardinality == tf.data.UNKNOWN_CARDINALITY:
    raise ValueError("Could not determine the size of the validation/test dataset. Please check your dataset.")

cardinality_int = val_test_cardinality.numpy()
test_set_size = int(cardinality_int * TEST_SPLIT)

test_set = val_test_temp_set.take(test_set_size)
validation_set = val_test_temp_set.skip(test_set_size)

# --- 5. Configure Datasets for Performance and Augmentation ---
print("\nConfiguring datasets for performance...")

data_augmentation = Sequential([
    RandomFlip("horizontal"),
    RandomRotation(0.1),
    RandomZoom(0.1),
])

def prepare_dataset(ds, augment=False):
    ds = ds.map(lambda x, y: (preprocess_input(x), y), num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.map(lambda x, y: (x, tf.one_hot(y, depth=num_classes)), num_parallel_calls=tf.data.AUTOTUNE)
    if augment:
        ds = ds.map(lambda x, y: (data_augmentation(x, training=True), y), num_parallel_calls=tf.data.AUTOTUNE)
    return ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)

training_set = prepare_dataset(training_set, augment=True)
validation_set = prepare_dataset(validation_set)
test_set = prepare_dataset(test_set)
print("✅ Datasets configured.")

# --- 6. Build and Compile the VGG16 Model ---
print("\nBuilding VGG16 model with transfer learning...")
vgg = VGG16(input_shape=IMAGE_SIZE + (3,), weights='imagenet', include_top=False)
for layer in vgg.layers:
    layer.trainable = False
x = Flatten()(vgg.output)
prediction = Dense(num_classes, activation='softmax')(x)
model = Model(inputs=vgg.input, outputs=prediction)
model.compile(
  loss='categorical_crossentropy',
  optimizer='adam',
  metrics=['accuracy']
)
print("✅ Model built and compiled successfully.")
model.summary()

# --- 7. Train the Model ---
print("\n🐌🚀 Starting model training on CPU (this will be slow)...")
history = model.fit(
  training_set,
  validation_data=validation_set,
  epochs=10
)
print("✅ Training complete.")

# --- 8. Evaluate the Model ---
print("\n📊 Evaluating model on the test set...")
loss, accuracy = model.evaluate(test_set)
print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy*100:.2f}%")

# --- 9. Plot Training History ---
print("\n📈 Plotting training history...")
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.tight_layout()
plt.show()

# --- 10. Export the Model ---
print(f"\n💾 Saving the model as '{MODEL_FILENAME}'...")
model.save(MODEL_FILENAME)
print(f"✅ Model saved successfully! You can find '{MODEL_FILENAME}' in your current directory.")