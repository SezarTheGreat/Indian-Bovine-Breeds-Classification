import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import Callback
import os
import time
import pickle
import numpy as np
import sys

# 0. Print Library Versions
print("--- Library Versions ---")
print(f"Python: {sys.version}")
print(f"TensorFlow: {tf.__version__}")
print(f"NumPy: {np.__version__}")
print("------------------------\n")

# 1. Force TensorFlow to use the CPU
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
# To make sure it runs on CPU
tf.config.set_visible_devices([], 'GPU')
print("TensorFlow will use CPU.")


# 2. Define data paths and parameters
data_dir = 'Indian_bovine_breeds'
img_height, img_width = 224, 224
batch_size = 32

# Check if the dataset directory exists
if not os.path.isdir(data_dir):
    print(f"Error: Dataset directory not found at '{data_dir}'")
    print("Please make sure the 'Indian_bovine_breeds' folder is in the same directory as the script.")
    exit()

# 3. Create datasets for training, validation, and testing (80/10/10 split)
# Load the full dataset first
full_dataset = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size
)
class_names = full_dataset.class_names
num_classes = len(class_names)
print(f"Found {num_classes} classes: {class_names}")

# Create a 80% training set and a 20% temporary set (for validation and testing)
dataset_size = tf.data.experimental.cardinality(full_dataset).numpy()
train_size = int(0.8 * dataset_size)
temp_size = dataset_size - train_size

# Shuffle the dataset before splitting
full_dataset = full_dataset.shuffle(buffer_size=dataset_size * batch_size, seed=123)

train_dataset = full_dataset.take(train_size)
temp_dataset = full_dataset.skip(train_size)

# Split the 20% temporary set into 10% validation and 10% testing
val_size = temp_size // 2
test_size = temp_size - val_size
val_dataset = temp_dataset.take(val_size)
test_dataset = temp_dataset.skip(val_size)

# Define a function to apply ResNet50 preprocessing
def preprocess_data(images, labels):
    return tf.keras.applications.resnet50.preprocess_input(images), labels

# Apply preprocessing and optimize dataset performance
AUTOTUNE = tf.data.AUTOTUNE
train_dataset = train_dataset.map(preprocess_data, num_parallel_calls=AUTOTUNE).prefetch(buffer_size=AUTOTUNE)
val_dataset = val_dataset.map(preprocess_data, num_parallel_calls=AUTOTUNE).prefetch(buffer_size=AUTOTUNE)
test_dataset = test_dataset.map(preprocess_data, num_parallel_calls=AUTOTUNE).prefetch(buffer_size=AUTOTUNE)

print(f"Using {tf.data.experimental.cardinality(train_dataset).numpy() * batch_size} images for training.")
print(f"Using {tf.data.experimental.cardinality(val_dataset).numpy() * batch_size} images for validation.")
print(f"Using {tf.data.experimental.cardinality(test_dataset).numpy() * batch_size} images for testing.")


# 4. Load pre-trained ResNet50 and build the new model
# Load ResNet50 with 'imagenet' weights, without the top classification layer
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(img_height, img_width, 3))

# Freeze the layers of the base model
base_model.trainable = False

# Add custom layers on top of the base model
inputs = Input(shape=(img_height, img_width, 3))
x = base_model(inputs, training=False) # Set training=False for frozen layers
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(num_classes, activation='softmax')(x)

# This is the model we will train
model = Model(inputs=inputs, outputs=predictions)

# 5. Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# 6. Set up custom callback to save the best model as a .pkl file
best_model_path_pkl = 'best_resnet50_bovine_classifier.pkl'

class PickleModelCheckpoint(Callback):
    def __init__(self, filepath, monitor='val_accuracy'):
        super(PickleModelCheckpoint, self).__init__()
        self.filepath = filepath
        self.monitor = monitor
        self.best = -np.inf

    def on_epoch_end(self, epoch, logs=None):
        current_val_acc = logs.get(self.monitor)
        if current_val_acc > self.best:
            print(f'\nEpoch {epoch+1}: {self.monitor} improved from {self.best:.4f} to {current_val_acc:.4f}, saving model to {self.filepath}')
            self.best = current_val_acc
            with open(self.filepath, 'wb') as f:
                pickle.dump(self.model, f)

checkpoint = PickleModelCheckpoint(best_model_path_pkl)

# 7. Training loop
num_epochs = 20
print("\nStarting model training...")
start_time = time.time()

history = model.fit(
    train_dataset,
    epochs=num_epochs,
    validation_data=val_dataset,
    callbacks=[checkpoint]
)

training_time = time.time() - start_time
print(f"\nTraining finished in {training_time // 60:.0f}m {training_time % 60:.0f}s")

# 8. Evaluate the best model on the test set
print(f"\nLoading best model from '{best_model_path_pkl}' and evaluating on the test set...")
with open(best_model_path_pkl, 'rb') as f:
    best_model = pickle.load(f)

# Re-compile the model after loading, which is good practice for pickled models
best_model.compile(optimizer=Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

test_loss, test_acc = best_model.evaluate(test_dataset)
print(f"\nTest Accuracy: {test_acc:.4f}")
print(f"Test Loss: {test_loss:.4f}")

print(f"\nBest model saved successfully to '{best_model_path_pkl}'")