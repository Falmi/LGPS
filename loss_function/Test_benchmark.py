import sys
import time
sys.path.append("../")

import numpy as np
import tensorflow as tf
from tensorflow.keras.metrics import MeanIoU, Recall, Precision, BinaryAccuracy
from models.generator import build_generator
import os

# Allow TensorFlow to use all available GPU memory by setting memory growth before any operation
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Set memory growth before initializing any models or performing any operations
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"Using all available GPU memory on {len(gpus)} GPU(s).")
    except RuntimeError as e:
        print(f"Error: {e}")

# F2 Score calculation
def f2_score(y_true, y_pred):
    precision = Precision()(y_true, y_pred)
    recall = Recall()(y_true, y_pred)
    return (5 * precision * recall) / (4 * precision + recall + tf.keras.backend.epsilon())

# Dice Coefficient calculation
def dice_coefficient(y_true, y_pred, smooth=1e-15): #smooth=1e-6):
    y_true = tf.keras.backend.flatten(y_true)
    y_pred = tf.keras.backend.flatten(y_pred)
    intersection = tf.reduce_sum(y_true * y_pred)
    return (2. * intersection + smooth) / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + smooth)

# Load the dataset
#test_images = np.load('data/Kvasir-SEG/test_images.npy')
#test_masks = np.load('data/Kvasir-SEG/test_masks.npy')

#test_images = np.load('data/CVC-ClinicDB/test_images.npy')
#test_masks = np.load('data/CVC-ClinicDB/test_masks.npy')

#CVC-Colon DB,
# Load the dataset
#test_images = np.load('data/CVC-ColonDB/test_images.npy')
#test_masks = np.load('data/CVC-ColonDB/test_masks.npy')

#ETIS
# Load the dataset
test_images = np.load('data/ETIS/test_images.npy')
test_masks = np.load('data/ETIS/test_masks.npy')

#CVC-300
# Load the dataset
#test_images = np.load('data/CVC-300/test_images.npy')
#test_masks = np.load('data/CVC-300/test_masks.npy')

# Resize images to match the input shape
test_images = tf.image.resize(test_images, (256, 256))
test_masks = tf.image.resize(test_masks, (256, 256))

# Build the generator model
input_shape = (256, 256, 3)
generator = build_generator(input_shape)

# Load the trained weights from the best model (you can change this path to your checkpoint)
checkpoint_path = "/home/avl/Documents/ME/EffiPolpNet/models/checkpoints_BCE_Dice/epoch_49_val_loss_0.8419.h5"
generator.load_weights(checkpoint_path)
print(f"Model loaded from {checkpoint_path}")

# Measure inference time
start_time = time.time()  # Start time before inference

# Inference and evaluation
predictions = generator(test_images, training=False)

# Measure inference time
end_time = time.time()  # End time after inference
inference_time = end_time - start_time  # Calculate the time taken for inference

# Calculate images processed per second
images_per_second = len(test_images) / inference_time

# Convert grayscale to RGB (if needed)
predictions = tf.image.grayscale_to_rgb(predictions)
predictions = tf.cast(predictions > 0.5, tf.float32)

# --- Basic Evaluation Metrics ---
accuracy_metric = BinaryAccuracy()
miou_metric = MeanIoU(num_classes=2)  # Number of classes: 2 (foreground, background)
recall_metric = Recall()
precision_metric = Precision()

# Calculate metrics
accuracy_metric_value = accuracy_metric(test_masks, predictions).numpy()
miou_metric_value = miou_metric(test_masks, predictions).numpy()
recall_metric_value = recall_metric(test_masks, predictions).numpy()
precision_metric_value = precision_metric(test_masks, predictions).numpy()
f2_metric_value = f2_score(test_masks, predictions).numpy()
dice_metric_value = dice_coefficient(test_masks, predictions).numpy()  # Dice Coefficient

# Print results
print(f"Test results - Accuracy: {accuracy_metric_value}")
print(f"Mean IoU: {miou_metric_value}")
print(f"Recall: {recall_metric_value}")
print(f"Precision: {precision_metric_value}")
print(f"F2 Score: {f2_metric_value}")
print(f"Dice Coefficient: {dice_metric_value}")
print(f"Inference time: {inference_time:.4f} seconds")
print(f"Images processed per second: {images_per_second:.2f} images/second")

# Optionally, save predictions as images
# for i in range(len(test_images)):
#     tf.io.write_file(f'predictions/pred_{i}.png', tf.image.encode_png(predictions[i]))

print("Testing completed.")