import os
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from scipy.ndimage import gaussian_filter, map_coordinates
import random
from PIL import Image
import tifffile

def load_data_from_txt(txt_file, img_size, normalize=True):
    images = []
    masks = []
    with open(txt_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            img_name = line.strip()
            img_path = os.path.join('CVC-ClinicDB', 'Original', img_name )
            mask_path = os.path.join('CVC-ClinicDB', 'GroundTruth', img_name )
            
            # Load and normalize image
            img = tifffile.imread(img_path)
            # Resize the image (e.g., resize to 256x256)
            #resized_image = img.resize((256, 256))
            pil_image = Image.fromarray(img)

            # Resize the image (e.g., resize to 256x256)
            resized_image = pil_image.resize((256, 256))
            
            
            img = img_to_array(resized_image)
            if normalize:
                img = (img / 127.5) - 1.0  # Normalize to [-1, 1]
            else:
                img = img / 255.0  # Optional: Normalize to [0, 1] (if preferred)
            
            # Load and normalize mask
            mask = tifffile.imread(mask_path)
            #grayscale_image = mask.mean(axis=-1).astype('uint8')
            pil_image = Image.fromarray(mask)
            resized_image = pil_image.resize((256, 256))
            mask = img_to_array(resized_image)

            
            mask = mask / 255.0  # Normalize mask to [0, 1]
            
            # Convert mask to 3 channels
            mask = tf.convert_to_tensor(mask)  # Convert mask to tensor
            if mask.shape[-1] != 3:
                mask = tf.image.grayscale_to_rgb(mask)  # Convert grayscale mask to RGB (3 channels)
            
            mask = tf.keras.backend.eval(mask)  # Convert back to NumPy array after tensor operation
            
            images.append(img)
            masks.append(mask)
    
    return np.array(images), np.array(masks)

def random_crop(image, mask, crop_size):
    """Ensure image and mask are cropped properly"""
    # Check if the crop size is smaller than the image dimensions
    if image.shape[0] < crop_size or image.shape[1] < crop_size:
        raise ValueError(f"Crop size ({crop_size}) is larger than image dimensions ({image.shape[0]}, {image.shape[1]})")
    
    combined = tf.concat([image, mask], axis=-1)
    cropped_combined = tf.image.random_crop(combined, size=[crop_size, crop_size, combined.shape[-1]])
    return cropped_combined[..., :-3], cropped_combined[..., -3:]

def elastic_deformation(image, mask, alpha=34, sigma=4):
    """Apply elastic deformation to image and mask."""
    random_state = np.random.RandomState(None)
    
    # Generate random displacement fields
    shape = image.shape
    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    dz = np.zeros_like(dx)  # No deformation in the third axis
    
    # Apply displacement fields to image and mask using map_coordinates
    x, y, z = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), np.arange(shape[2]), indexing="ij")
    distorted_image = map_coordinates(image, [x + dx, y + dy, z + dz], order=1, mode='reflect')
    distorted_mask = map_coordinates(mask, [x + dx, y + dy, z + dz], order=1, mode='reflect')

    return distorted_image, distorted_mask


def random_zoom(image, mask, zoom_range=(0.8, 1.2)):
    """Randomly zoom into the image and mask to highlight foreground objects."""
    zoom_factor = random.uniform(zoom_range[0], zoom_range[1])
    
    # Resize the image to a larger size if zooming down
    new_height = int(image.shape[0] * zoom_factor)
    new_width = int(image.shape[1] * zoom_factor)
    
    # Ensure the zoomed image is at least the crop size (256, 256)
    if new_height < 256 or new_width < 256:
        new_height, new_width = 256, 256
    
    # Resize image and mask
    image_zoomed = tf.image.resize(image, (new_height, new_width))
    mask_zoomed = tf.image.resize(mask, (new_height, new_width))
    
    # Randomly crop back to the original size
    crop_size = image.shape[0]
    image_zoomed, mask_zoomed = random_crop(image_zoomed, mask_zoomed, crop_size)
    
    return image_zoomed, mask_zoomed

def intensity_transform(image, mask, brightness_factor=(0.8, 1.2), contrast_factor=(0.8, 1.2)):
    """Enhance the contrast and brightness of the foreground."""
    # Random brightness adjustment
    brightness_factor = random.uniform(brightness_factor[0], brightness_factor[1])
    image = tf.image.adjust_brightness(image, delta=brightness_factor - 1)
    
    # Random contrast adjustment
    contrast_factor = random.uniform(contrast_factor[0], contrast_factor[1])
    image = tf.image.adjust_contrast(image, contrast_factor)
    
    return image, mask

def augment(image, mask):
    # Apply random zoom
    image, mask = random_zoom(image, mask)
    
    # Apply intensity transformations (contrast & brightness)
    image, mask = intensity_transform(image, mask)
    
    # Apply elastic deformations
    image, mask = elastic_deformation(image, mask)
    
    return image, mask

# Define the data augmentation techniques
data_gen_args = dict(
    rotation_range=20,  # Random rotation between 0 and 20 degrees
    width_shift_range=0.1,  # Random horizontal shift
    height_shift_range=0.1,  # Random vertical shift
    shear_range=0.2,  # Shear transformations
    zoom_range=0.2,  # Random zoom
    horizontal_flip=True,  # Random horizontal flip
    vertical_flip=True,  # Random vertical flip
    fill_mode='nearest',  # Fill mode for points outside the boundaries
    brightness_range=[0.8, 1.2],  # Random brightness adjustment
    channel_shift_range=0.1,  # Random channel shift
    rescale=1./255,  # Rescale the image values to [0, 1]
    preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input  # Preprocessing function for MobileNetV2
)

# Create an ImageDataGenerator for the training set
train_datagen = ImageDataGenerator(**data_gen_args)

if __name__ == "__main__":
    img_size = (256, 256)

    # Load training data
    train_images, train_masks = load_data_from_txt('CVC-ClinicDB/train.txt', img_size, normalize=True)
    
    # Apply data augmentation only on training set
    augmented_images = []
    augmented_masks = []
    for img, mask in zip(train_images, train_masks):
        aug_img, aug_mask = augment(img, mask)
        augmented_images.append(aug_img)
        augmented_masks.append(aug_mask)
    
    train_images = np.array(augmented_images)
    train_masks = np.array(augmented_masks)
    
    train_generator = train_datagen.flow(train_images, train_masks, batch_size=8)
    
    np.save('CVC-ClinicDB/train_images.npy', train_images)
    np.save('CVC-ClinicDB/train_masks.npy', train_masks)

    # Load validation data with normalization
    val_images, val_masks = load_data_from_txt('CVC-ClinicDB/test.txt', img_size, normalize=True)
    
    np.save('CVC-ClinicDB/test_images.npy', val_images)
    np.save('CVC-ClinicDB/test_masks.npy', val_masks)