import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from models.generator_With_out_SE import build_generator
from models.DiscriminatorWithConvCRF import DiscriminatorWithConvCRF
from loss_function.loss import improved_combined_loss
from tensorflow.keras.metrics import MeanIoU, Recall, Precision, BinaryAccuracy
from Evalution_Metrics.Metrics import dice_coefficient
import datetime
import os
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler


gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_virtual_device_configuration(
                gpu,
                [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=48000)]  # 48 GB memory limit
            )
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        print(e)

# Load the dataset
train_images = np.load('data/Seg_Cvc/train_images.npy')
train_masks = np.load('data/Seg_Cvc/train_masks.npy')
val_images = np.load('data/Kvasir-SEG/test_images.npy')
val_masks = np.load('data/Kvasir-SEG/test_masks.npy')

# Resize images to match the input shape
train_images = tf.image.resize(train_images, (256, 256))
train_masks = tf.image.resize(train_masks, (256, 256))
val_images = tf.image.resize(val_images, (256, 256))
val_masks = tf.image.resize(val_masks, (256, 256))

# Preprocess the images by normalizing them using the custom preprocess function
# train_images = preprocess_image(train_images)
# val_images = preprocess_image(val_images)

# Build the SegGAN model
input_shape = (256, 256, 3)
generator = build_generator(input_shape)
discriminator = DiscriminatorWithConvCRF(input_shape)

# Compile the models
initial_learning_rate = 3e-4
gen_optimizer = Adam(learning_rate=initial_learning_rate, beta_1=0.5, clipvalue=1.0)
disc_optimizer = Adam(learning_rate=initial_learning_rate, beta_1=0.5, clipvalue=1.0)

#generator.compile(optimizer=gen_optimizer, loss=improved_combined_loss, metrics=[dice_coef_loss, iou_loss])
generator.compile(optimizer=gen_optimizer, loss=improved_combined_loss, metrics=[MeanIoU(num_classes=2), dice_coefficient])
discriminator.compile(optimizer=disc_optimizer, loss=tf.keras.losses.BinaryCrossentropy())

# TensorBoard setup
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_writer = tf.summary.create_file_writer(log_dir)

# Checkpoint setup
checkpoint_dir = 'models/checkpoints'
os.makedirs(checkpoint_dir, exist_ok=True)

# Learning rate scheduler
def scheduler(epoch, lr):
    if epoch > 50:
        return lr * 0.9  # Reduce learning rate after 50 epochs
    return lr

lr_scheduler = LearningRateScheduler(scheduler)

# Early stopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Training loop
epochs = 200
batch_size = 16
best_val_loss = float('inf')  # Track the best validation loss
patience = 3
factor = 0.5
min_lr = 1e-6
no_improve_counter = 0

for epoch in range(epochs):
    print(f"Epoch {epoch+1}/{epochs}")
    epoch_gen_loss = 0
    epoch_disc_loss = 0
    epoch_dice = 0
    epoch_iou = 0
    steps_per_epoch = len(train_images) // batch_size

    for i in range(0, len(train_images), batch_size):
        image_batch = train_images[i:i + batch_size]
        mask_batch = train_masks[i:i + batch_size]
        
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            # Generator forward pass
            fake_masks = generator(image_batch, training=True)
            fake_masks = tf.image.grayscale_to_rgb(fake_masks)

            # Prepare inputs for the discriminator
            real_output = discriminator([image_batch, mask_batch], training=True)
            fake_output = discriminator([image_batch, fake_masks], training=True)

            # Loss computation
            gen_loss = improved_combined_loss(mask_batch, fake_masks)

            # Label smoothing for discriminator
            real_labels = tf.ones_like(real_output) * 0.9  # Apply label smoothing
            fake_labels = tf.zeros_like(fake_output)

            disc_loss_real = tf.keras.losses.BinaryCrossentropy()(real_labels, real_output)
            disc_loss_fake = tf.keras.losses.BinaryCrossentropy()(fake_labels, fake_output)
            disc_loss = disc_loss_real + disc_loss_fake

        # Backpropagation and weight updates
        gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

        gen_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
        disc_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

        epoch_gen_loss += gen_loss
        epoch_disc_loss += disc_loss
        
        
        fake_masks = tf.cast(fake_masks > 0.5, tf.float32)
    
        
        miou_metric = MeanIoU(num_classes=2)
            
        batch_dice = dice_coefficient(mask_batch, fake_masks).numpy()
        batch_iou = miou_metric(mask_batch, fake_masks).numpy()
        epoch_dice += batch_dice
        epoch_iou += batch_iou
        
        # Log images and metrics to TensorBoard for this batch
        with tensorboard_writer.as_default():
            tf.summary.image("Input Image", image_batch, step=i)
            tf.summary.image("Ground Truth Mask", mask_batch, step=i)
            tf.summary.image("Generated Mask", tf.cast(fake_masks * 255, tf.uint8), step=i)
            tf.summary.scalar("Generator Loss", gen_loss.numpy(), step=i)
            tf.summary.scalar("Discriminator Loss", disc_loss.numpy(), step=i)
            tf.summary.scalar("Dice Coefficient", batch_dice, step=i)
            tf.summary.scalar("Mean IoU", batch_iou, step=i)

    # Average metrics for the epoch
    avg_gen_loss = epoch_gen_loss / steps_per_epoch
    avg_disc_loss = epoch_disc_loss / steps_per_epoch
    avg_dice = epoch_dice / steps_per_epoch
    avg_iou = epoch_iou / steps_per_epoch

    print(f"Epoch {epoch+1}: Generator Loss: {avg_gen_loss:.4f}, Discriminator Loss: {avg_disc_loss:.4f}, Dice: {avg_dice:.4f}, IoU: {avg_iou:.4f}")

    # Validation evaluation
    
    val_predictions = generator(val_images, training=False)
    val_predictions = tf.image.grayscale_to_rgb(val_predictions)
    val_predictions_bin = tf.cast(val_predictions > 0.5, tf.float32)
    
    # Convert y_true to grayscale if it has 3 channels
    #if val_masks.shape[-1] == 3:
    #    val_masks = tf.image.rgb_to_grayscale(val_masks)
    miou_metric = MeanIoU(num_classes=2)
    
       
    val_loss = improved_combined_loss(val_masks, val_predictions_bin).numpy()
    val_dice = dice_coefficient(val_masks, val_predictions_bin).numpy()
    val_iou = miou_metric(val_masks, val_predictions_bin).numpy()

    print(f"Validation - Loss: {val_loss:.4f}, Dice: {val_dice:.4f}, IoU: {val_iou:.4f}")

    # Log metrics and images to TensorBoard
    with tensorboard_writer.as_default():
        tf.summary.scalar('Training Generator Loss', avg_gen_loss, step=epoch)
        tf.summary.scalar('Training Dice Coefficient', avg_dice, step=epoch)
        tf.summary.scalar('Training IoU', avg_iou, step=epoch)
        tf.summary.scalar('Validation Loss', val_loss, step=epoch)
        tf.summary.scalar('Validation Dice Coefficient', val_dice, step=epoch)
        tf.summary.scalar('Validation IoU', val_iou, step=epoch)

        # Log validation images and predicted masks
        tf.summary.image("Validation Images", val_images[:5], step=epoch)
        tf.summary.image("Validation Masks", val_masks[:5], step=epoch)
        tf.summary.image("Validation Predictions", val_predictions_bin[:5], step=epoch)

    # Save the best model based on validation loss
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        no_improve_counter = 0
        checkpoint_path = os.path.join(checkpoint_dir, f'epoch_{epoch+1}_val_loss_{val_loss:.4f}.h5')
        generator.save(checkpoint_path)
        print(f"Model saved: {checkpoint_path}")
    else:
        no_improve_counter += 1
        if no_improve_counter >= patience:
            current_lr = gen_optimizer.learning_rate.numpy()
            new_lr = max(current_lr * factor, min_lr)
            gen_optimizer.learning_rate.assign(new_lr)
            print(f"Reduced learning rate to {new_lr:.6f}")
            no_improve_counter = 0
