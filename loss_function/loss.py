import tensorflow as tf
import numpy as np

# weighted_iou
def iou_loss(y_true, y_pred, smooth=1e-6, alpha=0.7):
    """Compute weighted IoU with foreground and background weighting."""
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    
    # Convert y_true to grayscale if it has 3 channels
    
    #corrected due to channel Feb 28
    #if y_true.shape[-1] == 3:
    #    y_true = tf.image.rgb_to_grayscale(y_true)
    
    # Flatten the tensors for element-wise operations
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])

    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    union = tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) - intersection
    
    # Weighted IoU
    fg_weight = alpha  # Weight for foreground
    bg_weight = 1 - alpha  # Weight for background
    
    fg_intersection = tf.reduce_sum(y_true_f * y_pred_f)
    fg_union = tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) - fg_intersection

    bg_intersection = tf.reduce_sum((1 - y_true_f) * (1 - y_pred_f))
    bg_union = tf.reduce_sum((1 - y_true_f)) + tf.reduce_sum((1 - y_pred_f)) - bg_intersection

    fg_iou = (fg_intersection + smooth) / (fg_union + smooth)
    bg_iou = (bg_intersection + smooth) / (bg_union + smooth)

    return fg_weight * fg_iou + bg_weight * bg_iou





def dice_coef_loss(y_true, y_pred, smooth=1e-15):
    # Ensure inputs are float32
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    #corrected due to channel Feb 28
    # Convert y_true to grayscale if it has 3 channels
    #if y_true.shape[-1] == 3:
    #    y_true = tf.image.rgb_to_grayscale(y_true)

    # Flatten the tensors for element-wise operations
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])

    # Calculate Dice coefficient
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)


    
def improved_combined_loss(y_true, y_pred):
    #corrected due to channel Feb 28
    # Ensure ground truth is single channel
    #if tf.shape(y_true)[-1] == 3:
    #    y_true = tf.image.rgb_to_grayscale(y_true)

    # Binary Crossentropy Loss
    bce = tf.keras.losses.BinaryCrossentropy()(y_true, y_pred)

    # IoU Loss
    iou_loss_value = 1 - iou_loss(y_true, y_pred)

    # Dice Loss
    dice_loss_value = 1 - dice_coef_loss(y_true, y_pred)

    # Weighted combination of losses
    #base line
    combined_loss = 0.4 * bce + 0.3 * iou_loss_value + 0.3 * dice_loss_value
    #combined_loss = 0.3 * bce + 0.4 * iou_loss_value + 0.3 * dice_loss_value
    return combined_loss

