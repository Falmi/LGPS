import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Conv2D, BatchNormalization, ReLU, Add, UpSampling2D, Concatenate, GlobalAveragePooling2D, Dense, Reshape
)
from tensorflow.keras.models import Model

def modified_residual_block(x, filters):
    """
    Custom residual block with 1x1 and 3x3 convolutions without Squeeze & Excitation.

    Args:
        x (Tensor): Input tensor.
        filters (int): Number of filters for the convolution layers.

    Returns:
        Tensor: Output tensor after applying the residual block.
    """
    shortcut = x  # Save the input for the residual connection

    # 1x1 convolution (bottleneck)
    x = Conv2D(filters // 4, (1, 1), padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    # 3x3 convolution
    x = Conv2D(filters // 4, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)

    # Adjust the shortcut dimensions to match the processed tensor
    if shortcut.shape[-1] != x.shape[-1]:
        shortcut = Conv2D(filters // 4, (1, 1), padding='same')(shortcut)
        shortcut = BatchNormalization()(shortcut)

    # Add the shortcut to the processed tensor
    x = Add()([shortcut, x])
    x = ReLU()(x)

    return x


# Build Encoder using MobileNetV2
def build_encoder(input_shape):
    """Encoder using pre-trained MobileNetV2."""
    base_model = tf.keras.applications.MobileNetV2(input_shape=input_shape, include_top=False, weights='imagenet')
    encoder_output = base_model.get_layer('block_13_expand_relu').output
    skip1 = base_model.get_layer('block_1_expand_relu').output
    skip2 = base_model.get_layer('block_3_expand_relu').output
    skip3 = base_model.get_layer('block_6_expand_relu').output
    return Model(inputs=base_model.input, outputs=[encoder_output, skip1, skip2, skip3])

def build_generator(input_shape):
    inputs = Input(shape=input_shape)
    encoder = build_encoder(input_shape)
    encoder_output, skip1, skip2, skip3 = encoder(inputs)

    # Bridge (Modified Residual Block)
    bottleneck = modified_residual_block(encoder_output, 576)

    # Decoder
    up1 = UpSampling2D(size=(2, 2), interpolation='bilinear')(bottleneck)
    up1 = Concatenate()([up1, skip3])  # Skip Connection
    up1 = modified_residual_block(up1, 256)

    up2 = UpSampling2D(size=(2, 2), interpolation='bilinear')(up1)
    up2 = Concatenate()([up2, skip2])  # Skip Connection
    up2 = modified_residual_block(up2, 128)

    up3 = UpSampling2D(size=(2, 2), interpolation='bilinear')(up2)
    up3 = Concatenate()([up3, skip1])  # Skip Connection
    up3 = modified_residual_block(up3, 64)

    up4 = UpSampling2D(size=(2, 2), interpolation='bilinear')(up3)
    outputs = Conv2D(1, (1, 1), activation='sigmoid')(up4)  # Binary Segmentation

    return Model(inputs=inputs, outputs=outputs)

# Example Usage
input_shape = (256, 256, 3)
generator_model = build_generator(input_shape)
generator_model.summary()