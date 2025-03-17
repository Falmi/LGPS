import tensorflow as tf
from tensorflow.keras.layers import Conv2D
from tensorflow.keras import layers

class ConvCRF(layers.Layer):
    def __init__(self, num_classes=1, kernel_size=(3, 3), **kwargs):
        super(ConvCRF, self).__init__(**kwargs)
        self.num_classes = num_classes
        self.kernel_size = kernel_size
        self.conv = Conv2D(filters=num_classes, kernel_size=self.kernel_size, strides=1, padding='same')

    def call(self, inputs):
        # Apply convolution to refine the features
        x = self.conv(inputs)
        
        # Use sigmoid for binary classification (0 or 1)
        refined_output = tf.sigmoid(x)  # Sigmoid activation instead of softmax
        return refined_output
