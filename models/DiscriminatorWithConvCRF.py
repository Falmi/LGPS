import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, LeakyReLU, BatchNormalization, Concatenate
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
# Assuming ConvCRF is imported from your custom module or library
from convcrf import ConvCRF  # Ensure that the ConvCRF layer is correctly imported

class DiscriminatorWithConvCRF(Model):
    def __init__(self, input_shape, num_classes=1, **kwargs):
        super(DiscriminatorWithConvCRF, self).__init__(**kwargs)
        
        # Convolutional layers
        self.conv1 = Conv2D(64, (3, 3), strides=2, padding='same')
        self.conv2 = Conv2D(128, (3, 3), strides=2, padding='same')
        self.conv3 = Conv2D(256, (3, 3), strides=2, padding='same')
        self.conv4 = Conv2D(512, (3, 3), strides=2, padding='same')
        self.conv5 = Conv2D(1, (3, 3), strides=2, padding='same')  # Final output layer
        
        # LeakyReLU activation
        self.leaky_relu = LeakyReLU(alpha=0.2)
        
        # ConvCRF layers
        self.convcrf1 = ConvCRF(num_classes=num_classes)
        self.convcrf2 = ConvCRF(num_classes=num_classes)
        self.convcrf3 = ConvCRF(num_classes=num_classes)
        self.convcrf4 = ConvCRF(num_classes=num_classes)
        
    def call(self, inputs):
        # Unpack the inputs
        input_image, target_image = inputs  # inputs should be a list of two tensors
        
        # Concatenate the input and target images
        x = Concatenate()([input_image, target_image])
        
        # Apply convolution layers with LeakyReLU
        x = self.conv1(x)
        x = self.leaky_relu(x)
        
        x = self.conv2(x)
        x = self.leaky_relu(x)
        
        x = self.conv3(x)
        x = self.leaky_relu(x)
        
        x = self.conv4(x)
        x = self.leaky_relu(x)
        
        x = self.conv5(x)  # Final output layer
        
        # Apply ConvCRF layers for refinement
        x = self.convcrf1(x)
        x = self.convcrf2(x)
        x = self.convcrf3(x)
        x = self.convcrf4(x)
        
        # Apply sigmoid for binary classification (0 or 1)
        x = tf.sigmoid(x)  # Sigmoid activation for binary classification (0 or 1)
        
        return x
