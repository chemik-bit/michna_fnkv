from tensorflow.keras import layers
from tensorflow.keras.backend import epsilon
from tensorflow.math import reduce_sum, square, maximum, sqrt
from tensorflow.keras.applications.vgg16 import VGG16
import tensorflow as tf
from tensorflow import keras

def create_model(input_size):
    base_model =tf.keras.applications.vgg16.VGG16(include_top=False,
           input_shape=(input_size, input_size, 3), pooling='max', weights='imagenet')
    base_model.trainable = True  ## Not trainable weights
    top_model = base_model.output
    x = layers.Flatten()(top_model)
    x = layers.Dense(4096, activation="relu", kernel_regularizer=keras.regularizers.l2(0.1))(x)
    x = layers.Dense(4096, activation="relu", kernel_regularizer=keras.regularizers.l2(0.1))(x)
    x = layers.Dense(4096, activation="relu", kernel_regularizer=keras.regularizers.l2(0.1))(x)
    x = layers.Dense(1, activation="sigmoid")(x)
    model = tf.keras.Model(inputs=base_model.input, outputs=x)
    return model