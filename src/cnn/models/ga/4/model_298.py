from tensorflow.keras import layers
import tensorflow as tf


def create_model(input_size):
    x = tf.keras.Sequential()
    x.add(layers.Conv2D(27, (1, 1), activation='leaky_relu', padding='same', input_shape=(input_size[0], input_size[1], 1)))
    x.add(layers.MaxPooling2D(pool_size=(1, 1), strides=(2, 2)))
    x.add(layers.GlobalAveragePooling2D())
    x.add(layers.Dense(378, activation='leaky_relu'))
    x.add(layers.Dense(1200, activation='leaky_relu'))
    x.add(layers.Dropout(0.8))
    x.add(layers.Dense(1, activation="sigmoid"))
    return x