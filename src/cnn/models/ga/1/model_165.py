from tensorflow.keras import layers
import tensorflow as tf


def create_model(input_size):
    x = tf.keras.Sequential()
    x.add(layers.Conv2D(40, (7, 7), activation='leaky_relu', padding='same', input_shape=(input_size[0], input_size[1], 1)))
    x.add(layers.GlobalAveragePooling2D())
    x.add(layers.Dense(1469, activation='relu'))
    x.add(layers.Dense(773, activation='tanh'))
    x.add(layers.Dropout(0.3))
    x.add(layers.Dense(581, activation='relu'))
    x.add(layers.Dense(1, activation="sigmoid"))
    return x