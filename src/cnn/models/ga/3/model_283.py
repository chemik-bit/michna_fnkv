from tensorflow.keras import layers
import tensorflow as tf


def create_model(input_size):
    x = tf.keras.Sequential()
    x.add(layers.Conv2D(1, (4, 4), activation='relu', padding='same', input_shape=(input_size[0], input_size[1], 1)))
    x.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    x.add(layers.GlobalAveragePooling2D())
    x.add(layers.Dense(1251, activation='relu'))
    x.add(layers.Dropout(0.8))
    x.add(layers.Dense(2037, activation='tanh'))
    x.add(layers.Dropout(0.2))
    x.add(layers.Dense(976, activation='leaky_relu'))
    x.add(layers.Dense(1, activation="sigmoid"))
    return x