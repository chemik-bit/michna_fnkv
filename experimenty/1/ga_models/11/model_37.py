from tensorflow.keras import layers
import tensorflow as tf


def create_model(input_size):
    x = tf.keras.Sequential()
    x.add(layers.Conv2D(44, (4, 4), activation='relu', padding='same', input_shape=(input_size[0], input_size[1], 1)))
    x.add(layers.BatchNormalization())
    x.add(layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
    x.add(layers.GlobalAveragePooling2D())
    x.add(layers.Dense(570, activation='tanh'))
    x.add(layers.Dropout(0.5))
    x.add(layers.Dense(228, activation='leaky_relu'))
    x.add(layers.Dropout(0.2))
    x.add(layers.Dense(709, activation='relu'))
    x.add(layers.Dense(1, activation="sigmoid"))
    return x