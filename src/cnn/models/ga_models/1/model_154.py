from tensorflow.keras import layers
import tensorflow as tf


def create_model(input_size):
    x = tf.keras.Sequential()
    x.add(layers.Conv2D(52, (3, 3), activation='relu', padding='same', input_shape=(input_size[0], input_size[1], 1)))
    x.add(layers.BatchNormalization())
    x.add(layers.MaxPooling2D(pool_size=(1, 1), strides=(2, 2)))
    x.add(layers.BatchNormalization())
    x.add(layers.GlobalAveragePooling2D())
    x.add(layers.Dense(1570, activation='tanh'))
    x.add(layers.Dropout(0.3))
    x.add(layers.Dense(1425, activation='tanh'))
    x.add(layers.Dense(1519, activation='tanh'))
    x.add(layers.Dense(1, activation="sigmoid"))
    return x