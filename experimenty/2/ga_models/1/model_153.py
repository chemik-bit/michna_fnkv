from tensorflow.keras import layers
import tensorflow as tf


def create_model(input_size):
    x = tf.keras.Sequential()
    x.add(layers.Conv2D(62, (7, 7), activation='leaky_relu', padding='same', input_shape=(input_size[0], input_size[1], 1)))
    x.add(layers.BatchNormalization())
    x.add(layers.BatchNormalization())
    x.add(layers.Conv2D(56, (5, 5), activation='sigmoid', padding='same'))
    x.add(layers.BatchNormalization())
    x.add(layers.GlobalAveragePooling2D())
    x.add(layers.Dense(2045, activation='tanh'))
    x.add(layers.Dropout(0.6))
    x.add(layers.Dense(1610, activation='leaky_relu'))
    x.add(layers.Dropout(0.6))
    x.add(layers.Dense(384, activation='relu'))
    x.add(layers.Dense(1, activation="sigmoid"))
    return x