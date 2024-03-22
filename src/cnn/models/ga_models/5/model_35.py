from tensorflow.keras import layers
import tensorflow as tf


def create_model(input_size):
    x = tf.keras.Sequential()
    x.add(layers.Conv2D(59, (1, 1), activation='sigmoid', padding='same', input_shape=(input_size[0], input_size[1], 1)))
    x.add(layers.BatchNormalization())
    x.add(layers.BatchNormalization())
    x.add(layers.GlobalAveragePooling2D())
    x.add(layers.Dense(512, activation='leaky_relu'))
    x.add(layers.Dropout(0.7))
    x.add(layers.Dense(1857, activation='tanh'))
    x.add(layers.Dropout(0.3))
    x.add(layers.Dense(808, activation='relu'))
    x.add(layers.Dense(1, activation="sigmoid"))
    return x