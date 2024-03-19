from tensorflow.keras import layers
import tensorflow as tf


def create_model(input_size):
    x = tf.keras.Sequential()
    x.add(layers.Conv2D(36, (6, 6), activation='sigmoid', padding='same', input_shape=(input_size[0], input_size[1], 1)))
    x.add(layers.BatchNormalization())
    x.add(layers.GlobalAveragePooling2D())
    x.add(layers.Dense(774, activation='tanh'))
    x.add(layers.Dropout(0.7))
    x.add(layers.Dense(907, activation='relu'))
    x.add(layers.Dense(1, activation="sigmoid"))
    return x