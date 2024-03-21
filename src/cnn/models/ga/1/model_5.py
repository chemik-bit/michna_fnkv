from tensorflow.keras import layers
import tensorflow as tf


def create_model(input_size):
    x = tf.keras.Sequential()
    x.add(layers.Conv2D(58, (7, 7), activation='relu', padding='same', input_shape=(input_size[0], input_size[1], 1)))
    x.add(layers.BatchNormalization())
    x.add(layers.BatchNormalization())
    x.add(layers.Conv2D(11, (5, 5), activation='tanh', padding='same'))
    x.add(layers.BatchNormalization())
    x.add(layers.BatchNormalization())
    x.add(layers.Conv2D(41, (1, 1), activation='relu', padding='same'))
    x.add(layers.GlobalAveragePooling2D())
    x.add(layers.Dense(1752, activation='relu'))
    x.add(layers.Dropout(0.9))
    x.add(layers.Dense(1150, activation='sigmoid'))
    x.add(layers.Dense(407, activation='sigmoid'))
    x.add(layers.Dense(1, activation="sigmoid"))
    return x