from tensorflow.keras import layers
import tensorflow as tf


def create_model(input_size):
    x = tf.keras.Sequential()
    x.add(layers.Conv2D(55, (5, 5), activation='sigmoid', padding='same', input_shape=(input_size[0], input_size[1], 1)))
    x.add(layers.BatchNormalization())
    x.add(layers.Conv2D(38, (3, 3), activation='tanh', padding='same'))
    x.add(layers.Conv2D(50, (3, 3), activation='relu', padding='same'))
    x.add(layers.BatchNormalization())
    x.add(layers.BatchNormalization())
    x.add(layers.GlobalAveragePooling2D())
    x.add(layers.Dense(215, activation='relu'))
    x.add(layers.Dropout(0.4))
    x.add(layers.Dense(1685, activation='relu'))
    x.add(layers.Dense(816, activation='relu'))
    x.add(layers.Dropout(0.4))
    x.add(layers.Dense(1, activation="sigmoid"))
    return x