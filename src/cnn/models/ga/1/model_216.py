from tensorflow.keras import layers
import tensorflow as tf


def create_model(input_size):
    x = tf.keras.Sequential()
    x.add(layers.Conv2D(49, (5, 5), activation='sigmoid', padding='same', input_shape=(input_size[0], input_size[1], 1)))
    x.add(layers.BatchNormalization())
    x.add(layers.BatchNormalization())
    x.add(layers.Conv2D(54, (3, 3), activation='tanh', padding='same'))
    x.add(layers.BatchNormalization())
    x.add(layers.BatchNormalization())
    x.add(layers.Conv2D(41, (1, 1), activation='relu', padding='same'))
    x.add(layers.BatchNormalization())
    x.add(layers.Conv2D(44, (3, 3), activation='relu', padding='same'))
    x.add(layers.BatchNormalization())
    x.add(layers.GlobalAveragePooling2D())
    x.add(layers.Dense(246, activation='sigmoid'))
    x.add(layers.Dropout(0.4))
    x.add(layers.Dense(1, activation="sigmoid"))
    return x