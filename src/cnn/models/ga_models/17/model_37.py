from tensorflow.keras import layers
import tensorflow as tf


def create_model(input_size):
    x = tf.keras.Sequential()
    x.add(layers.Conv2D(60, (4, 4), activation='relu', padding='same', input_shape=(input_size[0], input_size[1], 1)))
    x.add(layers.BatchNormalization())
    x.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    x.add(layers.Conv2D(57, (3, 3), activation='relu', padding='same'))
    x.add(layers.Conv2D(38, (7, 7), activation='relu', padding='same'))
    x.add(layers.BatchNormalization())
    x.add(layers.Conv2D(10, (3, 3), activation='sigmoid', padding='same'))
    x.add(layers.BatchNormalization())
    x.add(layers.BatchNormalization())
    x.add(layers.GlobalAveragePooling2D())
    x.add(layers.Dense(1324, activation='tanh'))
    x.add(layers.Dropout(0.2))
    x.add(layers.Dense(175, activation='tanh'))
    x.add(layers.Dense(981, activation='tanh'))
    x.add(layers.Dense(1, activation="sigmoid"))
    return x