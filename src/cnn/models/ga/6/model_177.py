from tensorflow.keras import layers
import tensorflow as tf


def create_model(input_size):
    x = tf.keras.Sequential()
    x.add(layers.Conv2D(15, (1, 1), activation='relu', padding='same', input_shape=(input_size[0], input_size[1], 1)))
    x.add(layers.BatchNormalization())
    x.add(layers.MaxPooling2D(pool_size=(1, 1), strides=(2, 2)))
    x.add(layers.Conv2D(50, (6, 6), activation='leaky_relu', padding='same'))
    x.add(layers.BatchNormalization())
    x.add(layers.Conv2D(9, (7, 7), activation='relu', padding='same'))
    x.add(layers.BatchNormalization())
    x.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    x.add(layers.BatchNormalization())
    x.add(layers.Conv2D(45, (4, 4), activation='leaky_relu', padding='same'))
    x.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    x.add(layers.BatchNormalization())
    x.add(layers.Conv2D(56, (4, 4), activation='leaky_relu', padding='same'))
    x.add(layers.Conv2D(57, (7, 7), activation='sigmoid', padding='same'))
    x.add(layers.BatchNormalization())
    x.add(layers.MaxPooling2D(pool_size=(1, 1), strides=(2, 2)))
    x.add(layers.GlobalAveragePooling2D())
    x.add(layers.Dense(670, activation='relu'))
    x.add(layers.Dropout(0.7))
    x.add(layers.Dense(1, activation="sigmoid"))
    return x