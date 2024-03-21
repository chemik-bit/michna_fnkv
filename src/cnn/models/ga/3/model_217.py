from tensorflow.keras import layers
import tensorflow as tf


def create_model(input_size):
    x = tf.keras.Sequential()
    x.add(layers.Conv2D(4, (5, 5), activation='sigmoid', padding='same', input_shape=(input_size[0], input_size[1], 1)))
    x.add(layers.Conv2D(20, (5, 5), activation='leaky_relu', padding='same'))
    x.add(layers.BatchNormalization())
    x.add(layers.MaxPooling2D(pool_size=(1, 1), strides=(2, 2)))
    x.add(layers.Conv2D(3, (6, 6), activation='sigmoid', padding='same'))
    x.add(layers.MaxPooling2D(pool_size=(1, 1), strides=(2, 2)))
    x.add(layers.Conv2D(45, (6, 6), activation='relu', padding='same'))
    x.add(layers.BatchNormalization())
    x.add(layers.Conv2D(46, (7, 7), activation='relu', padding='same'))
    x.add(layers.Conv2D(10, (2, 2), activation='tanh', padding='same'))
    x.add(layers.MaxPooling2D(pool_size=(1, 1), strides=(2, 2)))
    x.add(layers.BatchNormalization())
    x.add(layers.GlobalAveragePooling2D())
    x.add(layers.Dense(739, activation='relu'))
    x.add(layers.Dropout(0.7))
    x.add(layers.Dense(1128, activation='tanh'))
    x.add(layers.Dropout(0.3))
    x.add(layers.Dense(1463, activation='sigmoid'))
    x.add(layers.Dropout(0.5))
    x.add(layers.Dense(1, activation="sigmoid"))
    return x