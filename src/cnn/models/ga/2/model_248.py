from tensorflow.keras import layers
import tensorflow as tf


def create_model(input_size):
    x = tf.keras.Sequential()
    x.add(layers.Conv2D(16, (4, 4), activation='leaky_relu', padding='same', input_shape=(input_size[0], input_size[1], 1)))
    x.add(layers.BatchNormalization())
    x.add(layers.Conv2D(57, (7, 7), activation='relu', padding='same'))
    x.add(layers.BatchNormalization())
    x.add(layers.Conv2D(53, (1, 1), activation='tanh', padding='same'))
    x.add(layers.BatchNormalization())
    x.add(layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
    x.add(layers.Conv2D(22, (2, 2), activation='leaky_relu', padding='same'))
    x.add(layers.BatchNormalization())
    x.add(layers.GlobalAveragePooling2D())
    x.add(layers.Dense(2004, activation='leaky_relu'))
    x.add(layers.Dropout(0.7))
    x.add(layers.Dense(675, activation='tanh'))
    x.add(layers.Dense(1129, activation='sigmoid'))
    x.add(layers.Dropout(0.9))
    x.add(layers.Dense(1, activation="sigmoid"))
    return x