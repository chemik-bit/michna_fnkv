from tensorflow.keras import layers
import tensorflow as tf


def create_model(input_size):
    x = tf.keras.Sequential()
    x.add(layers.Conv2D(15, (1, 1), activation='sigmoid', padding='same', input_shape=(input_size[0], input_size[1], 1)))
    x.add(layers.BatchNormalization())
    x.add(layers.MaxPooling2D(pool_size=(1, 1), strides=(2, 2)))
    x.add(layers.BatchNormalization())
    x.add(layers.Conv2D(63, (5, 5), activation='leaky_relu', padding='same'))
    x.add(layers.Conv2D(54, (1, 1), activation='sigmoid', padding='same'))
    x.add(layers.Conv2D(9, (4, 4), activation='tanh', padding='same'))
    x.add(layers.Conv2D(62, (7, 7), activation='relu', padding='same'))
    x.add(layers.BatchNormalization())
    x.add(layers.GlobalAveragePooling2D())
    x.add(layers.Dense(244, activation='relu'))
    x.add(layers.Dense(1, activation="sigmoid"))
    return x