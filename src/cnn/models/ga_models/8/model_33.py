from tensorflow.keras import layers
import tensorflow as tf


def create_model(input_size):
    x = tf.keras.Sequential()
    x.add(layers.Conv2D(16, (7, 7), activation='tanh', padding='same', input_shape=(input_size[0], input_size[1], 1)))
    x.add(layers.BatchNormalization())
    x.add(layers.Conv2D(42, (6, 6), activation='sigmoid', padding='same'))
    x.add(layers.BatchNormalization())
    x.add(layers.BatchNormalization())
    x.add(layers.Conv2D(17, (5, 5), activation='tanh', padding='same'))
    x.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    x.add(layers.GlobalAveragePooling2D())
    x.add(layers.Dense(1450, activation='leaky_relu'))
    x.add(layers.Dropout(0.8))
    x.add(layers.Dense(1050, activation='sigmoid'))
    x.add(layers.Dropout(0.5))
    x.add(layers.Dense(316, activation='sigmoid'))
    x.add(layers.Dense(1, activation="sigmoid"))
    return x