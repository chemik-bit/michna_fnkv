from tensorflow.keras import layers
import tensorflow as tf


def create_model(input_size):
    x = tf.keras.Sequential()
    x.add(layers.Conv2D(10, (7, 7), activation='sigmoid', padding='same', input_shape=(input_size[0], input_size[1], 1)))
    x.add(layers.BatchNormalization())
    x.add(layers.Conv2D(9, (2, 2), activation='leaky_relu', padding='same'))
    x.add(layers.MaxPooling2D(pool_size=(1, 1), strides=(2, 2)))
    x.add(layers.GlobalAveragePooling2D())
    x.add(layers.Dense(716, activation='leaky_relu'))
    x.add(layers.Dense(1820, activation='sigmoid'))
    x.add(layers.Dropout(0.7))
    x.add(layers.Dense(1041, activation='tanh'))
    x.add(layers.Dense(1, activation="sigmoid"))
    return x