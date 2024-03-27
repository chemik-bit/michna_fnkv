from tensorflow.keras import layers
import tensorflow as tf


def create_model(input_size):
    x = tf.keras.Sequential()
    x.add(layers.Conv2D(31, (3, 3), activation='leaky_relu', padding='same', input_shape=(input_size[0], input_size[1], 1)))
    x.add(layers.BatchNormalization())
    x.add(layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
    x.add(layers.BatchNormalization())
    x.add(layers.GlobalAveragePooling2D())
    x.add(layers.Dense(756, activation='tanh'))
    x.add(layers.Dropout(0.3))
    x.add(layers.Dense(1106, activation='relu'))
    x.add(layers.Dropout(0.5))
    x.add(layers.Dense(864, activation='tanh'))
    x.add(layers.Dense(1, activation="sigmoid"))
    return x