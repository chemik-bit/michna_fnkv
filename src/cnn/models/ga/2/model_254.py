from tensorflow.keras import layers
import tensorflow as tf


def create_model(input_size):
    x = tf.keras.Sequential()
    x.add(layers.Conv2D(39, (2, 2), activation='tanh', padding='same', input_shape=(input_size[0], input_size[1], 1)))
    x.add(layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
    x.add(layers.BatchNormalization())
    x.add(layers.GlobalAveragePooling2D())
    x.add(layers.Dense(612, activation='leaky_relu'))
    x.add(layers.Dropout(0.6))
    x.add(layers.Dense(752, activation='sigmoid'))
    x.add(layers.Dropout(0.8))
    x.add(layers.Dense(1951, activation='tanh'))
    x.add(layers.Dense(1, activation="sigmoid"))
    return x