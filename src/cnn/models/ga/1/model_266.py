from tensorflow.keras import layers
import tensorflow as tf


def create_model(input_size):
    x = tf.keras.Sequential()
    x.add(layers.Conv2D(18, (7, 7), activation='leaky_relu', padding='same', input_shape=(input_size[0], input_size[1], 1)))
    x.add(layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
    x.add(layers.GlobalAveragePooling2D())
    x.add(layers.Dense(1167, activation='relu'))
    x.add(layers.Dropout(0.8))
    x.add(layers.Dense(346, activation='relu'))
    x.add(layers.Dropout(0.9))
    x.add(layers.Dense(936, activation='sigmoid'))
    x.add(layers.Dense(1, activation="sigmoid"))
    return x