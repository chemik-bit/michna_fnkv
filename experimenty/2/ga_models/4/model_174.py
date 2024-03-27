from tensorflow.keras import layers
import tensorflow as tf


def create_model(input_size):
    x = tf.keras.Sequential()
    x.add(layers.Conv2D(36, (4, 4), activation='tanh', padding='same', input_shape=(input_size[0], input_size[1], 1)))
    x.add(layers.BatchNormalization())
    x.add(layers.GlobalAveragePooling2D())
    x.add(layers.Dense(1226, activation='relu'))
    x.add(layers.Dropout(0.4))
    x.add(layers.Dense(340, activation='sigmoid'))
    x.add(layers.Dense(1997, activation='leaky_relu'))
    x.add(layers.Dense(1, activation="sigmoid"))
    return x