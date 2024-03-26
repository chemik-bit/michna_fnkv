from tensorflow.keras import layers
import tensorflow as tf


def create_model(input_size):
    x = tf.keras.Sequential()
    x.add(layers.Conv2D(33, (7, 7), activation='relu', padding='same', input_shape=(input_size[0], input_size[1], 1)))
    x.add(layers.BatchNormalization())
    x.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    x.add(layers.BatchNormalization())
    x.add(layers.Conv2D(41, (7, 7), activation='leaky_relu', padding='same'))
    x.add(layers.Conv2D(39, (6, 6), activation='tanh', padding='same'))
    x.add(layers.Conv2D(39, (3, 3), activation='sigmoid', padding='same'))
    x.add(layers.BatchNormalization())
    x.add(layers.GlobalAveragePooling2D())
    x.add(layers.Dense(2028, activation='leaky_relu'))
    x.add(layers.Dropout(0.4))
    x.add(layers.Dense(630, activation='relu'))
    x.add(layers.Dense(1, activation="sigmoid"))
    return x