from tensorflow.keras import layers
import tensorflow as tf


def create_model(input_size):
    x = tf.keras.Sequential()
    x.add(layers.Conv2D(40, (1, 1), activation='tanh', padding='same', input_shape=(input_size[0], input_size[1], 1)))
    x.add(layers.BatchNormalization())
    x.add(layers.Conv2D(29, (5, 5), activation='sigmoid', padding='same'))
    x.add(layers.BatchNormalization())
    x.add(layers.Conv2D(4, (7, 7), activation='tanh', padding='same'))
    x.add(layers.BatchNormalization())
    x.add(layers.Conv2D(4, (6, 6), activation='leaky_relu', padding='same'))
    x.add(layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
    x.add(layers.GlobalAveragePooling2D())
    x.add(layers.Dense(1336, activation='tanh'))
    x.add(layers.Dropout(0.5))
    x.add(layers.Dense(483, activation='relu'))
    x.add(layers.Dropout(0.6))
    x.add(layers.Dense(105, activation='sigmoid'))
    x.add(layers.Dense(1, activation="sigmoid"))
    return x