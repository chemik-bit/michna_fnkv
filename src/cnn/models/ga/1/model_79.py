from tensorflow.keras import layers
import tensorflow as tf


def create_model(input_size):
    x = tf.keras.Sequential()
    x.add(layers.Conv2D(25, (1, 1), activation='relu', padding='same', input_shape=(input_size[0], input_size[1], 1)))
    x.add(layers.BatchNormalization())
    x.add(layers.Conv2D(26, (1, 1), activation='tanh', padding='same'))
    x.add(layers.BatchNormalization())
    x.add(layers.Conv2D(17, (1, 1), activation='tanh', padding='same'))
    x.add(layers.BatchNormalization())
    x.add(layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
    x.add(layers.BatchNormalization())
    x.add(layers.Conv2D(3, (3, 3), activation='tanh', padding='same'))
    x.add(layers.BatchNormalization())
    x.add(layers.Conv2D(4, (2, 2), activation='sigmoid', padding='same'))
    x.add(layers.BatchNormalization())
    x.add(layers.BatchNormalization())
    x.add(layers.GlobalAveragePooling2D())
    x.add(layers.Dense(1640, activation='leaky_relu'))
    x.add(layers.Dropout(0.7))
    x.add(layers.Dense(1, activation="sigmoid"))
    return x