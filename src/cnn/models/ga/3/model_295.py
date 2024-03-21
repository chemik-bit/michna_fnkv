from tensorflow.keras import layers
import tensorflow as tf


def create_model(input_size):
    x = tf.keras.Sequential()
    x.add(layers.Conv2D(50, (3, 3), activation='tanh', padding='same', input_shape=(input_size[0], input_size[1], 1)))
    x.add(layers.BatchNormalization())
    x.add(layers.Conv2D(23, (6, 6), activation='tanh', padding='same'))
    x.add(layers.MaxPooling2D(pool_size=(1, 1), strides=(2, 2)))
    x.add(layers.BatchNormalization())
    x.add(layers.Conv2D(16, (5, 5), activation='leaky_relu', padding='same'))
    x.add(layers.BatchNormalization())
    x.add(layers.BatchNormalization())
    x.add(layers.Conv2D(5, (2, 2), activation='tanh', padding='same'))
    x.add(layers.BatchNormalization())
    x.add(layers.MaxPooling2D(pool_size=(1, 1), strides=(2, 2)))
    x.add(layers.GlobalAveragePooling2D())
    x.add(layers.Dense(1433, activation='leaky_relu'))
    x.add(layers.Dense(381, activation='relu'))
    x.add(layers.Dropout(0.7))
    x.add(layers.Dense(1114, activation='sigmoid'))
    x.add(layers.Dense(1, activation="sigmoid"))
    return x