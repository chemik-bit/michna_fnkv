from tensorflow.keras import layers
import tensorflow as tf


def create_model(input_size):
    x = tf.keras.Sequential()
    x.add(layers.Conv2D(63, (3, 3), activation='leaky_relu', padding='same', input_shape=(input_size[0], input_size[1], 1)))
    x.add(layers.BatchNormalization())
    x.add(layers.Conv2D(35, (1, 1), activation='leaky_relu', padding='same'))
    x.add(layers.BatchNormalization())
    x.add(layers.Conv2D(38, (3, 3), activation='relu', padding='same'))
    x.add(layers.Conv2D(39, (2, 2), activation='relu', padding='same'))
    x.add(layers.MaxPooling2D(pool_size=(1, 1), strides=(2, 2)))
    x.add(layers.BatchNormalization())
    x.add(layers.Conv2D(58, (2, 2), activation='sigmoid', padding='same'))
    x.add(layers.Conv2D(30, (2, 2), activation='relu', padding='same'))
    x.add(layers.BatchNormalization())
    x.add(layers.GlobalAveragePooling2D())
    x.add(layers.Dense(92, activation='leaky_relu'))
    x.add(layers.Dropout(0.6))
    x.add(layers.Dense(1156, activation='relu'))
    x.add(layers.Dropout(0.4))
    x.add(layers.Dense(1815, activation='sigmoid'))
    x.add(layers.Dense(1, activation="sigmoid"))
    return x