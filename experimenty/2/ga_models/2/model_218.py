from tensorflow.keras import layers
import tensorflow as tf


def create_model(input_size):
    x = tf.keras.Sequential()
    x.add(layers.Conv2D(58, (1, 1), activation='sigmoid', padding='same', input_shape=(input_size[0], input_size[1], 1)))
    x.add(layers.BatchNormalization())
    x.add(layers.Conv2D(29, (3, 3), activation='sigmoid', padding='same'))
    x.add(layers.BatchNormalization())
    x.add(layers.MaxPooling2D(pool_size=(1, 1), strides=(2, 2)))
    x.add(layers.GlobalAveragePooling2D())
    x.add(layers.Dense(1277, activation='relu'))
    x.add(layers.Dense(393, activation='sigmoid'))
    x.add(layers.Dropout(0.7))
    x.add(layers.Dense(1346, activation='sigmoid'))
    x.add(layers.Dropout(0.9))
    x.add(layers.Dense(1, activation="sigmoid"))
    return x