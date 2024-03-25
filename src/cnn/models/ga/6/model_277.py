from tensorflow.keras import layers
import tensorflow as tf


def create_model(input_size):
    x = tf.keras.Sequential()
    x.add(layers.Conv2D(57, (1, 1), activation='sigmoid', padding='same', input_shape=(input_size[0], input_size[1], 1)))
    x.add(layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
    x.add(layers.BatchNormalization())
    x.add(layers.GlobalAveragePooling2D())
    x.add(layers.Dense(1735, activation='relu'))
    x.add(layers.Dense(411, activation='sigmoid'))
    x.add(layers.Dropout(0.2))
    x.add(layers.Dense(1527, activation='sigmoid'))
    x.add(layers.Dense(1, activation="sigmoid"))
    return x