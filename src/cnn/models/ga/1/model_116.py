from tensorflow.keras import layers
import tensorflow as tf


def create_model(input_size):
    x = tf.keras.Sequential()
    x.add(layers.Conv2D(58, (4, 4), activation='leaky_relu', padding='same', input_shape=(input_size[0], input_size[1], 1)))
    x.add(layers.BatchNormalization())
    x.add(layers.Conv2D(46, (6, 6), activation='sigmoid', padding='same'))
    x.add(layers.BatchNormalization())
    x.add(layers.GlobalAveragePooling2D())
    x.add(layers.Dense(613, activation='leaky_relu'))
    x.add(layers.Dropout(0.4))
    x.add(layers.Dense(1, activation="sigmoid"))
    return x