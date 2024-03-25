from tensorflow.keras import layers
import tensorflow as tf


def create_model(input_size):
    x = tf.keras.Sequential()
    x.add(layers.Conv2D(8, (6, 6), activation='relu', padding='same', input_shape=(input_size[0], input_size[1], 1)))
    x.add(layers.BatchNormalization())
    x.add(layers.BatchNormalization())
    x.add(layers.Conv2D(48, (2, 2), activation='relu', padding='same'))
    x.add(layers.Conv2D(51, (2, 2), activation='relu', padding='same'))
    x.add(layers.BatchNormalization())
    x.add(layers.GlobalAveragePooling2D())
    x.add(layers.Dense(373, activation='sigmoid'))
    x.add(layers.Dropout(0.5))
    x.add(layers.Dense(554, activation='sigmoid'))
    x.add(layers.Dropout(0.5))
    x.add(layers.Dense(148, activation='leaky_relu'))
    x.add(layers.Dense(1, activation="sigmoid"))
    return x