from tensorflow.keras import layers
import tensorflow as tf


def create_model(input_size):
    x = tf.keras.Sequential()
    x.add(layers.Conv2D(14, (6, 6), activation='tanh', padding='same', input_shape=(input_size[0], input_size[1], 1)))
    x.add(layers.Conv2D(61, (4, 4), activation='sigmoid', padding='same'))
    x.add(layers.MaxPooling2D(pool_size=(1, 1), strides=(2, 2)))
    x.add(layers.BatchNormalization())
    x.add(layers.Conv2D(19, (1, 1), activation='sigmoid', padding='same'))
    x.add(layers.BatchNormalization())
    x.add(layers.Conv2D(35, (2, 2), activation='sigmoid', padding='same'))
    x.add(layers.Conv2D(18, (4, 4), activation='leaky_relu', padding='same'))
    x.add(layers.GlobalAveragePooling2D())
    x.add(layers.Dense(28, activation='sigmoid'))
    x.add(layers.Dropout(0.3))
    x.add(layers.Dense(1616, activation='relu'))
    x.add(layers.Dropout(0.6))
    x.add(layers.Dense(1, activation="sigmoid"))
    return x