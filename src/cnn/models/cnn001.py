from tensorflow.keras import layers
from tensorflow.keras.backend import epsilon
from tensorflow.math import reduce_sum, square, maximum, sqrt
import tensorflow as tf
from tensorflow import keras
# MODEL

def create_model(input_size):
    x = tf.keras.Sequential()
    x.add(layers.Input((input_size, input_size, 3)))
#    x = tf.keras.layers.BatchNormalization()(input)
    x.add(layers.Conv2D(64, (3, 3), activation="relu", padding="same"))
    x.add(layers.MaxPooling2D(pool_size=(2, 2)))
    x.add(layers.SpatialDropout2D(0.1))
    x.add(layers.Conv2D(64, (3, 3), activation="relu", padding="same"))
    x.add(layers.MaxPooling2D(pool_size=(2, 2)))
    x.add(layers.SpatialDropout2D(0.1))
    x.add(layers.Conv2D(64, (3, 3), activation="relu", padding="same"))
#    x = tf.keras.layers.BatchNormalization()(input)
    x.add(layers.MaxPooling2D(pool_size=(2, 2)))
    x.add(layers.SpatialDropout2D(0.1))
    x.add(layers.Conv2D(128, (3, 3), activation="relu", padding="same"))
#    x = tf.keras.layers.BatchNormalization()(input)
    x.add(layers.MaxPooling2D(pool_size=(2, 2)))
    x.add(layers.SpatialDropout2D(0.1))
    x.add(layers.Conv2D(128, (3, 3), activation="relu", padding="same"))

    x.add(layers.MaxPooling2D(pool_size=(2, 2)))
    x.add(layers.SpatialDropout2D(0.1))
    x.add(layers.Conv2D(256, (3, 3), activation="relu", padding="same"))
    x.add(layers.MaxPooling2D(pool_size=(2, 2)))
    x.add(layers.SpatialDropout2D(0.1))

    x.add(layers.Conv2D(256, (3, 3), activation="relu", padding="same"))
    x.add(layers.MaxPooling2D(pool_size=(2, 2)))
    x.add(layers.SpatialDropout2D(0.1))


    x.add(layers.Flatten())
    x.add(layers.Dense(4096, activation="relu"))
    x.add(layers.Dense(4096, activation="relu"))
    x.add(layers.Dense(1, activation="sigmoid"))

    """
    Compile the model with the contrastive loss
    """
    x.compile(loss=tf.losses.binary_crossentropy, optimizer="adam", metrics=["accuracy"])
    x.summary()
    x.save("./siamese_tf")
    x = tf.keras.models.load_model("./siamese_tf")

    checkpoint_filepath = './siamese_tf_checkpoint'
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=False,
        monitor='val_accuracy',
        mode='min',
        save_best_only=True)

    return x

