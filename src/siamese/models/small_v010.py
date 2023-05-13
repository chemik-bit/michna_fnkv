from tensorflow.keras import layers
from tensorflow.keras.backend import epsilon
from tensorflow.math import reduce_sum, square, maximum, sqrt
import tensorflow as tf
from tensorflow import keras
# MODEL

def create_model(input_size):
    input = layers.Input((input_size, input_size, 3))
#    x = tf.keras.layers.BatchNormalization()(input)
    x = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(input)
    x = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(x)
    x = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(x)
    x = layers.MaxPooling2D(pool_size=(3, 3))(x)
    x = layers.SpatialDropout2D(0.1)(x)
    x = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(x)
    x = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(x)
    x = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(x)
    x = layers.MaxPooling2D(pool_size=(3, 3))(x)
    x = layers.SpatialDropout2D(0.1)(x)
    x = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(x)
    x = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(x)
    x = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(x)
#    x = tf.keras.layers.BatchNormalization()(input)
    x = layers.MaxPooling2D(pool_size=(3, 3))(x)

#     x = layers.SpatialDropout2D(0.1)(x)
#     x = layers.Conv2D(128, (3, 3), activation="relu", padding="same")(x)
# #    x = tf.keras.layers.BatchNormalization()(input)
#     x = layers.MaxPooling2D(pool_size=(2, 2))(x)
#     x = layers.SpatialDropout2D(0.1)(x)




    x = layers.Flatten()(x)
    x = layers.Dense(4096, activation="relu")(x)

    # x = layers.Dense(4096, activation="relu")(x)
    # x = layers.Dense(4096, activation="relu")(x)
    # x = layers.Dense(4096, activation="relu")(x)
    embedding_network = keras.Model(input, x)

    input_1 = layers.Input((input_size, input_size, 3))
    input_2 = layers.Input((input_size, input_size, 3))

    tower_1 = embedding_network(input_1)
    tower_2 = embedding_network(input_2)

    merge_layer = layers.Lambda(lambda x: sqrt(maximum(reduce_sum(square(x[0]-x[1]), axis=1, keepdims=True), epsilon())))([tower_1, tower_2])

    normal_layer = tf.keras.layers.BatchNormalization()(merge_layer)
    output_layer = layers.Dense(1, activation="sigmoid")(normal_layer)
    siamese = keras.Model(inputs=[input_1, input_2], outputs=output_layer)

    """
    ## Define the constrastive Loss
    """
    def loss(margin=1):
        """Provides 'constrastive_loss' an enclosing scope with variable 'margin'.
        Arguments:
            margin: Integer, defines the baseline for distance for which pairs
                    should be classified as dissimilar. - (default is 1).
        Returns:
            'constrastive_loss' function with data ('margin') attached.
        """
        def contrastive_loss(y_true, y_pred):
            """Calculates the constrastive loss.
            Arguments:
                y_true: List of labels, each label is of type float32.
                y_pred: List of predictions of same length as of y_true,
                        each label is of type float32.
            Returns:
                A tensor containing constrastive loss as floating point value.
            """

            square_pred = tf.math.square(y_pred)
            margin_square = tf.math.square(tf.math.maximum(margin - (y_pred), 0))
            return tf.math.reduce_mean(
                (1 - y_true) * square_pred + (y_true) * margin_square
            )

        return contrastive_loss


    def contrastive_loss(y_true, y_pred):
        """Calculates the constrastive loss.
        Arguments:
            y_true: List of labels, each label is of type float32.
            y_pred: List of predictions of same length as of y_true,
                    each label is of type float32.
        Returns:
            A tensor containing constrastive loss as floating point value.
        """
        margin=1
        square_pred = tf.math.square(y_pred)
        margin_square = tf.math.square(tf.math.maximum(margin - (y_pred), 0))
        return tf.math.reduce_mean(
            (1 - y_true) * square_pred + (y_true) * margin_square
        )


    """
    Compile the model with the contrastive loss
    """
    siamese.compile(loss=contrastive_loss, optimizer="RMSprop", metrics=["accuracy"])
    siamese.summary()
    siamese.save("./siamese_tf")
    siamese = tf.keras.models.load_model("./siamese_tf", custom_objects=({
                "contrastive_loss": contrastive_loss,
                # "euclidean_distance": euclidean_distance,
                # "euclidean_distance_output_shape": euclidean_distance_output_shape
            }))

    checkpoint_filepath = './siamese_tf_checkpoint'
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=False,
        monitor='val_accuracy',
        mode='min',
        save_best_only=True)

    return siamese
