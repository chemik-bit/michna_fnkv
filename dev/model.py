import keras
from keras.layers import Input, Dense, InputLayer, Conv2D, MaxPooling2D, UpSampling2D, InputLayer, Concatenate, Flatten, \
    Reshape, Lambda, Embedding, dot, AveragePooling2D, Dropout
from keras.models import Model, load_model, Sequential
import matplotlib.pyplot as plt
import keras.backend as K
from sklearn.model_selection import train_test_split
import os, sys
import tensorflow as tf
from keras.utils.vis_utils import plot_model



# Train autoencoder and save encoder model and encodings
def train_color_encoder(X1, X2, y):
    # Color Encoder
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    tf.autograph.set_verbosity(0)
    # input = Input((28, 28, 3))
    # x = Conv2D(32, (3, 3), activation='relu')(input)
    # x = AveragePooling2D(pool_size=(2, 2))(x)
    # x = Conv2D(64, (3, 3), activation='tanh')(x)
    # x = MaxPooling2D(pool_size=(2, 2))(x)
    # x = Dropout(0.25)(x)
    # x = Flatten()(x)
    # x = Dense(128, activation='tanh')(x)
    # x = Dropout(0.5)(x)
    # x = Dense(64, activation='tanh')(x)
    # x = Dropout(0.5)(x)
    # x = Dense(2, activation='tanh')(x)
    #model = Model(input, x)
    input_layer = Input((28, 28, 3))
    layer1 = Conv2D(16, (3, 3), activation='relu', padding='same')(input_layer)
    layer2 = MaxPooling2D((2, 2), padding='same')(layer1)
    layer3 = Conv2D(8, (3, 3), activation='relu', padding='same')(layer2)
    layer4 = MaxPooling2D((2, 2), padding='same')(layer3)
    layer5 = Flatten()(layer4)
    layer6 = Dense(16, activation="tanh")(layer5)
    layer7 = Dense(8, activation="tanh")(layer6)
    layer8 = Dense(2, activation="tanh")(layer7)
    # embeddings = Dense(16, activation=None)(layer5)
    # norm_embeddings = tf.nn.l2_normalize(embeddings, axis=-1)
    #
    # # Create model
    model = Model(input_layer, layer8)

    # Create siamese model
    input1 = Input((28, 28, 3))
    input2 = Input((28, 28, 3))

    # Create left and right twin models
    left_model = model(input1)
    right_model = model(input2)

    # Dot product layer
    dot_product = dot([left_model, right_model], axes=1, normalize=False)

    siamese_model = Model([input1, input2], model)

    # Model summary
    print(siamese_model.summary())

    # Compile model
    siamese_model.compile(optimizer='adam', loss='mse')

    # Plot flowchart fo model
    plot_model(siamese_model, to_file=os.getcwd() + '/siamese_model_mnist.png', show_shapes=1, show_layer_names=1)

    # Fit model
    try:
        print(X1.shape, X2.shape, y.shape)
        siamese_model.fit([X1, X2], y, epochs=100, batch_size=5, shuffle=True, verbose=True)
    except Exception as ex:
        print(ex)

    print("going to save")

    model.save(os.getcwd() + "/color_encoder.h5")
    siamese_model.save(os.getcwd() + "/color_siamese_model.h5")

    return model, siamese_model