import os, sys, cv2, matplotlib.pyplot as plt, numpy as np, shutil
from pathlib import Path
from random import random, randint, seed
import random
import pickle, itertools, sklearn, pandas as pd, seaborn as sn
from scipy.spatial import distance
from keras.models import Model, load_model, Sequential
from keras import backend as K
from keras.utils.vis_utils import plot_model
from scipy import spatial
from sklearn.metrics import confusion_matrix


from model import train_color_encoder


with open(Path("../data/voiced.pickled"), "rb") as f:
    images = np.asarray(pickle.load(f))
    targets = np.asarray(pickle.load(f))

training_set_size = 50

# Generate positive samples
healthy_im = images[np.where(np.asarray(targets) == 1)]
nonhealthy_im = images[np.where(np.asarray(targets) == 0)]

# Test images
test_healthy_im = healthy_im[training_set_size:]
test_nonhealthy_im = nonhealthy_im[training_set_size:]
test_labels = targets[training_set_size:]

# Training sets
healthy_im = healthy_im[:training_set_size]
nonhealthy_im = nonhealthy_im[:training_set_size]


positive_healthy = list(itertools.combinations(healthy_im, 2))
positive_nonhealthy = list(itertools.combinations(nonhealthy_im, 2))

# Generate negative samples
negative = list(itertools.product(healthy_im, nonhealthy_im))

# Create pairs of images and set target label for them.
# Target output is 1 if pair of images have diagnosis else it is 0.

image_X1 = []
image_X2 = []
target_y = []
positive_samples = positive_healthy + positive_nonhealthy
negative_samples = negative
dim = (28, 28)

for fname in positive_samples:
    im = cv2.imread(str(fname[0]))
    im = cv2.resize(im, dim, interpolation=cv2.INTER_AREA)
    image_X1.append(im)
    im = cv2.imread(str(fname[1].resolve()))
    im = cv2.resize(im, dim, interpolation=cv2.INTER_AREA)
    image_X2.append(im)
    target_y.append(1)

for fname in negative_samples:
    im = cv2.imread(str(fname[0]))
    im = cv2.resize(im, dim, interpolation=cv2.INTER_AREA)
    image_X1.append(im)
    im = cv2.imread(str(fname[1]))
    im = cv2.resize(im, dim, interpolation=cv2.INTER_AREA)
    image_X2.append(im)
    target_y.append(0)

target_y = np.array(target_y)
image_X1 = np.array(image_X1)
image_X2 = np.array(image_X2)
image_X1 = image_X1.reshape((len(negative_samples) + len(positive_samples), 28, 28, 3))
image_X2 = image_X2.reshape((len(negative_samples) + len(positive_samples), 28, 28, 3))

image_X1 = 1 - image_X1/255
image_X2 = 1 - image_X2/255

f = open(os.getcwd()+"/test_images.pkl", 'wb')
pickle.dump([test_healthy_im, test_nonhealthy_im, test_labels], f)
f.close()

# check dimensions
print("Dimensions> ", image_X1.shape, image_X2.shape, target_y.shape)

# train model
train_color_encoder(image_X1, image_X2, target_y)
