import itertools
from pathlib import Path
import os
import cv2
import numpy as np
from model import train_color_encoder
import pickle
import os, sys, cv2, matplotlib.pyplot as plt, numpy as np, shutil
from random import random, randint, seed
import random
import pickle, itertools, sklearn, pandas as pd, seaborn as sn
from scipy.spatial import distance
from keras.models import Model, load_model, Sequential
from keras import backend as K
from keras.utils.vis_utils import plot_model
from scipy import spatial
from sklearn.metrics import confusion_matrix

VOICED_PATH = Path("../data/voiced")
SPECTROGRAMS_PATH = Path("../data/spectrograms")
healthy_subjects = []
nonhealthy_subjects = []

for hea_file in VOICED_PATH.glob("*.hea"):
    with open(hea_file, "rt") as f:
        data = f.readlines()
        image_path = SPECTROGRAMS_PATH.joinpath(data[0][0:8] + ".png")
        if "healthy" in data[-1]:
            print("healthy: ", data )
            healthy_subjects.append(image_path)
        else:
            print("nonhealthy: ", data)
            nonhealthy_subjects.append(image_path)

print("HEALTHY SUBJECTS")
print(healthy_subjects)
print("NON-HEALTHY SUBJECTS")
print(nonhealthy_subjects)

training_test_size = 79
# prepare positive pairs (healhty, non-healthy)
positive_healthy = list(itertools.combinations(healthy_subjects[:training_test_size], 2))
positive_nonhealthy = list(itertools.combinations(nonhealthy_subjects[:training_test_size], 2))

# prepare negative pairs
negative_samples = list(itertools.product(healthy_subjects[:training_test_size], nonhealthy_subjects[:training_test_size]))

print("negative: ", negative_samples)
# read images and append them to list
spectrograms1 = []
spectrograms2 = []
# 1 = both images are same type, 0 = healhty/non-healthy pair
target_value = []

positive_samples = positive_healthy + positive_nonhealthy
dim = (28, 28)

# resize image


for file_name in positive_samples:
    im = cv2.imread(str(file_name[0].resolve()))
    resized = cv2.resize(im, dim, interpolation=cv2.INTER_AREA)
    spectrograms1.append(resized)
    im = cv2.imread(str(file_name[1].resolve()))
    resized = cv2.resize(im, dim, interpolation=cv2.INTER_AREA)
    spectrograms2.append(resized)
    target_value.append(1)

for file_name in negative_samples:
    im = cv2.imread(str(file_name[0].resolve()))
    resized = cv2.resize(im, dim, interpolation=cv2.INTER_AREA)
    spectrograms1.append(resized)
    im = cv2.imread(str(file_name[1].resolve()))
    resized = cv2.resize(im, dim, interpolation=cv2.INTER_AREA)
    spectrograms2.append(resized)
    target_value.append(0)

target_value = np.array(target_value)
spectrograms1 = np.array(spectrograms1)
spectrograms2 = np.array(spectrograms2)
spectrograms1 = spectrograms1.reshape((len(negative_samples) + len(positive_samples), 28, 28, 3))
spectrograms2 = spectrograms2.reshape((len(negative_samples) + len(positive_samples), 28, 28, 3))

spectrograms1 = 1 - spectrograms1/255
spectrograms2 = 1 - spectrograms2/255

print("Spectrograms data : ", spectrograms1.shape, spectrograms2.shape, target_value.shape)

# Save test data
test_healthy = healthy_subjects[training_test_size:]
test_nonhealthy = nonhealthy_subjects[training_test_size:]
f = open(os.getcwd()+"/test_images.pkl", 'wb')
pickle.dump([test_healthy, test_nonhealthy], f)
f.close()

# train model
train_color_encoder(spectrograms1, spectrograms2, target_value)


# compute final accuracy on training and test sets
model = load_model(os.getcwd()+"/color_encoder.h5")
siamese_model = load_model(os.getcwd()+"/color_siamese_model.h5")
def compute_accuracy(y_true, y_pred):
    '''Compute classification accuracy with a fixed threshold on distances.
    '''
    pred = y_pred.ravel() < 0.5
    return np.mean(pred == y_true)

im = cv2.imread(str(negative_samples[0][0].resolve()))
resized = cv2.resize(im, dim, interpolation=cv2.INTER_AREA)

print("test shape", spectrograms1.shape)
y_pred = siamese_model.predict([spectrograms1, spectrograms2])
print(y_pred)
tr_acc = compute_accuracy(target_value, y_pred)
# y_pred = model.predict([te_pairs[:, 0], te_pairs[:, 1]])
# te_acc = compute_accuracy(te_y, y_pred)

print('* Accuracy on training set: %0.2f%%' % (100 * tr_acc))
# print('* Accuracy on test set: %0.2f%%' % (100 * te_acc))