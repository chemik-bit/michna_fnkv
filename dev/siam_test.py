from mpl_toolkits.mplot3d import Axes3D
import os, sys, cv2, matplotlib.pyplot as plt, numpy as np, pickle
import sklearn, pandas as pd, seaborn as sn
from keras.models import Model, load_model, Sequential
from keras import backend as K
from sklearn.metrics import confusion_matrix
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
model = load_model(os.getcwd()+"/color_encoder.h5")
siamese_model = load_model(os.getcwd()+"/color_siamese_model.h5")


# Load test data
f = open(os.getcwd()+"/test_images.pkl", 'rb')
test_healthy, test_nonhealthy = pickle.load(f)
f.close()

names1 = []
names2 = []
names = list(test_healthy) + list(test_nonhealthy)
VOICED_PATH = Path("../data/voiced")
SPECTROGRAMS_PATH = Path("../data/spectrograms")
for idx, hea_file in enumerate(VOICED_PATH.glob("*.hea")):
    if idx > 81:
        with open(hea_file, "rt") as f:
            data = f.readlines()
            image_path = SPECTROGRAMS_PATH.joinpath(data[0][0:8] + ".png")
            if "healthy" in data[-1]:
                print("healthy: ", data )
                names1.append(image_path)
            else:
                print("nonhealthy: ", data)
                names2.append(image_path)



dim = (28, 28)
test_im = []
for i in range(len(names)) :
    im = cv2.imread(str(names[i].resolve()))
    resized = cv2.resize(im, dim, interpolation=cv2.INTER_AREA)
    test_im.append(resized)

r,c,_ = test_im[0].shape
test_im = np.array(test_im)
test_im = test_im.reshape((len(test_im), r,c,3))
names = [str(x.resolve()).split("/")[-1] for x in names]
print("names: ", names)
test_im = 1 - test_im/255
print("preprediction")
print(test_im.shape)

pred = model.predict(test_im)
print("prediction done")
num = int(pred.shape[0]/2)
colors = ['red', 'blue']
y = [colors[0] for i in range(num)]
y += [colors[1] for i in range(num)]

print("prediction shape", pred.shape)
feat1 = pred[:,0]
feat2 = pred[:,1]
feat3 = pred[:,2]
print(pred)

fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(feat1, feat2, feat3, c=y, marker='.')
plt.show()
#
# compute final accuracy on training and test sets
y_pred = model.predict([tr_pairs[:, 0], tr_pairs[:, 1]])
tr_acc = compute_accuracy(tr_y, y_pred)
y_pred = model.predict([te_pairs[:, 0], te_pairs[:, 1]])
te_acc = compute_accuracy(te_y, y_pred)

print('* Accuracy on training set: %0.2f%%' % (100 * tr_acc))
print('* Accuracy on test set: %0.2f%%' % (100 * te_acc))