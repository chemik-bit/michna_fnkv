from mpl_toolkits.mplot3d import Axes3D
import os, sys, cv2, matplotlib.pyplot as plt, numpy as np, pickle
import sklearn, pandas as pd, seaborn as sn
from keras.models import Model, load_model, Sequential
from keras import backend as K
from sklearn.metrics import confusion_matrix

import warnings
warnings.filterwarnings('ignore')

# Load models
model = load_model(os.getcwd()+"/color_encoder.h5")
siamese_model = load_model(os.getcwd()+"/color_siamese_model.h5")


# Load test data
f = open(os.getcwd()+"/test_images.pkl", 'rb')
test_healthy, test_nonhealthy, test_labels = pickle.load(f)
f.close()


# Read files
names = list(test_healthy) + list(test_nonhealthy)# + list(test_cyan_im) #+ list(test_yellow_im)
names1 = ["healthy" for x in list(test_healthy)]
#names2 = [x for x in names if 'nonhealthy' in x]
names2 = ["nonhealthy" for x in list(test_nonhealthy)]
dim = (28, 28)
test_im = []
for i in range(len(names)) :
    im = cv2.imread(str(names[i].resolve()))
    im = cv2.resize(im, dim, interpolation=cv2.INTER_AREA)
    test_im.append(im)

r,c,_ = test_im[0].shape
test_im = np.array(test_im)
test_im = test_im.reshape((len(test_im), r,c,3))
#names = [x.split("/")[-1] for x in names]

test_im = 1 - test_im/255
print(test_im.shape)
# Predict
pred = model.predict(test_im)

num = int(pred.shape[0]/2)
status = ['healthy', 'nonhealthy'] # set colors of target labels

print("non-health", len(test_nonhealthy), len(test_healthy))
# Set target labels
y = [status[0] for i in range(len(list(test_healthy)))]
y += [status[1] for i in range(len(list(test_nonhealthy)))]


feat1 = pred[:,0]
feat2 = pred[:,1]
feat3 = pred[:,2]

# Plot 3d scatter plot
fig = plt.figure()
ax = Axes3D(fig)
colors = []
print(y)
print(pred.shape)
for item in y:

    if "non" in item:
        colors.append("blue")
    else:
        colors.append("red")

ax.scatter(feat1, feat2, feat3, c=colors, marker='.')
plt.show()

result = siamese_model.predict([test_im, test_im])

y_true = [1 for x in range(len(names))]
rounded = [np.round(x) for x in result]
print(confusion_matrix(y_true, rounded))
