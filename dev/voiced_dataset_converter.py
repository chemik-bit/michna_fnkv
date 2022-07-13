from pathlib import Path
import pickle
from random import choice
import cv2
import numpy as np


def make_pairs(image_paths: Path, image_labels: list, desired_image_size: tuple):
    """
    Load images and make pairs for siamese network [Image1 (uint8), Image2 (uint8)] and corresponding Label
    (1 if both images are in same class, 0 otherwise).
    :param desired_image_size: desired image size in output list (rescaling image to match siamese input)
    :param image_paths:
    :param image_labels:
    :return: list of lists with pairs [[ImageX, ImageY], [ImageX, ImageZ], ...] and
     list of labels [label_pair1, label_pair2, ....]
    """
    images_list_healthy = []
    images_list_nonhealthy = []
    new_labels = []
    # load images and resize them
    for idx, image_path in enumerate(image_paths):
        # read image
        image = cv2.imread(str(image_path.resolve()))
        # resize image according to desired_image_size
        image = cv2.resize(image, desired_image_size, interpolation=cv2.INTER_AREA)
        if image_labels[idx] == 1:
            images_list_healthy.append(image)
            new_labels.append(1)
        else:
            images_list_nonhealthy.append(image)
            new_labels.append(0)

    # make pairs
    image_pairs = []
    label_pairs = []
    images = np.asarray(images_list_healthy + images_list_nonhealthy)
    new_labels = np.asarray(new_labels)
    for idx, image in enumerate(images):
        # make matching pair
        for i in range(10):
            selection_idx = np.random.choice(np.where(new_labels == new_labels[idx])[0])
            image_pairs.append([image, images[selection_idx]])
            label_pairs.append(1)
            # make nonmatching pair
            selection_idx = np.random.choice(np.where(new_labels != new_labels[idx])[0])
            image_pairs.append([image, images[selection_idx]])
            label_pairs.append(0)

    path_to_save = Path("../data/voiced_pairs.pickled")
    with open(path_to_save, "wb") as f:
        pickle.dump(image_pairs, f)
        pickle.dump(label_pairs, f)

pickled_voiced_path = Path("../data/voiced.pickled")
with open(pickled_voiced_path, "rb") as f:
    image_paths = pickle.load(f)
    image_labels = pickle.load(f)


make_pairs(image_paths, image_labels, (28, 28))

pickled_pairs_path = Path("../data/voiced_pairs.pickled")
with open(pickled_pairs_path, "rb") as f:
    pairs = pickle.load(f)
    labels = pickle.load(f)

# for pair_image in pairs:
#     cv2.imshow("1", np.asarray(pair_image[0], dtype=np.uint8))
#     cv2.imshow("2", np.asarray(pair_image[1], dtype=np.uint8))
#     cv2.waitKey(1000)
print(len(labels))
