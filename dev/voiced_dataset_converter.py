from pathlib import Path
import pickle
from random import choice
import cv2
import numpy as np


def make_pairs(image_paths: Path, image_labels: list, desired_image_size: tuple, path_to_save: Path):
    """
    Load images and make pairs for siamese network [Image1 (uint8), Image2 (uint8)] and corresponding Label
    (1 if both images are in same class, 0 otherwise) and save pickle it.
    :param path_to_save: path to pickled dataset
    :param desired_image_size: desired image size in output list (rescaling image to match siamese input)
    :param image_paths: paths to images with spectrograms
    :param image_labels: list with images labels
    :return: None
    """
    images_list_healthy = []
    images_list_nonhealthy = []
    new_labels = []
    image_paths_healthy = []
    image_paths_nonhealthy = []
    # load images and resize them
    for idx, image_path in enumerate(image_paths):
        # read image

        image = cv2.imread(str(image_path.resolve()))
        # resize image according to desired_image_size
        image = cv2.resize(image, desired_image_size, interpolation=cv2.INTER_AREA)
        if image_labels[idx] == 1:
            images_list_healthy.append(image)
            new_labels.append(1)
            image_paths_healthy.append(str(image_path.resolve()))
        else:
            images_list_nonhealthy.append(image)
            new_labels.append(0)
            image_paths_nonhealthy.append(str(image_path.resolve()))

    # make pairs
    image_pairs = []
    label_pairs = []

    images_pairs_paths = image_paths_healthy + image_paths_nonhealthy
    images = np.asarray(images_list_healthy + images_list_nonhealthy)
    new_labels = np.asarray(new_labels)
    image_pairs_paths_toprint = []
    for idx, image in enumerate(images):
        # make matching pair
        for i in range(10):
            selection_idx = np.random.choice(np.where(new_labels == new_labels[idx])[0])
            image_pairs.append([image, images[selection_idx]])
            image_pairs_paths_toprint.append([images_pairs_paths[idx], images_pairs_paths[selection_idx]])
            label_pairs.append(1)
            # make nonmatching pair
            selection_idx = np.random.choice(np.where(new_labels != new_labels[idx])[0])
            image_pairs.append([image, images[selection_idx]])
            label_pairs.append(0)


    with open(path_to_save, "wb") as f:
        pickle.dump(image_pairs, f)
        pickle.dump(label_pairs, f)

    print(image_pairs_paths_toprint)
    print(label_pairs)

pickled_sets = {"train": (Path("../data/voiced_train.pickled"), Path("../data/voiced_pairs_train.pickled")),
                "validation": (Path("../data/voiced_validation.pickled"),
                               Path("../data/voiced_pairs_validation.pickled")),
                "test": (Path("../data/voiced_test.pickled"), Path("../data/voiced_test_train.pickled"))
                }

for item in pickled_sets.values():
    with open(item[0], "rb") as f:
        image_paths = pickle.load(f)
        image_labels = pickle.load(f)
    make_pairs(image_paths, image_labels, (32, 32), item[1])
    with open(item[1], "rb") as f:
        pairs = pickle.load(f)
        labels = pickle.load(f)

    print(f"{item[1]} - {len(labels)}")


