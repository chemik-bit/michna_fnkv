from pathlib import Path
import pickle
import random
import cv2
import itertools
import numpy as np


def make_pairs(image_paths: Path, image_labels: list, desired_image_size: tuple,
               path_to_save: Path, instance_pairs: int):
    """
    Load images and make pairs for siamese network [Image1 (uint8), Image2 (uint8)] and corresponding Label
    (1 if both images are in same class, 0 otherwise) and save pickle it.
    :param path_to_save: path to pickled dataset
    :param desired_image_size: desired image size in output list (rescaling image to match siamese input)
    :param image_paths: paths to images with spectrograms
    :param image_labels: list with images labels
    :return: None
    """

    image_paths_healthy = []
    image_paths_nonhealthy = []
    # load images and resize them
    for idx, image_path in enumerate(image_paths):
        if image_labels[idx] == 1:
            image_paths_healthy.append(str(image_path.resolve()))
        else:
            image_paths_nonhealthy.append(str(image_path.resolve()))

    image_pairs = []
    pairs_labels = []
    random_paths = []
    healthy_images = []
    nonhealthy_images = []
    final_labels = []
    for healthy_image_path in image_paths_healthy:
        # read image
        image = cv2.imread(healthy_image_path)
        # resize image according to desired_image_size
        image = cv2.resize(image, desired_image_size, interpolation=cv2.INTER_AREA)
        healthy_images.append(image)
    for nonhealthy_image_path in image_paths_nonhealthy:
        # read image
        image = cv2.imread(nonhealthy_image_path)
        # resize image according to desired_image_size
        image = cv2.resize(image, desired_image_size, interpolation=cv2.INTER_AREA)
        nonhealthy_images.append(image)

    image_labels = []
    pairs = list(itertools.product(healthy_images, healthy_images))
    random.shuffle(pairs)
    image_pairs += pairs
    image_labels += len(pairs) * [1]
    print(f"{len(image_pairs)} - {len(image_labels)}")

    pairs = list(itertools.product(nonhealthy_images, nonhealthy_images))
    random.shuffle(pairs)
    image_pairs += pairs
    image_labels += len(pairs) * [1]

    pairs = list(itertools.product(healthy_images, nonhealthy_images))
    random.shuffle(pairs)
    image_pairs += pairs
    image_labels += len(pairs) * [0]
    print(len(image_pairs))
    tf_dict = {"data": [], "labels": []}
    for idx, image_pair in enumerate(image_pairs):
        tf_dict["data"].append(image_pair)
        tf_dict["labels"].append(image_labels[idx])
        if idx > 0 and idx % 1000 == 0:
            path_to_save_pickle = path_to_save.joinpath(f"voiced_pairs_{int(idx / 1000):05d}.pickled")
            with open(path_to_save_pickle, "wb") as f:
                # pickle.dump(image_pairs, f)
                # pickle.dump(image_labels, f)
                pickle.dump(tf_dict, f)
            print(f"Saving to {path_to_save_pickle}")
            tf_dict = {"data": [], "labels": []}

    with open(path_to_save.joinpath("voiced_pairs.pickled"), "wb") as f:
        #pickle.dump(image_pairs, f)
        #pickle.dump(image_labels, f)
        pickle.dump(tf_dict, f)

pickled_sets = {"train": (Path("../data/voiced_train.pickled"), Path("../data/splited_voiced/train"), 20),
                "validation": (Path("../data/voiced_validation.pickled"),
                               Path("../data/splited_voiced/val"), 4),
                "test": (Path("../data/voiced_test.pickled"), Path("../data/splited_voiced/test"), 2)
                }

for item in pickled_sets.values():
    with open(item[0], "rb") as f:
        image_paths = pickle.load(f)
        image_labels = pickle.load(f)

    make_pairs(image_paths, image_labels, (224, 224), item[1], item[2])
    # with open(item[1], "rb") as f:
    #     # pairs = pickle.load(f)
    #     # labels = pickle.load(f)
    #     tf_test = pickle.load(f)
    # print(f"{item[1]} - {len(tf_test)}")


