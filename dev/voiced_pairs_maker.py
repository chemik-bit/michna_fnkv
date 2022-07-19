from pathlib import Path
import pickle
import random
import cv2
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
    for healthy_image_path in image_paths_healthy:
        print(f"processing {healthy_image_path}")
        # read image
        image = cv2.imread(healthy_image_path)
        # resize image according to desired_image_size
        image = cv2.resize(image, desired_image_size, interpolation=cv2.INTER_AREA)
        random_images = random.sample(image_paths_healthy, instance_pairs)
        for random_path in random_images:
            random_paths.append([healthy_image_path, random_path])
            # sample random images
            random_image = cv2.imread(random_path)
            random_image = cv2.resize(random_image, desired_image_size, interpolation=cv2.INTER_AREA)
            image_pairs.append([image, random_image])
            pairs_labels.append(1)

        random_images = random.sample(image_paths_nonhealthy, instance_pairs)
        for random_path in random_images:
            # sample random images
            random_paths.append([healthy_image_path, random_path])
            random_image = cv2.imread(random_path)
            random_image = cv2.resize(random_image, desired_image_size, interpolation=cv2.INTER_AREA)
            image_pairs.append([image, random_image])
            pairs_labels.append(0)

    for nonhealthy_image_path in image_paths_nonhealthy:
        print(f"processing {nonhealthy_image_path}")
        # read image
        image = cv2.imread(nonhealthy_image_path)
        # resize image according to desired_image_size
        image = cv2.resize(image, desired_image_size, interpolation=cv2.INTER_AREA)
        random_images = random.sample(image_paths_nonhealthy, instance_pairs)
        for random_path in random_images:
            random_paths.append([nonhealthy_image_path, random_path])
            # sample random images
            random_image = cv2.imread(random_path)
            random_image = cv2.resize(random_image, desired_image_size, interpolation=cv2.INTER_AREA)
            image_pairs.append([image, random_image])
            pairs_labels.append(1)

        random_images = random.sample(image_paths_healthy, instance_pairs)
        for random_path in random_images:
            random_paths.append([nonhealthy_image_path, random_path])
            # sample random images
            random_image = cv2.imread(random_path)
            random_image = cv2.resize(random_image, desired_image_size, interpolation=cv2.INTER_AREA)
            image_pairs.append([image, random_image])
            pairs_labels.append(0)


    with open(path_to_save, "wb") as f:
        pickle.dump(image_pairs, f)
        pickle.dump(pairs_labels, f)

    print(random_paths)
    print(pairs_labels)

pickled_sets = {"train": (Path("../data/voiced_train.pickled"), Path("../data/voiced_pairs_train.pickled"), 10),
                "validation": (Path("../data/voiced_validation.pickled"),
                               Path("../data/voiced_pairs_validation.pickled"), 5),
                "test": (Path("../data/voiced_test.pickled"), Path("../data/voiced_pairs_test.pickled"), 5)
                }

for item in pickled_sets.values():
    with open(item[0], "rb") as f:
        image_paths = pickle.load(f)
        image_labels = pickle.load(f)

    make_pairs(image_paths, image_labels, (224, 224), item[1], item[2])
    with open(item[1], "rb") as f:
        pairs = pickle.load(f)
        labels = pickle.load(f)

    print(f"{item[1]} - {len(labels)}")
