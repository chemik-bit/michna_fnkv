"""
Script to make all possible unique pairs for siamese network.
It needs:
1) path to spectrogram images folder
2) list with image labels (0/1)
"""
from pathlib import Path
import pickle
import random
import sys
import os
import itertools


def make_pairs(images_paths: Path, path_to_save: Path):
    """
    Load images and make pairs for siamese network and save pickle it.
    :param path_to_save: path to pickled dataset
    :param images_paths: paths to images with spectrograms
    :return: None
    """

    image_paths_healthy = []
    image_paths_nonhealthy = []
    # load images and resize them
    for image_path in images_paths:
        if "nonhealthy" in str(image_path.resolve()):
            image_paths_nonhealthy.append(str(image_path.resolve()))
        else:
            image_paths_healthy.append(str(image_path.resolve()))
    print(f"{len(image_paths_healthy)} - healthy, {len(image_paths_nonhealthy)} - nonhealthy")
    image_pairs = []
    image_labels = []
    pairs = list(itertools.product(image_paths_healthy, image_paths_healthy))
    image_pairs += pairs
    image_labels += len(pairs) * [1]
    print(f"{len(image_pairs)} - {len(image_labels)}")

    pairs = list(itertools.product(image_paths_nonhealthy, image_paths_nonhealthy))
    image_pairs += pairs
    image_labels += len(pairs) * [1]

    pairs = list(itertools.product(image_paths_healthy, image_paths_nonhealthy))
    image_pairs += pairs
    image_labels += len(pairs) * [0]
    temp = list(zip(image_pairs, image_labels))
    random.shuffle(temp)
    res1, res2 = zip(*temp)
    image_pairs, image_labels = list(res1), list(res2)

    tf_dict = {"data": [], "labels": []}

    # dump pairs to smaller files (due to memory limitation)
    pairs_in_file = 4000
    for idx, image_pair in enumerate(image_pairs):
        tf_dict["data"].append(image_pair)
        tf_dict["labels"].append(image_labels[idx])
        if idx > 0 and idx % pairs_in_file == 0:
            path_to_save_pickle = path_to_save.joinpath(f"voiced_pairs_paths_{int(idx / pairs_in_file):05d}.pickled")
            with open(path_to_save_pickle, "wb") as f:
                pickle.dump(tf_dict, f)
            print(f"Saving to {path_to_save_pickle}")
            tf_dict = {"data": [], "labels": []}

    with open(path_to_save.joinpath("voiced_pairs.pickled"), "wb") as f:
        pickle.dump(tf_dict, f)


if os.name == "nt":
    from config import WINDOWS_PATHS as PATHS
else:
    from config import CENTOS_PATHS as PATHS
os.chdir(sys.path[1])

pickled_sets = {"train": (PATHS["PATH_DATASET_TRAIN"],
                          PATHS["PATH_DATASET"].joinpath("voiced_train.pickled")),
                "validation": (PATHS["PATH_DATASET_VAL"],
                               PATHS["PATH_DATASET"].joinpath("voiced_validation.pickled")),
                "test": (PATHS["PATH_DATASET_TEST"],
                         PATHS["PATH_DATASET"].joinpath("voiced_test.pickled"))
                }

for item in pickled_sets.values():
    with open(item[1], "rb") as f:
        image_paths = pickle.load(f)

    make_pairs(image_paths, item[0])
