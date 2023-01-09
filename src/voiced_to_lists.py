"""
Voiced to train/validation/test sets.
"""
import pickle
import os
import sys


def voiced_to_pickle(validation_sample_size: int, test_sample_size: int = 0):
    """
    Function that creates lists with filenames and their respective labels. Those lists are pickled.
    Should be used to create train, validation and test histograms.
    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    !!!!!!!  The validation set size has to correspond with number of chunks. I.e. there are 3 spectrograms !!!!!
    !!!!!!!  for each patient. So the validation set size should be x * 3. Same for test size..             !!!!!
    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    :param validation_sample_size: size of validation set
    :param test_sample_size: size of test set
    :return: None
    """
    if os.name == "nt":
        from config import WINDOWS_PATHS as PATHS
    else:
        from config import CENTOS_PATHS as PATHS
    os.chdir(sys.path[1])

    # path to voiced database
    voiced_spectrograms_path = PATHS["PATH_SPECTROGRAMS"]

    # list with paths to wav files
    voiced_spectrograms = []
    # list with wav file status (healthy/nonhealthy)
    voiced_target = []
    spectograms_list = list(voiced_spectrograms_path.glob("*.png"))
    # random.shuffle(spectograms_list)
    for spectrogram_file in spectograms_list:
        # append path to wav file
        voiced_spectrograms.append(spectrogram_file.resolve())
        if "nonhealthy" in spectrogram_file.stem:
            voiced_target.append(0)
        else:
            voiced_target.append(1)

    print("test ", len(voiced_target[-(test_sample_size + validation_sample_size):-validation_sample_size]))
    print("train ", len(voiced_target[:-(test_sample_size + validation_sample_size)]))
    print("validation ", len(voiced_target[-validation_sample_size:]))


    # pickle first paths then target
    with open(PATHS["PATH_DATASET"].joinpath("voiced_train.pickled"), "wb") as f:
        pickle.dump(voiced_spectrograms[:-(test_sample_size + validation_sample_size)], f)
        pickle.dump(voiced_target[:-(test_sample_size + validation_sample_size)], f)

    # pickle first paths then target
    with open(PATHS["PATH_DATASET"].joinpath("voiced_test.pickled"), "wb") as f:
        pickle.dump(voiced_spectrograms[-(test_sample_size + validation_sample_size):-validation_sample_size], f)
        pickle.dump(voiced_target[-(test_sample_size + validation_sample_size):-validation_sample_size], f)

    with open(PATHS["PATH_DATASET"].joinpath("voiced_validation.pickled"), "wb") as f:
        pickle.dump(voiced_spectrograms[-validation_sample_size:], f)
        pickle.dump(voiced_target[-validation_sample_size:], f)
    print(voiced_spectrograms[-validation_sample_size:])

"""
Voiced to train/validation/test sets.
"""

def voiced_to_list():
    """
    Function that creates lists with filenames and their respective labels.
    Should be used to utilize tf.data.Dataset.
    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    !!!!!!!  The validation set size has to correspond with number of chunks. I.e. there are 3 spectrograms !!!!!
    !!!!!!!  for each patient. So the validation set size should be x * 3. Same for test size..             !!!!!
    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    :return: (voiced_histograms, voiced_targets): list with paths to histograms, list with target values
    """
    if os.name == "nt":
        from config import WINDOWS_PATHS as PATHS
    else:
        from config import CENTOS_PATHS as PATHS
    os.chdir(sys.path[1])

    # path to voiced database
    voiced_spectrograms_path = PATHS["PATH_SPECTROGRAMS"]

    # list with paths to wav files
    voiced_spectrograms = []
    # list with wav file status (healthy/nonhealthy)
    voiced_target = []
    spectograms_list = list(voiced_spectrograms_path.glob("*.png"))
    # random.shuffle(spectograms_list)
    for spectrogram_file in spectograms_list:
        # append path to wav file
        voiced_spectrograms.append(str(spectrogram_file.resolve()))
        if "nonhealthy" in spectrogram_file.stem:
            voiced_target.append(0)
        else:
            voiced_target.append(1)

    return voiced_spectrograms, voiced_target
