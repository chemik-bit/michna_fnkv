"""
Script that creates lists with filenames and their respective labels. Those lists are pickled.
Should be used to create train, validation and test sets.
"""
from pathlib import Path
import pickle
import os, sys
if os.name == "nt":
    from config import WINDOWS_PATHS as PATHS
else:
    from config import CENTOS_PATHS as PATHS
os.chdir(sys.path[1])

test_sample_size = 20  # number of files
validation_sample_size = 20  # number of files

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
