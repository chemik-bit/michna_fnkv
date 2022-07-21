from pathlib import Path
import pickle
import random

# path to voiced database
voiced_path = Path("../data/voiced_renamed/spectrograms")
test_sample_size = 80
validation_sample_size = 80
# list with paths to wav files
voiced_spectrograms = []
# list with wav file status (healthy/nonhealthy)
voiced_target = []
spectograms_list = list(voiced_path.glob("*.png"))
random.shuffle(spectograms_list)
for spectrogram_file in spectograms_list:
    # append path to wav file
    voiced_spectrograms.append(spectrogram_file.resolve())
    if "nonhealthy" in spectrogram_file.stem:
        voiced_target.append(0)
    else:
        voiced_target.append(1)
print(voiced_spectrograms)
print(voiced_target)
# check print
print("test ", len(voiced_target[-(test_sample_size + validation_sample_size):-validation_sample_size]))
print("train ", len(voiced_target[:-(test_sample_size + validation_sample_size)]))
print("validation ", len(voiced_target[-validation_sample_size:]))


# pickle first paths then target
with open(Path("../data/voiced_train.pickled"), "wb") as f:
    pickle.dump(voiced_spectrograms[:-(test_sample_size + validation_sample_size)], f)
    pickle.dump(voiced_target[:-(test_sample_size + validation_sample_size)], f)

# pickle first paths then target
with open(Path("../data/voiced_test.pickled"), "wb") as f:
    pickle.dump(voiced_spectrograms[-(test_sample_size + validation_sample_size):-validation_sample_size], f)
    pickle.dump(voiced_target[-(test_sample_size + validation_sample_size):-validation_sample_size], f)

with open(Path("../data/voiced_validation.pickled"), "wb") as f:
    pickle.dump(voiced_spectrograms[-validation_sample_size:], f)
    pickle.dump(voiced_target[-validation_sample_size:], f)
