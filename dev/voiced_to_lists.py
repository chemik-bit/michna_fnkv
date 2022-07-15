from pathlib import Path
import pickle

# path to voiced database
voiced_path = Path("../data/voiced")
test_sample_size = 30
validation_sample_size = 40
# list with paths to wav files
voiced_spectrograms = []
# list with wav file status (healthy/nonhealthy)
voiced_target = []

for description_file in voiced_path.glob("*.hea"):
    # append path to wav file
    voiced_spectrograms.append(voiced_path.joinpath("spectrograms",description_file.stem + ".png").resolve())
    with open(description_file, "r") as f:
        data = f.readlines()
        # append to target the diagnosis corresponding to wav file (healthy/nonhealthy)
        if "healthy" in data[-1]: print(data[-1])
        if "healthy" in data[-1]:
            voiced_target.append(1) # 1 represents healthy subject
        else:
            voiced_target.append(0) # 0 represents nonhealthy subject

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
