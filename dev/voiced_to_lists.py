from pathlib import Path
import pickle

# path to voiced database
voiced_path = Path("../data/voiced")
images_for_test = 30
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
        if "healthy" in data[-1]:
            voiced_target.append(1) # 1 represents healthy subject
        else:
            voiced_target.append(0) # 0 represents nonhealthy subject

# check print
print(voiced_spectrograms[-images_for_test:])
print(voiced_target[-images_for_test:])

# pickle first paths then target
with open(Path("../data/voiced_train.pickled"), "wb") as f:
    pickle.dump(voiced_spectrograms[:-images_for_test], f)
    pickle.dump(voiced_target[:-images_for_test], f)

# pickle first paths then target
with open(Path("../data/voiced_test.pickled"), "wb") as f:
    pickle.dump(voiced_spectrograms[-images_for_test:], f)
    pickle.dump(voiced_target[-images_for_test:], f)