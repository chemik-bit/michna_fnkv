from pathlib import Path

# PATHS
WINDOWS_PATHS = {
    "PATH_DATA": Path(__file__).parent.joinpath("./data"),
    "PATH_SOUNDFILES": Path(__file__).parent.joinpath("./data/sound_files"),
    "PATH_MONO": Path(__file__).parent.joinpath("./data/mono"),
    "PATH_VOICED": Path(__file__).parent.joinpath("./data/voiced"),
    "PATH_VOICED_RENAMED": Path(__file__).parent.joinpath("./data/voiced/voiced_renamed"),
    "PATH_VOICED_WAV": Path(__file__).parent.joinpath("./data/voiced/wav"),
    "PATH_SVD": Path(__file__).parent.joinpath("./data/svd"),
    "PATH_SVD_RENAMED": Path(__file__).parent.joinpath("./data/svd/svd_renamed"),
    "PATH_SVD_WAV": Path(__file__).parent.joinpath("./data/svd/wav"),
    "PATH_WAV": Path(__file__).parent.joinpath("./data/wav"),
    "PATH_DATASET": Path(__file__).parent.joinpath("./data/dataset"),
    "PATH_DATASET_TRAIN": Path(__file__).parent.joinpath("./data/dataset/train"),
    "PATH_DATASET_TEST": Path(__file__).parent.joinpath("./data/dataset/test"),
    "PATH_DATASET_VAL": Path(__file__).parent.joinpath("./data/dataset/val"),
    "PATH_SPECTROGRAMS": Path(__file__).parent.joinpath("./data/spectrograms"),
    "PATH_EXPERIMENTS": Path("C:/siamese_runs"),
    "PATH_CSV": Path(__file__).parent.joinpath("./data")
}

CENTOS_PATHS = {
    "PATH_DATA": Path(__file__).parent.joinpath("./data"),
    "PATH_SOUNDFILES": Path(__file__).parent.joinpath("./data/sound_files"),
    "PATH_MONO": Path(__file__).parent.joinpath("./data/mono"),
    "PATH_VOICED": Path(__file__).parent.joinpath("./data/voiced"),
    "PATH_VOICED_RENAMED": Path(__file__).parent.joinpath("./data/voiced/voiced_renamed"),
    "PATH_VOICED_WAV": Path(__file__).parent.joinpath("./data/voiced/wav"),
    "PATH_SVD": Path(__file__).parent.joinpath("./data/svd"),
    "PATH_SVD_RENAMED": Path(__file__).parent.joinpath("./data/svd/svd_renamed"),
    "PATH_SVD_WAV": Path(__file__).parent.joinpath("./data/svd/wav"),
    "PATH_WAV": Path(__file__).parent.joinpath("./data/wav"),
    "PATH_DATASET": Path(__file__).parent.joinpath("./data/dataset"),
    "PATH_DATASET_TRAIN": Path(__file__).parent.joinpath("./data/dataset/train"),
    "PATH_DATASET_TEST": Path(__file__).parent.joinpath("./data/dataset/test"),
    "PATH_DATASET_VAL": Path(__file__).parent.joinpath("./data/dataset/val"),
    "PATH_SPECTROGRAMS": Path(__file__).parent.joinpath("./data/spectrograms"),
    "PATH_EXPERIMENTS": Path("C:/siamese_runs"),
    "PATH_CSV": Path(__file__).parent.joinpath("./data")
}
