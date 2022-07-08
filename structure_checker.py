"""
Project directories and files checker.
"""
from pathlib import Path


def check_consistency():
    """
    Fast checker of project structure. Expected structure:
    ./data (data folder)
    ./data/sound_files (folder with sound files)
    ./data/mono (folder with mono sound files)
    ./data/spectrograms (folder with spectrograms)
    ./data/database.db (database with patients information)

    :return: None
    """
    consistency = True
    # check data folder
    data_folder_path = Path("./data")
    if data_folder_path.exists():
        print(f"{data_folder_path.resolve()} .......... data folder - OK")
    else:
        print("Missing data folder.. creating")
        data_folder_path.mkdir(parents=True, exist_ok=True)
        consistency = False

    # check soundfiles
    sound_files_folder = Path("./data/sound_files")
    if sound_files_folder.exists():
        print(f"{sound_files_folder.resolve()} .......... sound files folder - OK")
        if not any(sound_files_folder.iterdir()):
            print(f"{sound_files_folder.resolve()} is empty")
            consistency = False
        else:
            print("Files in sound_files .......... OK")
    else:
        print("Missing sound_files folder in data folder.. creating")
        sound_files_folder.mkdir(parents=True, exist_ok=True)

    # check mono folder
    mono_folder_path = Path("./data/mono")
    if mono_folder_path.exists():
        if mono_folder_path.exists():
            print(f"{mono_folder_path.resolve()} .......... mono folder - OK")
        else:
            print("Missing mono folder.. creating")
            mono_folder_path.mkdir(parents=True, exist_ok=True)
            consistency = False

    # check spectrograms folder
    spectrograms_folder_path = Path("./data/spectrograms")
    if spectrograms_folder_path.exists():
        if spectrograms_folder_path.exists():
            print(f"{spectrograms_folder_path.resolve()} .......... spectrogram folder - OK")
        else:
            print("Missing spectrogram folder.. creating")
            spectrograms_folder_path.mkdir(parents=True, exist_ok=True)
            consistency = False

    # check db file
    if data_folder_path.joinpath("database.db").exists():
        print("Database (database.db) .......... OK")
    else:
        print("Database missing in data folder.. ")
        consistency = False

    if consistency:
        print("Project structure .......... OK")
    else:
        print("Inconsistent project. Check missing parts.")


if __name__ == "__main__":
    check_consistency()
