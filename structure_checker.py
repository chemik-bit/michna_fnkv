"""
Project directories and files checker.
"""
from pathlib import Path


def check_consistency():
    """
    Fast checker of project directory structure. Expected structure:
    ./data (data folder)
    ./data/sound_files (folder with sound files)
    ./data/mono (folder with mono sound files)
    ./data/spectrograms (folder with spectrograms)
    ./data/voiced (folder fo Voiced database files)
    ./data/database.db (database with patients information)
    (presence of files in ./data/sound_files is also checked)
    :return: None
    """
    consistency = True
    folders_to_check = {"data": Path("./data"),
                        "sound_files": Path("./data/sound_files"),
                        "mono": Path("./data/mono"),
                        "voiced": Path("./data/voiced"),
                        "spectrograms": Path("./data/spectrograms")}

    for to_check in folders_to_check.items():

        if to_check[1].exists():
            print(f"{to_check[1]} ...... {to_check[0]} folder - OK")
        else:
            print(f"Missing {to_check[1]} folder ......... creating")
            to_check[1].mkdir(parents=True, exist_ok=True)
            consistency = False

    # check db file
    if folders_to_check["data"].joinpath("database.db").exists():
        print("Database (database.db) .......... OK")
    else:
        print("Database missing in data folder.. ")
        consistency = False

    # check sound_files folder is empty
    if not any(folders_to_check["sound_files"].iterdir()):
        print(f'{folders_to_check["sound_files"].resolve()} folder is empty')
        consistency = False
    else:
        print("Files in sound_files .......... OK")

    if consistency:
        print("Project structure .......... OK")
    else:
        print("Inconsistent project. Check missing parts.")


if __name__ == "__main__":
    check_consistency()
