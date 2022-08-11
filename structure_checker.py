"""
Project directories and files checker.
"""
import os
from colorama import Fore, Style

if os.name == "nt":
    from config import WINDOWS_PATHS as PATHS
else:
    from config import CENTOS_PATHS as PATHS


def check_consistency():
    """
    Fast checker of project directory structure. Expected structure is given by PATHS dictionary
    :return: None
    """
    consistency = True
    for to_check in PATHS.items():
        if to_check[1].exists():
            print(f"{to_check[1]} ...... {to_check[0]} folder - OK")
        else:
            print(f"Missing {to_check[1]} folder ......... creating")
            to_check[1].mkdir(parents=True, exist_ok=True)
            consistency = False

    # check db file
    if PATHS["PATH_DATA"].joinpath("database.db").exists():
        print("Database (database.db) .......... OK")
    else:
        print("Database missing in data folder.. ")
        consistency = False

    # check sound_files folder is empty
    if not any(PATHS["PATH_SOUNDFILES"].iterdir()):
        print(f'{PATHS["PATH_SOUNDFILES"].resolve()} folder is empty')
        consistency = False
    else:
        print("Files in sound_files .......... OK")

    if consistency:
        print(f"{Fore.LIGHTGREEN_EX}Project structure .......... OK{Style.RESET_ALL}")
    else:
        print(f"{Fore.RED}Inconsistent project. Check missing parts.{Style.RESET_ALL}")


if __name__ == "__main__":
    check_consistency()
