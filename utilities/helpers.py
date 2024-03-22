"""
Modul with various helper-functions.
"""
from pathlib import Path


def clear_folder(folder_path: Path):
    """
    Deletes all files in given folder_path
    :param folder_path: Path to folder where all files will be deleted
    :return: None
    """
    files_to_delete = folder_path.glob("*")
    for file in files_to_delete:
        file.unlink()

def count_large_files():
    """
    Reads through the files in ../data/svdadult/svdadult_renamed/ and checks their length.
    If the number of floats is > 100k, it adds 1 to the sum and calculates this sum.
    :return: The sum of files with more than 100k floats.
    """
    svdadult_renamed_path = Path(__file__).parent.parent.joinpath("data/svdadult/svdadult_renamed/")
    file_count = 0
    total_files = len(list(svdadult_renamed_path.glob("*.txt")))
    print(f"Total number of files in the directory: {total_files}")
    for file in svdadult_renamed_path.glob("*.txt"):
        line_count = sum(1 for line in open(file, "r"))
        if line_count > 100000:  # Check if the file has more than 100k lines
            file_count += 1
    return file_count

#count = count_large_files()
#print(count)

