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
        if line_count < 50000:  # Check if the file has more than 100k lines
            file_count += 1
    return file_count

count = count_large_files()
print(count)

def delete_small_files(min_line_count: int) -> int:
    """
    Deletes files in ../data/svdadult/svdadult_renamed/ if they have fewer than min_line_count lines and returns the count of deleted files.
    :param min_line_count: Minimum number of lines a file must have to not be deleted.
    :return: The number of files deleted.
    """
    svdadult_renamed_path = Path(__file__).parent.parent.joinpath("data/svdadult/svdadult_renamed/")
    deleted_files_count = 0
    for file in svdadult_renamed_path.glob("*.txt"):
        line_count = sum(1 for line in open(file, "r"))
        if line_count <= min_line_count:  # Check if the file has fewer than min_line_count lines
            file.unlink()  # Delete the file
            deleted_files_count += 1
    return deleted_files_count

#deleted_count = delete_small_files(50000)
#print(deleted_count)