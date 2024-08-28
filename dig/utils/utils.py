import os
import sys


def get_videos(
    root_dir: str, excluded_dirs: list[str] = [], extentions: list[str] = []
) -> list[str]:
    """Get the videos or video frames from the root directory and its sub-directories (if exist)

    Args:
        root_dir (str): The root directory
        excluded_dirs (List[str], optional): A lsit containing sub-dirs to exclude. Defaults to []
        extentions (List[str], optional): A list containing the extentions of the files to get. Defaults to []

    Returns:
        List[str]: A list containing the absolute paths of the video images found in the given directory
    """
    videos = []
    for dir_name, subdir_list, file_list in os.walk(root_dir):
        for subdir in (f.name for f in os.scandir(dir_name) if f.is_dir()):
            if subdir in excluded_dirs:
                print(f"Excluding sub-directory: {subdir}")
                subdir_list.remove(subdir)
            print("\t- subdirectory: %s" % subdir)

        for fname in file_list:
            if fname.endswith(tuple(extentions)):
                file_path = os.path.abspath(os.path.join(dir_name, fname))
                videos.append(file_path)
            else:
                file_path = os.path.abspath(os.path.join(dir_name, fname))
                videos.append(file_path)
    print(f"\nNumber of videos found: {len(videos)}")
    return videos


def calculate_execution_time(start_time: float, end_time: float) -> tuple:
    """
    Calculate the execution time of a program.

    Args:
        start_time (float): The start time of the program.
        end_time (float): The end time of the program.

    Returns:
        tuple: The execution time in minutes, seconds, and milliseconds.
    """
    execution_time = end_time - start_time
    minutes, seconds = divmod(execution_time, 60)
    seconds, milliseconds = divmod(seconds, 1)
    milliseconds = int(milliseconds * 1000)
    print(
        f"Program executed in {int(minutes)} minutes, {int(seconds)} seconds, and {milliseconds} milliseconds"
    )


def blockPrint():
    """Block printing messages to the console."""
    sys.stdout = open(os.devnull, "w")


def is_dir(directory: str) -> bool:
    """Check if the given directory exists.

    Args:
        directory (str): The directory to check.

    Returns:
        bool: True if the directory exists, False otherwise.
    """
    return os.path.isdir(directory)


def is_file(filename: str) -> bool:
    """Check if the given file exists.

    Args:
        filename (str): The file to check.

    Returns:
        bool: True if the file exists, False otherwise.
    """
    return os.path.isfile(path=filename)


def create_multiple_dirs(path: str):
    """
    Create multiple directories recursively if they don't exist.

    Args:
        path (str): The path to create the directories.
    """
    if not os.path.exists(path):
        os.makedirs(path)


def create_dir(directory: str) -> bool:
    """Create a directory if it doesn't exist.

    Args:
        directory (str): The directory to create.

    Returns:
        bool: True if the directory was created, False otherwise.
    """
    try:
        return os.mkdir(directory)
    except FileExistsError:
        print(f"{directory} already exists")
        return False


def crawl_directory(directory: str, extension: str = None) -> list:
    """Crawling data directory
    Args:
        directory (str) : The directory to crawl
    Returns:
        tree (list)     : A list with all the filepaths
    """
    tree = []
    subdirs = [folder[0] for folder in os.walk(directory)]

    for subdir in subdirs:
        files = next(os.walk(subdir))[2]
        for _file in files:
            lowercase_extension = extension.lower()
            uppercase_extension = extension.upper()
            if _file.endswith(lowercase_extension) or _file.endswith(
                uppercase_extension
            ):
                tree.append(os.path.join(subdir, _file))
            else:
                tree.append(os.path.join(subdir, _file))

    if tree:
        print(f"Found {len(tree)} files in {directory}")
    return tree
