import os

def dircheck(path2dir):
    """
    Checks if a directory exists! if it does not exist, it creates it
    Args:
    dir_path    -   path to the directory you want to be created!
    """

    if not os.path.exists(path2dir):
        print(f"creating {path2dir}")
        os.makedirs(path2dir)