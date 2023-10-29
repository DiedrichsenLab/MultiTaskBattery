
def dircheck(path2dir):
    """
    Checks if a directory exists! if it does not exist, it creates it
    Args:
    dir_path    -   path to the directory you want to be created!
    """

    if not os.path.exists(path2dir):
        print(f"creating {path2dir}")
        os.makedirs(path2dir)

# use dirtree to make sure you have all the folders needed
def dirtree():
    """
    Create all the directories if they don't already exist
    """

    fpaths = [raw_dir, stim_dir, target_dir, run_dir]
    for fpath in fpaths:
        dircheck(fpath)