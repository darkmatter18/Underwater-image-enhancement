import os


def mkdirs(paths):
    """create empty directories if they don't exist

    :param paths: a list of directory paths
    """
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    """create a single empty directory if it didn't exist

    :param path: a single directory path
    """
    if not os.path.exists(path):
        os.makedirs(path)
