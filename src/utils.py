import os


def make_directory(path):
    if not os.path.isdir(path):
        os.mkdir(path)
    return path