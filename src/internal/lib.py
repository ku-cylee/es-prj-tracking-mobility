import os

def mkdir_nonexist(path):
    if not os.path.exists(path):
        os.mkdir(path)
