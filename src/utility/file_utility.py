import os


def get_directory_files(path):
    files = os.listdir(path)
    return files


def save_to_file(file, content):
    file = open(file, mode='w')
    file.write(content)
    file.close()


def remove_file(file):
    try:
        os.remove(file)
        return True
    except FileNotFoundError:
        print(file + ' not found')
        return False


def create_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
