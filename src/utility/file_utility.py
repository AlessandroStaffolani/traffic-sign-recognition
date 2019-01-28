from os import listdir


def get_directory_files(path):
    files = listdir(path)
    return files


def save_to_file(file, content):
    file = open(file, mode='w')
    file.write(content)
    file.close()
