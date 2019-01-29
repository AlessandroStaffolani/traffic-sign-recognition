from os import listdir, remove


def get_directory_files(path):
    files = listdir(path)
    return files


def save_to_file(file, content):
    file = open(file, mode='w')
    file.write(content)
    file.close()


def remove_file(file):
    try:
        remove(file)
        return True
    except FileNotFoundError:
        print(file + ' not found')
        return False
