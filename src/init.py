# https://stackoverflow.com/questions/5710867/downloading-and-unzipping-a-zip-file-without-writing-to-disk

from io import BytesIO
from zipfile import ZipFile
from urllib.request import urlopen

from src.utility.file_utility import remove_folder_tree, move_file, move_directory, create_directory


def init_training_data_folder(training_zip_path='http://benchmark.ini.rub.de/Dataset/GTSRB_Final_Training_Images.zip',
                              out_path='data/training/'):
    extract_dir_from_web(training_zip_path, out_path)
    move_file('data/training/GTSRB/Readme-Images.txt', 'data/training/readme-images.txt')
    move_directory('data/training/GTSRB/Final_Training/Images', 'data/training/images')
    remove_folder_tree('data/training/GTSRB')


def init_testing_data_folder(testing_zip_path='http://benchmark.ini.rub.de/Dataset/GTSRB_Final_Test_Images.zip',
                             out_path='data/testing/'):
    extract_dir_from_web(testing_zip_path, out_path)
    move_file('data/testing/GTSRB/Readme-Images-Final-test.txt', 'data/testing/readme-images-final-test.txt')
    move_directory('data/testing/GTSRB/Final_Test/Images', 'data/testing/images')
    remove_folder_tree('data/testing/GTSRB')


def init_testing_id_file(testing_zip_id_path='http://benchmark.ini.rub.de/Dataset/GTSRB_Final_Test_GT.zip',
                         out_path='data/testing/'):
    extract_dir_from_web(testing_zip_id_path, out_path)


def extract_dir_from_web(url, out_dir):
    print('Loading file from url ... \t(' + url + ')')
    res = urlopen(url)

    with ZipFile(BytesIO(res.read())) as zip_object:
        print('Extracting file in ' + out_dir)
        zip_object.extractall(out_dir)


def get_file_name(path):
    return path.split('/')[-1]

