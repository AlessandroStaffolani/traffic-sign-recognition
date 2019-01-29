from src.utility.image_utility import load_image, show_image
from src.Preprocessor import Preprocessor
from src.utility.file_utility import get_directory_files


def test_image_load_and_resize():
    image = load_image('data/training/images/00000/00000_00000.ppm')
    print(image.shape)
    print(image)
    show_image(image, '00000_00000.ppm', '00000')

    preprocessor = Preprocessor()
    resized = preprocessor.resize_image(image)
    print(resized.shape)
    print(resized)
    show_image(resized, 'resized 00000_00000.ppm', '00000')


def test_dirlist():
    folder_1 = 'data/training/images/'
    print(get_directory_files(folder_1))
    folder_2 = 'data/training/images/00000'
    print(get_directory_files(folder_2))
