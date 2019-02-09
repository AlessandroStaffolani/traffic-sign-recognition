from src.utility.image_utility import load_image, show_image
from src.Preprocessor import Preprocessor
from src.utility.file_utility import get_directory_files
from src.utility.dataset_utility import get_labels
import sys
import numpy as np

from src.Cnn import Cnn
from src.Dataset import Dataset


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


def test_cnn():
    cnn = Cnn(num_output=43)
    print(cnn.model.to_json())


def test_dataset():
    labels = get_labels(43)
    dataset = Dataset('data/train_data_processed/train_46x46_100.csv', chunk_size=10, labels=labels)

    generator = dataset.get_generator()
    for df in generator:
        images = df[0]
        labels = df[1]
        print(type(images[0]))




def main(argv):
    # test_cnn()

    # test_dataset()
    data = np.loadtxt('data/train_data_processed/train_46x46_100.csv', delimiter=',')
    print(data.shape)


if __name__ == '__main__':
    main(sys.argv)