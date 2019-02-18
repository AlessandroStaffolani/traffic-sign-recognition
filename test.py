from src.utility.image_utility import load_image, show_image
from src.utility.file_utility import get_directory_files
from src.utility.dataset_utility import get_labels
import sys
import numpy as np

from src.Cnn import Cnn
from src.Dataset import Dataset
from src.utility.preprocessor_utility import preprocess_image


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
    dataset = Dataset('data/training/images', preprocess_image, chunk_size=1000, labels=labels)

    dataframe = dataset.get_images()

    for row in dataframe:
        images, labels = row[0], row[1]
        print(images.shape, end='\t')
        print(labels.shape, end='\n\n---------------\n')

    # images = next(dataframe)
    # for img in images:
    #     show_image(img, 'image')


def main(argv):
    # test_cnn()

    test_dataset()


if __name__ == '__main__':
    main(sys.argv)