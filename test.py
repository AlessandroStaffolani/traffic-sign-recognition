from src.utility.file_utility import get_directory_files
from src.utility.dataset_utility import get_labels
import sys
import numpy as np
import pandas as pd
from keras.utils import to_categorical

from zipfile import ZipFile

from src.models.Model import Model
from src.Dataset import Dataset
from src.utility.preprocessor_utility import preprocess_image
from src.utility.dataset_utility import create_traing_data_table
from src.DataGenerator import DataGenerator
from src.utility.dataset_utility import split_train_data
from src.utility.image_utility import load_image, resize_image, img_to_grayscale, show_image
from src.utility.system_utility import progress_bar
from src.preprocessors.HistogramEqualizer import HistogramEqualizer
from src.preprocessors.Normalizer import Normalizer
from src.Pipeline import Pipeline

from src.init import init_training_data_folder, init_testing_id_file, init_testing_data_folder

np.random.seed(42)


def test_dirlist():
    folder_1 = 'data/training/images/'
    print(get_directory_files(folder_1))
    folder_2 = 'data/training/images/00000'
    print(get_directory_files(folder_2))


def test_cnn():
    cnn = Model(num_output=43)
    print(cnn.model.to_json())


def test_dataset():
    labels = get_labels(43, False)
    dataset = Dataset('data/training/images', preprocess_image, labels=labels)

    dataframe = dataset.get_images_generator('data/training/training_table.csv')
    for i in range(100):
        row = next(dataframe)
        print(row[0].shape, row[1])


def test_dataset_table_creation():
    create_traing_data_table('data/training/images', 'data/training/training_table.csv')


def main(argv):
    # image = load_image('data/training/images/00000/00000_00001.ppm', 0)
    #
    # image = np.array(image)[:, :, np.newaxis]
    #
    # print(image.shape)
    # print(image)
    #
    # pipeline = Pipeline(verbose=True)
    # pipeline.add_preprocessors((
    #     HistogramEqualizer(),
    #     Normalizer()
    # ))
    #
    # out = pipeline.evaluate(image)
    #
    # print(out.shape)
    # print(out)
    #
    # show_image(image, 'No elaboration')
    # show_image(out, 'Elaborated')

    init_testing_id_file()
    init_testing_data_folder()
    init_training_data_folder()


if __name__ == '__main__':
    main(sys.argv)
