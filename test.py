from src.utility.file_utility import get_directory_files
from src.utility.dataset_utility import get_labels
import sys
import numpy as np
import pandas as pd
from keras.utils import to_categorical

from src.models.Model import Model
from src.Dataset import Dataset
from src.utility.preprocessor_utility import preprocess_image
from src.utility.dataset_utility import create_traing_data_table
from src.DataGenerator import DataGenerator
from src.utility.dataset_utility import split_train_data
from src.utility.image_utility import load_image, resize_image, img_to_grayscale
from src.utility.system_utility import progress_bar

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
    # test_cnn()
    # test_folder = 'data/testing/images'
    #
    # test_images_names = get_directory_files(test_folder)
    # test_images_names.sort()
    # test_table = pd.read_csv('data/testing/GT-final_test.csv', sep=';')
    # print(test_table.info())
    # print(test_table.head())
    # test_data = np.empty((12630, 46, 46, 1), dtype=np.uint8)
    # test_label = np.empty((12630, 43))
    # count = 0
    # print()
    # for image_name in test_images_names:
    #     if 'ppm' in image_name:
    #         image = load_image(test_folder + '/' + image_name)
    #         image = resize_image(image, (46, 46))
    #         image = img_to_grayscale(image)
    #
    #         label = test_table.iloc[count]
    #         label = label['ClassId']
    #
    #         test_data[count, ] = image[:, :, np.newaxis]
    #         test_label[count] = to_categorical([label], 43)
    #
    #         progress_bar(count, 12630, 'Collectring test images')
    #         count += 1
    #
    # print()
    # print(test_data.shape)
    # print(test_label.shape)
    # print()

    generator = DataGenerator('data/validation')

    model = Model()
    model.load_model('model/6-epochs')

    model.compile()

    scores = model.evaluate_generator(generator.get_generator(), 7842 // 32)

    print(scores)


if __name__ == '__main__':
    main(sys.argv)