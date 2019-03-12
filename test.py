from src.utility.file_utility import get_directory_files
from src.utility.dataset_utility import get_labels
import sys
import numpy as np
import cv2 as cv
from skimage import io, color, exposure, transform
import pandas as pd
from keras.utils import to_categorical

from zipfile import ZipFile

from src.models.Model import Model
from src.utility.dataset_utility import create_traing_data_table
from src.DataGenerator import DataGenerator
from src.utility.dataset_utility import split_train_data, save_images_roi
from src.utility.image_utility import load_image, resize_image, img_to_grayscale, show_image, equalize_image_lightness, \
    equalize_colored_image
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


def test_dataset_table_creation():
    create_traing_data_table('data/training/images', 'data/training/training_table.csv')


def preprocess_img(img, img_size=48):
    # Histogram normalization in y
    hsv = color.rgb2hsv(img)
    hsv[:,:,2] = exposure.equalize_hist(hsv[:,:,2])
    img = color.hsv2rgb(hsv)

    # central scrop
    min_side = min(img.shape[:-1])
    centre = img.shape[0] // 2, img.shape[1] // 2
    img = img[centre[0] - min_side // 2:centre[0] + min_side // 2,
          centre[1] - min_side // 2:centre[1] + min_side // 2,
          :]

    # rescale to standard size
    img = transform.resize(img, (img_size, img_size))

    # roll color axis to axis 0
    img = np.rollaxis(img, -1)

    return img


def main(argv):
    image = load_image('data/training/images/00000/00000_00001.ppm')

    # image = np.array(image)[:, :, np.newaxis]

    datatable = pd.read_csv('data/training/training_table.csv', sep=',', nrows=210)

    save_images_roi(datatable['image_path'].values, datatable['label'].values, 'data/prova', datatable)

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

    # image_path = 'data/train/00000/00000_00001.ppm'
    # image = load_image(image_path)
    #
    # # central scrop
    # min_side = min(image.shape[:-1])
    # centre = image.shape[0] // 2, image.shape[1] // 2
    # img = image[centre[0] - min_side // 2:centre[0] + min_side // 2,
    #       centre[1] - min_side // 2:centre[1] + min_side // 2,
    #       :]
    # show_image(img, 'Central scrop')
    #
    # df = pd.read_csv('data/training/images/00000/GT-00000.csv', nrows=2, sep=';')
    # row = df.iloc[0]
    #
    # image = image[row['Roi.X1']: row['Roi.X2'], row['Roi.Y1']: row['Roi.Y2'], :]
    #
    # # image = resize_image(image, (200, 200))
    # show_image(image, 'ROI')
    #
    # show_image(image, 'No equalization')
    # equalized = equalize_image_lightness(image)
    # show_image(equalized, 'Equalized HSV')
    # equalized_colored = equalize_colored_image(image)
    # show_image(equalized_colored, 'Equalized Colored')


if __name__ == '__main__':
    main(sys.argv)
