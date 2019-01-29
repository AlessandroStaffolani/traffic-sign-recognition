import logging
from os import listdir
import pandas as pd
from src.utility.image_utility import load_image, resize_image
from src.utility.dataset_utility import get_image_label

logging.basicConfig(filename='log/preprocessor.log', level=logging.INFO,
                    format='%(levelname)s: %(asctime)s: %(message)s')


class Preprocessor:

    def __init__(self, save_path, image_shape=46, labels=None, image_ext='ppm'):
        """
        Preprocessor for training images
        :param save_path: path where will be saved the output of the preprocessor
        :param image_shape: size at which each image will be resized. Image shape should be a squared image
        :param labels: list containing all the possible labels for the images
        :param image_ext: image file extension
        """
        self.save_path = save_path
        self.image_shape = image_shape
        self.labels = labels
        self.image_ext = image_ext

        self.data_folder = None  # Folder containing the images to process

        self.images_to_process = list()  # List with the file name of all the images to process
        self.images_processed = 0  # Index of the current image to process

        self.dataframe = None  # Will contain the dataframe object with the images resized and their label value

        self.current_label = None  # Current folder or label that we are processing

        self.total_images_processed = 0

    def init(self):
        logging.debug('init called for training')
        if self.data_folder is None:
            print('data_folder attribute is None! Should be a string')
            logging.warning('data_folder attribute is None')
            return False

        try:
            # Get all the images from data_folder
            self.images_to_process = listdir(self.data_folder)
        except FileNotFoundError:
            print('Folder "' + self.data_folder + ' not found')
            logging.error('Folder "' + self.data_folder + ' not found')
            return False

        # Sort the image names and filter to only those with image_ext in the name
        self.images_to_process.sort()
        self.images_to_process = list(filter(lambda file: self.image_ext in file, self.images_to_process))

        # Create the list with all the columns of the dataframe that will be saved on csv
        columns = ['image'] + self.labels

        self.dataframe = pd.DataFrame(columns=columns)

        return True

    def clean(self):
        logging.debug('clean called')
        self.data_folder = None
        self.images_to_process = list()
        self.images_processed = 0
        del self.dataframe
        self.dataframe = None
        self.current_label = None

    def set_data_folder(self, data_folder):
        self.data_folder = data_folder
        logging.info('setted new data_folder: ' + str(self.data_folder))

    def set_current_label(self, label):
        self.current_label = label
        logging.info('setted new current_label: ' + str(self.current_label))

    def status(self):
        return {
            'image_to_process': len(self.images_to_process),
            'image_processed': self.images_processed,
            'percentage': (100 * self.images_processed) / len(self.images_to_process),
            'next': self.get_next(),
            'total_processed': self.total_images_processed
        }

    def get_next(self):
        if self.images_processed >= len(self.images_to_process):
            image = None
        else:
            image = self.images_to_process[self.images_processed]
        logging.debug('get_next result: ' + str(image))
        return image

    def process_next(self):
        next_image_name = self.get_next()
        if next_image_name is None or self.current_label is None:
            logging.warning('next_image_name is None or current_label is None')
            return False
        else:
            # load image
            next_image = load_image(self.data_folder + '/' + next_image_name)
            resized = resize_image(next_image, (self.image_shape, self.image_shape))

            label = get_image_label(self.current_label, self.labels)
            self.add_to_dataframe(resized, label)

            # update image processed
            self.images_processed += 1
            self.total_images_processed += 1

            logging.debug(str(next_image_name) + ' preprocessed and added on dataframe')

            return True

    def add_to_dataframe(self, image, label=None):
        self.dataframe.loc[self.images_processed] = [image] + label

    def save_results(self):
        if self.dataframe is not None:
            self.dataframe.to_csv(index=False, mode='a', path_or_buf=self.save_path, header=False)
            logging.info('saved into ' + str(self.save_path) + ' ' + str(len(self.dataframe)) + ' images')


class PreprocessorTest(Preprocessor):

    def __init__(self, save_path, batch_size=1000):
        super().__init__(save_path)
        self.batch_size = batch_size
        self.batch_break_count = 0

    def init(self):
        logging.debug('init called for testing')
        if self.data_folder is None:
            print('data_folder attribute is None! Should be a string')
            logging.warning('data_folder attribute is None')
            return False

        try:
            # Get all the images from data_folder
            self.images_to_process = listdir(self.data_folder)
        except FileNotFoundError:
            print('Folder "' + self.data_folder + ' not found')
            logging.error('Folder "' + self.data_folder + ' not found')
            return False

        # Sort the image names and filter to only those with image_ext in the name
        self.images_to_process.sort()
        self.images_to_process = list(filter(lambda file: self.image_ext in file, self.images_to_process))

        # Create the list with all the columns of the dataframe that will be saved on csv
        columns = ['image']

        self.dataframe = pd.DataFrame(columns=columns)

        return True

    def process_next(self):
        next_image_name = self.get_next()
        if next_image_name is None:
            logging.warning('next_image_name is None')
            return False
        else:
            # load image
            next_image = load_image(self.data_folder + '/' + next_image_name)
            resized = resize_image(next_image, (self.image_shape, self.image_shape))

            self.add_to_dataframe(resized)

            # update image processed
            self.images_processed += 1
            self.total_images_processed += 1
            self.batch_break_count += 1

            if self.batch_break_count == self.batch_size:
                self.save_batch_and_reinit_df()

            logging.debug(str(next_image_name) + ' preprocessed and added on dataframe')

            return True

    def add_to_dataframe(self, image, label=None):
        self.dataframe.loc[self.images_processed] = [image]

    def save_batch_and_reinit_df(self):
        self.save_results()
        del self.dataframe
        self.dataframe = self.dataframe = pd.DataFrame(columns=['image'])
        self.batch_break_count = 0
