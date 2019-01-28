import cv2
from os import listdir
import pandas as pd
from src.utility.image_utility import load_image
from src.utility.dataset_utility import get_image_label


class Preprocessor:

    def __init__(self, save_path, image_shape=46, training=True, labels=None, image_ext='ppm'):
        """
        Image shape should be a squared image
        :param save_path:
        :param image_shape:
        :param training:
        :param labels:
        :param image_ext:
        """
        self.data_folder = None
        self.image_shape = image_shape
        self.images_to_process = list()
        self.images_processed = 0
        self.is_training = training
        self.save_path = save_path
        self.processed = None
        self.labels = labels
        self.image_ext = image_ext
        self.current_label = None

    def init(self):
        if self.data_folder is None:
            return False

        try:
            self.images_to_process = listdir(self.data_folder)
        except FileNotFoundError:
            print('Folder "' + self.data_folder + ' not found')
            return False

        self.images_to_process.sort()
        self.images_to_process = list(filter(lambda file: self.image_ext in file, self.images_to_process))
        if self.is_training is True:
            columns = ['image'] + self.labels
        else:
            columns = ['image']

        # try:
        #     self.processed = pd.read_csv(self.save_path, header=None, names=columns)
        #     self.images_processed = len(self.processed)
        # except FileNotFoundError:
        #     self.images_processed = 0

        self.processed = pd.DataFrame(columns=columns)

        return True

    def set_data_folder(self, data_folder):
        self.data_folder = data_folder

    def resize_image(self, image):
        return cv2.resize(image, (self.image_shape, self.image_shape))

    def status(self):
        return {
            'image_to_process': len(self.images_to_process),
            'image_processed': self.images_processed,
            'percentage': (100 * self.images_processed) / len(self.images_to_process),
            'next': self.get_next()
        }

    def get_next(self):
        if self.images_processed >= len(self.images_to_process):
            return None
        else:
            return self.images_to_process[self.images_processed]

    def set_current_label(self, label):
        self.current_label = label

    def process_next(self):
        next_image_name = self.get_next()
        if next_image_name is None or self.current_label is None:
            return False
        else:
            # load image
            next_image = load_image(self.data_folder + '/' + next_image_name)
            resized = self.resize_image(next_image)

            if self.is_training:
                # Get label from image name
                label = get_image_label(self.current_label, self.labels)
                self.add_to_processed(resized, label)
            else:
                self.add_to_processed(resized)

            # update image processed
            self.images_processed += 1

            return True

    def add_to_processed(self, image, label=None):
        if label is None:
            self.processed.loc[self.images_processed] = [image]
        else:
            self.processed.loc[self.images_processed] = [image] + label

    def save_results(self):
        self.processed.to_csv(index=False, mode='a', path_or_buf=self.save_path, header=False)
