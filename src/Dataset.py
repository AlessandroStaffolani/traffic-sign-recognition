import pandas as pd
import numpy as np
import logging

from src.utility.file_utility import get_directory_files
from src.utility.image_utility import load_image
from src.utility.dataset_utility import get_image_label

logging.basicConfig(filename='log/dataset.log', level=logging.INFO,
                    format='%(levelname)s: %(asctime)s: %(message)s')


class Dataset:

    def __init__(self, folder_path, image_preprocessing, image_shape=46, chunk_size=1000, labels=None, image_extension='ppm'):
        self._folder_path = folder_path
        self.image_preprocessing = image_preprocessing
        self.image_shape = int(image_shape)
        self._chunk_size = int(chunk_size)
        self.image_extension = image_extension
        self.labels = labels

    def get_images(self):
        directories = get_directory_files(self._folder_path)
        directories.sort()

        total_count = 0

        # Iterate through image folders
        for directory in directories:
            current_directory = directory
            path_label_folder = self._folder_path + '/' + current_directory

            images = [image for image in get_directory_files(path_label_folder) if
                      self.image_extension in image]

            logging.info("Iterating through label folder: " + str(directory) + " with " + str(len(images)) + " images")

            count = 0
            image_processed = 0

            # While all images in folder are processed
            while count < len(images):
                # get the real chunk size based on the remaing images to process
                if len(images) - image_processed > self._chunk_size:
                    chunk = self._chunk_size
                else:
                    chunk = len(images) - image_processed

                img_array = list()
                labels_array = list()

                # for each image until the chunk is finished
                for i in range(0, chunk):
                    image = load_image(path_label_folder + '/' + images[image_processed + i])
                    # Call the preprocessing function over the loaded image
                    preprocessed = self.image_preprocessing(image, (self.image_shape, self.image_shape))
                    # Get image label
                    labels = get_image_label(current_directory, self.labels)

                    # append image and label to dataframe
                    img_array.append(preprocessed)
                    labels_array.append(labels)

                # Update indexes
                count += chunk
                image_processed += chunk
                total_count += chunk

                logging.info("Processed a chunk of " + str(chunk) + " images")
                logging.info("Total images processed until now: " + str(total_count))

                # Yield images and labels
                np_img_array = np.array(img_array)
                np_label_array = np.array(labels_array)
                yield np_img_array.astype(np.uint8), np_label_array.astype(np.uint8)

    def set_chunk_size(self, chunk):
        self._chunk_size = chunk