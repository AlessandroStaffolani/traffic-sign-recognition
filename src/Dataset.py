import pandas as pd
import numpy as np
from sklearn.utils import shuffle
import logging

from src.utility.file_utility import get_directory_files
from src.utility.image_utility import load_image
from src.utility.dataset_utility import get_image_label

logging.basicConfig(filename='log/dataset.log', level=logging.INFO,
                    format='%(levelname)s: %(asctime)s: %(message)s')


class Dataset:

    def __init__(self, data_path, pipeline, labels=None):
        self.data_path = data_path
        self.pipeline = pipeline
        self.labels = labels

    def get_images_generator(self, batch_size=32, input_shape=(32, 46, 46, 1), n_sample=None):
        table_data = pd.read_csv(self.data_path, nrows=n_sample)
        table_data = shuffle(table_data)  # implement (sliced) random permutations.

        while True:

            X = np.empty(input_shape)
            y = np.empty(batch_size, dtype=int)

            for index, row in table_data.iterrows():
                image_path = row['image_path']
                label = row['label']

                image = load_image(image_path)
                # Call the preprocessing function over the loaded image
                preprocessed = self.pipeline.evaluate(image)
                # Get image label in the format necessary for the CNN
                labels = get_image_label(label, self.labels)

                # Yield images and labels
                yield (np.array(preprocessed).astype(np.uint8), np.array(labels).astype(np.uint8))

    def get_images_generator_from_folders(self):
        directories = get_directory_files(self.data_path)
        directories.sort()

        while True:
            total_count = 0

            # Iterate through image folders
            for directory in directories:
                current_directory = directory
                path_label_folder = self.data_path + '/' + current_directory

                images = [image for image in get_directory_files(path_label_folder) if
                          'ppm' in image]

                logging.info("Iterating through label folder: " + str(current_directory) + " with " + str(len(images)) + " images")

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
                        preprocessed = self.pipeline(image, (46, 46))
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

    def get_images_subset_from_folder(self, num_element_for_label=5):
        directories = get_directory_files(self.data_path)
        directories.sort()

        while True:

            # Iterate through image folders
            for directory in directories:
                current_directory = directory
                path_label_folder = self.data_path + '/' + current_directory

                images = [image for image in get_directory_files(path_label_folder) if
                          self.image_extension in image]

                for i in range(0, num_element_for_label):

                    img_array = list()
                    labels_array = list()

                    image = load_image(path_label_folder + '/' + images[i])

                    preprocessed = self.pipeline(image, (self.image_shape, self.image_shape))
                    # Get image label
                    labels = get_image_label(current_directory, self.labels)

                    # append image and label to dataframe
                    img_array.append(preprocessed)
                    labels_array.append(labels)

                    # Yield images and labels
                    np_img_array = np.array([img_array])
                    np_label_array = np.array([labels_array])
                    yield np_img_array.astype(np.uint8), np_label_array.astype(np.uint8)

    def get_generators_no_chunk_from_folder(self):
        directories = get_directory_files(self.data_path)
        directories.sort()

        while True:
            # Iterate through image folders
            for directory in directories:
                current_directory = directory
                path_label_folder = self.data_path + '/' + current_directory

                images = [image for image in get_directory_files(path_label_folder) if
                          'ppm' in image]

                for img in images:
                    img_array = list()
                    labels_array = list()

                    image = load_image(path_label_folder + '/' + img)

                    preprocessed = self.pipeline(image, (46, 46))
                    # Get image label
                    labels = get_image_label(current_directory, self.labels)

                    # append image and label to dataframe
                    img_array.append(preprocessed)
                    labels_array.append(labels)

                    # Yield images and labels
                    np_img_array = np.array(img_array)
                    np_label_array = np.array(labels_array)
                    yield np_img_array.astype(np.uint8), np_label_array.astype(np.uint8)

    def set_chunk_size(self, chunk):
        self._chunk_size = chunk