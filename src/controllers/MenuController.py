from time import time

from src.utility.dataset_utility import get_labels
from src.utility.file_utility import get_directory_files, remove_file
from src.utility.preprocessor_utility import preprocess_image
from src.utility.dataset_utility import create_traing_data_table, split_train_data

from src.preprocessors.Resizer import Resizer
from src.preprocessors.GrayScale import GrayScale
from src.Pipeline import Pipeline
from src.Dataset import Dataset
from src.DataGenerator import DataGenerator
from src.Cnn import Cnn


class MenuController:

    def __init__(self, image_folder_training, image_folder_testing, labels_count=43, batch_size=1000, epochs=10,
                 image_shape=46, log_folder='log'):
        self.labels = get_labels(labels_count)
        self.image_folder_training = image_folder_training
        self.image_folder_testing = image_folder_testing
        self.batch_size = batch_size
        self.epochs = epochs
        self.image_shape = image_shape
        self.log_folder = log_folder
        self.model = None
        self.model_created = False

        self.current_action = 0  # Action selected by user on the menu

        self._init()

    def _init(self):
        while self.current_action != 9:
            print_menu()
            self.current_action = int(input())
            self.handle_menu_action()

    def handle_menu_action(self):
        if self.current_action == 1:
            # Prepare datatable
            folder = ask_param_with_default('Training images folder', self.image_folder_training)
            output = ask_param_with_default('Output file path', 'data/training/training_table.csv')
            create_traing_data_table(folder, output)

        elif self.current_action == 2:
            # Split training data

            data_table_path = ask_param_with_default('Training images data table file',
                                                     'data/training/training_table.csv')
            train_out_path = ask_param_with_default('Train data output path', 'data/train')
            validation_out_path = ask_param_with_default('Validation data output path', 'data/validation')
            validation_size = ask_param_with_default('Validation set proportion', 0.2)
            print()
            start = time()
            split_train_data(train_out_path, validation_out_path, data_table_path, validation_size=validation_size)
            end = time()
            print('\nTraing data split time: ' + str(round(end - start, 2)) + ' seconds')

        elif self.current_action == 3:
            # Train all images

            train_dir = ask_param_with_default('Training images data dir', 'data/train')
            n_train_samples = int(ask_param_with_default('Number of training samples ', 31367))
            validation_dir = ask_param_with_default('Validation images data dir', 'data/validation')
            n_valid_samples = int(ask_param_with_default('Number of validation samples ', 7842))
            batch_size = ask_param_with_default('Batch size to use for training', self.batch_size)
            epochs = ask_param_with_default('Number of epochs for training', self.epochs)
            image_shape = ask_param_with_default(
                'Dimension of all images, must be the same vertically and horizontally', self.image_shape)
            workers = int(ask_param_with_default('Number of processes to spin up when using process-based threading', 5))

            self.train_all_images(train_dir, validation_dir, n_train_samples, n_valid_samples, int(batch_size),
                                  int(epochs), int(image_shape), workers)
        elif self.current_action == 4:
            # save model
            model_out = ask_param_with_default('Where do you want to save the model', 'model/model')
            self.model.save_json_model(model_out)

        elif self.current_action == 5:
            # load model
            model_path = ask_param_with_default('Location of the saved model', 'model/model')
            image_shape = ask_param_with_default(
                'Dimension of all images, must be the same vertically and horizontally', self.image_shape)
            self.model = Cnn(input_shape=(image_shape, image_shape, 1))
            self.model.load_json_model(model_path)
            self.model.compile()
            self.model_created = True

        elif self.current_action == 8:
            # Clean log folder
            self.clean_log_folder()
        elif self.current_action == 9:
            # Exit
            print("Goodbye")
        else:
            # No action possible
            print("Possible actions are 1, 2, 3, 4, 8, 9")

    def train_all_images(self, train_dir, validation_dir, n_train_samples, n_valid_sample, batch_size, epochs,
                         image_shape, workers):
        start = time()

        train_generator = DataGenerator(train_dir, n_train_samples, batch_size=batch_size,
                                        image_shape=(image_shape, image_shape))

        validation_generator = DataGenerator(validation_dir, n_valid_sample, batch_size=batch_size,
                                             image_shape=(image_shape, image_shape))

        if self.model_created is False:
            self.model = Cnn(input_shape=(image_shape, image_shape, 1))
            self.model.create_model()
            self.model.compile()
            self.model_created = True

        self.model.fit_generator(
            train_generator.get_generator(),
            steps_per_epoch=n_train_samples // batch_size,
            epochs=epochs,
            validation_data=validation_generator.get_generator(),
            validation_steps=n_valid_sample // batch_size,
            workers=workers
        )

        end = time()
        print('Processing time: ' + str(round(end - start, 2)) + ' seconds')

    def clean_log_folder(self):
        log_files = get_directory_files(self.log_folder)
        print()
        for file in log_files:
            remove_file(self.log_folder + '/' + file)
        print('Log folder clean', end='\n')


def print_menu():
    print("\n\nPossible actions: (select the action)", end='\n\n')
    print('1) Prepare training datatable')
    print('2) Split training data')
    print('3) Train all images')
    print('4) Save trained model')
    print('5) Load existing model from json')
    print('\n-- System --------------')
    print('8) Clean log folder')
    print('9) Exit', end='\n\n')


def ask_param_with_default(question, default, end=':\t'):
    value = input(question + ' (' + str(default) + ')' + end)
    if value == '':
        return default
    else:
        return value
