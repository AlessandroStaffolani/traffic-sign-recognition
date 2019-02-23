from time import time, gmtime, strftime

from src.utility.dataset_utility import get_labels
from src.utility.file_utility import get_directory_files, remove_file
from src.utility.dataset_utility import create_traing_data_table, split_train_data, prepare_test_data

from src.DataGenerator import DataGenerator
from src.models.Model import Model


class MenuController:

    def __init__(self, image_folder_training, image_folder_testing, labels_count=43, batch_size=32, epochs=10,
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
        self.close_action = 9

        self._init()

    def _init(self):
        while self.current_action != 9:
            print_menu()
            try:
                self.current_action = int(input())
            except ValueError:
                self.error_action()
            self.handle_menu_action()
            if self.current_action != self.close_action:
                self.current_action = 0

    def handle_menu_action(self):
        if self.current_action == 1:
            # Prepare datatable

            self.prepare_datatable()

        elif self.current_action == 2:
            # Split training data

            self.split_training_data()

        elif self.current_action == 3:
            # Prepare test data

            self.prepare_test_data()

        elif self.current_action == 4:
            # Train all images

            self.train_all_images()

        elif self.current_action == 5:
            # save model

            self.save_model()

        elif self.current_action == 6:
            # load model

            self.load_model()

        elif self.current_action == 7:
            # Evaluate model

            self.evaluate_images()

        elif self.current_action == 8:
            # Clean log folder

            self.clean_log_folder()

        elif self.current_action == self.close_action:
            # Exit

            self.close()

        else:
            # No action possible

            self.error_action()

    def prepare_datatable(self):
        folder = ask_param_with_default('Training images folder', self.image_folder_training)
        output = ask_param_with_default('Output file path', 'data/training/training_table.csv')
        create_traing_data_table(folder, output)

    def split_training_data(self):
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

    def prepare_test_data(self):
        data_folder = ask_param_with_default('Folder of test images', 'data/testing/images')
        output_folder = ask_param_with_default('Test folder output', 'data/test')
        data_frame_path = ask_param_with_default('Test data csv path', 'data/testing/GT-final_test.csv')
        separator = ask_param_with_default('Separator for the data csv file', ';')
        label_col = ask_param_with_default('Name of the label column', 'ClassId')

        print()
        start = time()
        prepare_test_data(data_folder, output_folder, data_frame_path, sep=separator, label_col=label_col)
        end = time()
        print('\nTime to split test images: ' + str(round(end - start, 2)) + ' seconds')

    def train_all_images(self):

        train_dir = ask_param_with_default('Training images data dir', 'data/train')
        n_train_samples = int(ask_param_with_default('Number of training samples ', 31367))
        validation_dir = ask_param_with_default('Validation images data dir', 'data/validation')
        n_valid_sample = int(ask_param_with_default('Number of validation samples ', 7842))
        batch_size = ask_param_with_default('Batch size to use for training', self.batch_size)
        epochs = ask_param_with_default('Number of epochs for training', self.epochs)
        image_shape = ask_param_with_default(
            'Dimension of all images, must be the same vertically and horizontally', self.image_shape)
        workers = int(ask_param_with_default('Number of processes to spin up when using process-based threading', 1))
        save = bool(ask_param_with_default('Save at the end', True))

        start = time()

        train_generator = DataGenerator(train_dir, batch_size=batch_size,
                                        image_shape=(image_shape, image_shape))

        validation_generator = DataGenerator(validation_dir, batch_size=batch_size,
                                             image_shape=(image_shape, image_shape))

        if self.model_created is False:
            self.model = Model(input_shape=(image_shape, image_shape, 1))
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

        if save is True:
            now = strftime("%d-%m-%Y_%H:%M:%S", gmtime())
            self.model.save_model('model/train_' + now)

    def save_model(self):
        model_out = ask_param_with_default('Where do you want to save the model', 'model/model')
        self.model.save_model(model_out)

    def load_model(self):
        model_path = ask_param_with_default('Location of the saved model', 'model/model')
        image_shape = ask_param_with_default(
            'Dimension of all images, must be the same vertically and horizontally', self.image_shape)
        self.model = Model(input_shape=(image_shape, image_shape, 1))
        self.model.load_model(model_path)
        self.model.compile()
        self.model_created = True

    def evaluate_images(self):
        eval_data_folder = ask_param_with_default('Path of the test data', 'data/test')
        n_test_samples = int(ask_param_with_default('Number of test samples', 12630))
        batch_size = int(ask_param_with_default('Batch size to use for testing', self.batch_size))
        image_shape = int(ask_param_with_default(
            'Dimension of all images, must be the same vertically and horizontally', self.image_shape))

        test_generator = DataGenerator(eval_data_folder, batch_size=batch_size, image_shape=(image_shape, image_shape))
        if self.model_created is False:
            print('\nModel not initialized, please load a model')
        else:
            scores = self.model.evaluate_generator(test_generator.get_generator(), n_test_samples // batch_size)
            print()
            print('Loss: ' + str(scores[0]))
            print('Accuracy: ' + str(scores[1]))

    def clean_log_folder(self):
        log_files = get_directory_files(self.log_folder)
        print()
        for file in log_files:
            remove_file(self.log_folder + '/' + file)
        print('Log folder clean', end='\n')

    def close(self):
        print('Goodbye')

    def error_action(self):
        print("Possible actions are 1, 2, 3, 4, 5, 6, 8, 9")


def print_menu():
    print("\n\nPossible actions: (select the action)", end='\n\n')
    print('1) Prepare training datatable')
    print('2) Split training data')
    print('3) Prepare test data')
    print('4) Train all images')
    print('5) Save trained model')
    print('6) Load existing model from json')
    print('7) Test model performance')
    print('\n-- System --------------')
    print('8) Clean log folder')
    print('9) Exit', end='\n\n')


def ask_param_with_default(question, default, end=':\t'):
    value = input(question + ' (' + str(default) + ')' + end)
    if value == '':
        return default
    else:
        return value
