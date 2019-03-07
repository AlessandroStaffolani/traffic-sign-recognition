from time import time, gmtime, strftime

from src.utility.dataset_utility import get_labels
from src.utility.file_utility import get_directory_files, remove_file
from src.utility.dataset_utility import create_traing_data_table, split_train_data, prepare_test_data

from src.init import init_training_data_folder, init_testing_data_folder, init_testing_id_file, init_directories

from src.preprocessors.HistogramEqualizer import HistogramEqualizer
from src.preprocessors.Normalizer import Normalizer

from src.DataGenerator import DataGenerator
from src.models.Model import Model
from src.models.SGDModel import SGDModel
from src.Pipeline import Pipeline

ACTIONS = {
    '0': 'inti_project',
    '1': 'prepare_datatable',
    '2': 'split_training_data',
    '3': 'prepare_test_data',
    '4': 'train_all_images',
    '5': 'load_model',
    '6': 'evaluate_images',
    '8': 'clean_log_folder',
    '9': 'close'
}

MODELS = {
    '0': Model,
    '1': SGDModel
}


class MenuController:

    def __init__(self, mode=0, actions=None, model='0', labels_count=43, batch_size=400, epochs=10,
                 image_shape=46, num_workers=1, model_path='model/simple_model.json', weights_path='model/weights/weights.h5',
                 split_factor=0.25, n_train_samples=29406, n_validation_samples=9803, log_folder='log/'):
        self.labels = get_labels(labels_count)
        self.mode = mode
        self.actions = actions
        self.model_code = model
        self.batch_size = batch_size
        self.epochs = epochs
        self.image_shape = image_shape
        self.num_workers = num_workers
        self.model_path = model_path
        self.weights_path = weights_path
        self.split_factor = split_factor
        self.n_train_samples = n_train_samples
        self.n_validation_samples = n_validation_samples
        self.log_folder = log_folder
        self.model = None
        self.model_created = False
        self.pipeline = Pipeline()

        self.current_action = 0  # Action selected by user on the menu
        self.close_action = 9

        self._init()

    def _init(self):
        self.pipeline.add_preprocessors((
            HistogramEqualizer(),
            Normalizer()
        ))

        if self.mode == 0:
            # Interactive mode
            while self.current_action != 9:
                print_menu()
                try:
                    self.current_action = int(input())
                except ValueError:
                    self.error_action()
                self.handle_menu_action()
                if self.current_action != self.close_action:
                    self.current_action = 0
        else:
            # script mode
            print('\nSCRIPT MODE:', end='\n\n')
            print('Actions to execute: ' + str(self.actions))
            print('Configuration:')
            print('\tmodel-code:\t\t' + str(self.model_code))
            print('\tbatch-size:\t\t' + str(self.batch_size))
            print('\tepochs:\t\t\t' + str(self.epochs))
            print('\timage-shape:\t\t' + str(self.image_shape))
            print('\tnum-workers:\t\t' + str(self.num_workers))
            print('\tmodel-file:\t\t' + self.model_path)
            print('\tweights-file:\t\t' + self.weights_path)
            print('\tsplit-factor:\t\t' + str(self.split_factor))
            print('\tn-train-samples:\t' + str(self.n_train_samples))
            print('\tn-validation-samples:\t' + str(self.n_validation_samples), end='\n\n')

            self.execute_actions()

    def handle_menu_action(self):
        if self.current_action == 0:
            # Download datasets

            self.inti_project()

        elif self.current_action == 1:
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
            # load model

            self.load_model()

        elif self.current_action == 6:
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

    def inti_project(self):
        if self.mode == 0:
            # Interactive mode
            training_out_path = ask_param_with_default('Where to extract training data', 'data/training')
            testing_out_path = ask_param_with_default('Where to extract testing data', 'data/testing')
        else:
            # Script mode
            training_out_path = 'data/training'
            testing_out_path = 'data/testing'

        init_directories(training_out_path, testing_out_path)
        init_training_data_folder(out_path=training_out_path)
        init_testing_data_folder(out_path=testing_out_path)
        init_testing_id_file(out_path=testing_out_path)

    def prepare_datatable(self):
        if self.mode == 0:
            # Interactive mode
            folder = ask_param_with_default('Training images folder', 'data/training/images')
            output = ask_param_with_default('Output file path', 'data/training/training_table.csv')
        else:
            # Script mode
            folder = 'data/training/images'
            output = 'data/training/training_table.csv'

        print()
        start = time()
        create_traing_data_table(folder, output)
        end = time()
        print('\nDatatable preparation time: ' + str(round(end - start, 2)) + ' seconds')

    def split_training_data(self):
        if self.mode == 0:
            # Intercative mode
            data_table_path = ask_param_with_default('Training images data table file',
                                                     'data/training/training_table.csv')
            train_out_path = ask_param_with_default('Train data output path', 'data/train')
            validation_out_path = ask_param_with_default('Validation data output path', 'data/validation')
            validation_size = ask_param_with_default('Validation set proportion', self.split_factor)
        else:
            # Script mode
            data_table_path = 'data/training/training_table.csv'
            train_out_path = 'data/train'
            validation_out_path = 'data/validation'
            validation_size = self.split_factor

        print()
        start = time()
        split_train_data(train_out_path, validation_out_path, data_table_path, validation_size=validation_size)
        end = time()
        print('\nTraing data split time: ' + str(round(end - start, 2)) + ' seconds')

    def prepare_test_data(self):
        if self.mode == 0:
            # Intercative mode
            data_folder = ask_param_with_default('Folder of test images', 'data/testing/images')
            output_folder = ask_param_with_default('Test folder output', 'data/test')
            data_frame_path = ask_param_with_default('Test data csv path', 'data/testing/testing_table.csv')
            separator = ask_param_with_default('Separator for the data csv file', ';')
            label_col = ask_param_with_default('Name of the label column', 'ClassId')
        else:
            # Script mode
            data_folder = 'data/testing/images'
            output_folder = 'data/test'
            data_frame_path = 'data/testing/testing_table.csv'
            separator = ';'
            label_col = 'ClassId'

        print()
        start = time()
        prepare_test_data(data_folder, output_folder, data_frame_path, sep=separator, label_col=label_col)
        end = time()
        print('\nTime to split test images: ' + str(round(end - start, 2)) + ' seconds')

    def train_all_images(self):
        if self.mode == 0:
            # Intercative mode
            model_code = ask_param_with_default('Model to be used from these: ' + str(MODELS), self.model_code)
            train_dir = ask_param_with_default('Training images data dir', 'data/train')
            n_train_samples = int(ask_param_with_default('Number of training samples ', self.n_train_samples))
            validation_dir = ask_param_with_default('Validation images data dir', 'data/validation')
            n_valid_samples = int(ask_param_with_default('Number of validation samples ', self.n_validation_samples))
            batch_size = int(ask_param_with_default('Batch size to use for training', self.batch_size))
            epochs = int(ask_param_with_default('Number of epochs for training', self.epochs))
            image_shape = int(ask_param_with_default(
                'Dimension of all images, must be the same vertically and horizontally', self.image_shape))
            workers = int(ask_param_with_default('Number of processes to spin up when using process-based threading',
                                                 self.num_workers))
        else:
            # Script mode
            model_code = self.model_code
            train_dir = 'data/train'
            n_train_samples = self.n_train_samples
            validation_dir = 'data/validation'
            n_valid_samples = self.n_validation_samples
            batch_size = self.batch_size
            epochs = self.epochs
            image_shape = self.image_shape
            workers = self.num_workers

        self.model_code = model_code

        start = time()

        train_generator = DataGenerator(train_dir, batch_size=batch_size,
                                        image_shape=(image_shape, image_shape),
                                        preprocessing_function=self.pipeline.evaluate)

        validation_generator = DataGenerator(validation_dir, batch_size=batch_size,
                                             image_shape=(image_shape, image_shape),
                                             preprocessing_function=self.pipeline.evaluate)

        if self.model_created is False:
            self.create_model(image_shape=image_shape)

        if workers > 1:
            use_multiprocessing = True
        else:
            use_multiprocessing = False

        history = self.model.fit_generator(
            train_generator.get_generator(),
            steps_per_epoch=n_train_samples // batch_size,
            epochs=epochs,
            validation_data=validation_generator.get_generator(),
            validation_steps=n_valid_samples // batch_size,
            workers=workers,
            use_multiprocessing=use_multiprocessing
        )

        end = time()
        print('Processing time: ' + str(round(end - start, 2)) + ' seconds', end='\n\n')

        print('Loss history:\t' + str(history.history['loss']))
        print('Accuracy history:\t' + str(history.history['acc']))
        print('Loss validation history:\t' + str(history.history['val_loss']))
        print('Accuracy validation history:\t' + str(history.history['val_acc']))

    def load_model(self):
        if self.mode == 0:
            # Intercative mode
            model_path = ask_param_with_default('Location of the saved model', self.model_path)
            weights_path = ask_param_with_default('Location of the saved weights', self.weights_path)
            image_shape = ask_param_with_default(
                'Dimension of all images, must be the same vertically and horizontally', self.image_shape)
        else:
            # Script mode
            model_path = self.model_path
            weights_path = self.weights_path
            image_shape = self.image_shape

        self.create_model(loaded=True, image_shape=image_shape, model_path=model_path, weights_path=weights_path)

    def evaluate_images(self):
        if self.mode == 0:
            # Intercative mode
            eval_data_folder = ask_param_with_default('Path of the test data', 'data/test')
            n_test_samples = int(ask_param_with_default('Number of test samples', 12630))
            batch_size = int(ask_param_with_default('Batch size to use for testing', self.batch_size))
            image_shape = int(ask_param_with_default(
                'Dimension of all images, must be the same vertically and horizontally', self.image_shape))
        else:
            # Script mode
            eval_data_folder = 'data/test'
            n_test_samples = 12630
            batch_size = self.batch_size
            image_shape = self.image_shape

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
        print("Possible actions are 0, 1, 2, 3, 4, 5, 6, 8, 9")

    def execute_actions(self):
        if self.mode == 1:
            for action_name in self.actions:
                print('Executing action: ' + ACTIONS[str(action_name)])
                try:
                    action = getattr(self, ACTIONS[str(action_name)])
                    action()
                except AttributeError:
                    print('Action not exist')

        else:
            print("Current version 0 -> Interactive, use the menu")

    def create_model(self, loaded=False, image_shape=46, model_path='', weights_path=''):
        model_class = MODELS[self.model_code]
        if loaded:
            self.model = model_class(input_shape=(image_shape, image_shape, 1))
            self.model.load_model(model_path)
            self.model.compile()
            self.model.load_weights(weights_path)
            self.model.init_callbacks()
            self.model_created = True
        else:
            self.model = model_class(input_shape=(image_shape, image_shape, 1))
            self.model.create_model()
            self.model.compile()
            self.model.init_callbacks()
            self.model_created = True


def print_menu():
    print("\n\nPossible actions: (select the action)", end='\n\n')
    print('0) Download training and testing datasets')
    print('1) Prepare training datatable')
    print('2) Split training data')
    print('3) Prepare test data')
    print('4) Train all images')
    print('5) Load existing model from json')
    print('6) Test model performance')
    print('\n-- System --------------')
    print('8) Clean log folder')
    print('9) Exit', end='\n\n')


def ask_param_with_default(question, default, end=':\t'):
    value = input(question + ' (' + str(default) + ')' + end)
    if value == '':
        return default
    else:
        return value
