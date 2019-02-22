from time import time

from src.utility.dataset_utility import get_labels
from src.utility.file_utility import get_directory_files, remove_file
from src.utility.preprocessor_utility import preprocess_image
from src.utility.dataset_utility import create_traing_data_table, split_train_data

from src.preprocessors.Resizer import Resizer
from src.preprocessors.GrayScale import GrayScale
from src.Pipeline import Pipeline
from src.Dataset import Dataset
from src.Cnn import Cnn


class MenuController:

    def __init__(self, image_folder_training, image_folder_testing, labels_count=43, batch_size=1000, epochs=10, image_shape=46, log_folder='log'):
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

            data_table_path = ask_param_with_default('Training images data table file', 'data/training/training_table.csv')
            batch_size = ask_param_with_default('Batch size to use for training', self.batch_size)
            epochs = ask_param_with_default('Number of epochs for training', self.epochs)
            image_shape = ask_param_with_default('Dimension of all images, must be the same vertically and horizontally', self.image_shape)

            self.train_all_images(data_table_path, int(batch_size), int(epochs), int(image_shape))
        elif self.current_action == 4:
            # save model
            model_out = ask_param_with_default('Where do you want to save the model', 'model/model.json')
            self.model.save_json_model(model_out)

        elif self.current_action == 5:
            # load model
            model_path = ask_param_with_default('Location of the saved model', 'model/model.json')
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

    def train_all_images(self, data_table_path, batch_size, epochs, image_shape):
        start = time()
        pipeline = Pipeline(verbose=False)

        pipeline.add_preprocessors([
            Resizer('Image resize', image_shape),
            GrayScale('Image to gray scale')
        ])

        dataset = Dataset(data_table_path, pipeline, self.labels)

        dataframe_generator = dataset.get_images_generator()

        if self.model_created is False:
            self.model = Cnn(input_shape=(image_shape, image_shape, 1))
            self.model.create_model()
            self.model.compile()

        self.model.fit_generator(dataframe_generator, batch_size, epochs)

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