from time import time

from src.utility.dataset_utility import get_labels
from src.utility.file_utility import get_directory_files, remove_file
from src.utility.preprocessor_utility import preprocess_image
from src.utility.dataset_utility import create_traing_data_table

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
        self.model = Cnn()
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
            # Train all images

            folder = ask_param_with_default('Training images folder', self.image_folder_training)
            batch_size = ask_param_with_default('Batch size to use for training', self.batch_size)
            epochs = ask_param_with_default('Number of epochs for training', self.epochs)
            image_shape = ask_param_with_default('Dimension of all images, must be the same vertically and horizontally', self.image_shape)

            self.train_all_images(folder, batch_size, epochs, image_shape)
        elif self.current_action == 3:
            # save model
            model_out = ask_param_with_default('Where do you want to save the model', 'model/model.json')
            self.model.save_json_model(model_out)

        elif self.current_action == 4:
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
            print("Possible actions are 1, 2, 9")

    def train_all_images(self, folder, batch_size, epochs, image_shape):
        start = time()
        dataset = Dataset(folder, preprocess_image, image_shape, batch_size, self.labels)
        dataframe_generator = dataset.get_images_generator()

        if self.model_created is False:
            self.model.create_model()
            self.model.compile()

        self.model.fit_generator(dataframe_generator, batch_size, epochs)

        # count = 0
        # for row in dataframe_generator:
        #     images, labels = row[0], row[1]
        #     count += images.shape[0]
        #     print(images.shape, end='\t')
        #     print(labels.shape, end='\n\n---------------\n')
        end = time()
        # print('\n\nImages processed: ' + str(count))
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
    print('2) Train all images')
    print('3) Save trained model')
    print('4) Load existing model from json')
    print('\n-- System --------------')
    print('8) Clean log folder')
    print('9) Exit', end='\n\n')


def ask_param_with_default(question, default, end=':\t'):
    value = input(question + ' (' + str(default) + ')' + end)
    if value == '':
        return default
    else:
        return value