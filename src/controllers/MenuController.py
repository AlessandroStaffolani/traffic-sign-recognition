from time import time
from src.Preprocessor import Preprocessor, PreprocessorTest
from src.utility.system_utility import progress_bar
from src.utility.dataset_utility import get_labels
from src.utility.file_utility import get_directory_files, remove_file


class MenuController:

    def __init__(self, export_file_training, export_file_testing, image_folder_training, image_folder_testing,
                 labels_count=43):
        self.labels = get_labels(labels_count)
        self.export_file_training = export_file_training
        self.export_file_testing = export_file_testing
        self.image_folder_training = image_folder_training
        self.image_folder_testing = image_folder_testing

        self.current_action = 0  # Action selected by user on the menu

        self.preprocessor = Preprocessor(self.export_file_training, labels=self.labels)
        self.preprocessorTest = PreprocessorTest(self.export_file_testing)

        self._init()

    def _init(self):
        while self.current_action != 9:
            print_menu()
            self.current_action = int(input())
            self.handle_menu_action()

    def handle_menu_action(self):
        if self.current_action == 1:
            # One label
            label = input('Insert the label to process\n')
            self.preprocess_one_label(self.image_folder_training + '/' + label, label)

        elif self.current_action == 2:
            # All label
            folder = input('Insert the folder path (' + self.image_folder_training + '):\n')
            if folder != '':
                self.image_folder_training = folder

            self.preprocess_all_images()
        elif self.current_action == 3:
            # All test images
            folder = input('Insert the folder path (' + self.image_folder_testing + '):\n')
            if folder != '':
                self.image_folder_testing = folder

            self.preprocess_all_test_images()

        elif self.current_action == 8:
            # clean log and train
            self.clean_log_and_train_folder()

        elif self.current_action == 9:
            # Exit
            print("Goodbye")
        else:
            # No action possible
            print("Possible actions are 1, 2, 9")

    def preprocess_one_label(self, label_path, label_name):
        self.preprocessor.clean()
        self.preprocessor.set_data_folder(label_path)
        if not self.preprocessor.init():
            return

        self.preprocessor.set_current_label(label_name)

        while self.preprocessor.process_next():
            status = self.preprocessor.status()
            progress_bar(status['image_processed'], status['image_to_process'],
                         'Images processed with label ' + label_name)

        self.preprocessor.save_results()

    def preprocess_all_images(self):
        start = time()
        label_names = get_directory_files(self.image_folder_training)
        label_names.sort()
        for label in label_names:
            self.preprocess_one_label(self.image_folder_training + '/' + label, label)
            print('\n')

        end = time()
        print('The preprocessor has processed ' + str(self.preprocessor.total_images_processed) + ' images')
        print('The execution time was ' + str(end - start) + ' seconds')

    def preprocess_all_test_images(self):
        start = time()
        self.preprocessorTest.clean()
        self.preprocessorTest.set_data_folder(self.image_folder_testing)
        if not self.preprocessorTest.init():
            return

        while self.preprocessorTest.process_next():
            status = self.preprocessorTest.status()
            progress_bar(status['image_processed'], status['image_to_process'],
                         'Images processed')

        self.preprocessorTest.save_results()

        end = time()
        print('\nThe preprocessor has processed ' + str(self.preprocessorTest.total_images_processed) + ' images')
        print('The execution time was ' + str(end - start) + ' seconds')

    def clean_log_and_train_folder(self):
        log_file = input('Log file (log/preprocessor.log)\n')
        if log_file == '':
            log_file = 'log/preprocessor.log'
        train_file = input('Training output file (data/train_data_processed/train_46x46.csv)\n')
        if train_file == '':
            train_file = 'data/train_data_processed/train_46x46.csv'
        if remove_file(log_file) and remove_file(train_file):
            print('Clean completed')


def print_menu():
    print("\n\nPossible actions: (select the action)", end='\n\n')
    print('1) Process one class training images')
    print('2) Process all training images')
    print('3) Process all test images')
    print('\n-- System --------------')
    print('8) Clean preprocessing log and output folder')
    print('9) Exit', end='\n\n')
