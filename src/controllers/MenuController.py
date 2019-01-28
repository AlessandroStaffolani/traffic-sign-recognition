from os import listdir

from src.Preprocessor import Preprocessor
from src.utility.system_utility import progress_bar
from src.utility.dataset_utility import get_labels


class MenuController:

    def __init__(self, export_file, image_folder, labels_count=43):
        self.labels = get_labels(labels_count)
        self.export_file = export_file
        self.image_folder = image_folder
        self.current_action = 0
        self.preprocessor = Preprocessor(self.export_file, labels=self.labels)
        self._init()

    def _init(self):
        while self.current_action != 9:
            print_menu()
            self.current_action = int(input())
            self.handle_menu_action()

    def handle_menu_action(self):
        if self.current_action == 1:
            label = input('Insert the label to process\n')
            self.preprocess_one_label(self.image_folder + '/' + label, label)
        elif self.current_action == 2:
            # One Label
            folder = input('Insert the folder path (' + self.image_folder + '):\n')
            if folder != '':
                self.image_folder = folder

            self.preprocess_all_images()
            # self.preprocess_one_label(self.image_folder + '/' + label, label)
        elif self.current_action == 9:
            print("Goodbye")
            return
        else:
            # No action possible
            return

    def preprocess_one_label(self, label_path, label_name):
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
        label_names = listdir(self.image_folder)
        label_names.sort()
        for label in label_names:
            self.preprocess_one_label(self.image_folder + '/' + label, label)
            print('\n')


def print_menu():
    print("\n\nPossible actions: (select the action)", end='\n\n')
    print('1) Process one class training images', end='\n\n')
    print('2) Process all training images', end='\n\n')
    print('9) Exit', end='\n\n')
