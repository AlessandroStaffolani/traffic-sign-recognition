import pandas as pd
from src.utility.file_utility import get_directory_files
from src.utility.system_utility import progress_bar
from sklearn.model_selection import train_test_split


def get_labels(n_labels, as_string=True):
    if as_string:
        return ['0000' + str(i) if i < 10 else '000' + str(i) for i in range(n_labels)]
    else:
        return [int(i) for i in range(n_labels)]


def get_image_label(label_code, labels):
    return [1 if label_code == i else 0 for i in labels]


def create_traing_data_table(folder_path, output_path, img_ext='ppm'):
    directories = get_directory_files(folder_path)
    directories.sort()

    datatable = pd.DataFrame(columns=['image_path', 'label'])
    total_count = 0

    for label in directories:
        current_directory = label
        path_label_folder = folder_path + '/' + current_directory

        images = [image for image in get_directory_files(path_label_folder) if img_ext in image]
        images.sort()

        count = 0
        for img in images:
            img_path = path_label_folder + '/' + img

            datatable.loc[total_count] = [img_path, label]
            count += 1
            total_count += 1

            progress_bar(count, len(images), 'Processing label: ' + label + ' with ' + str(len(images)) + ' images')

        print()

    datatable.to_csv(output_path, index=False, header=True)


def create_validation_set(train_out_folder, validation_out_folder, dataset_path):
    dataframe = pd.read_csv(dataset_path)

    x_train, x_valid, y_train, y_valid = train_test_split(dataframe['image_path'].values, dataframe['label'].values,
                                                          test_size=0.20, shuffle=True)

    print(x_train.shape)
    print(x_valid.shape)
    print(y_train.shape)
    print(y_valid.shape)
