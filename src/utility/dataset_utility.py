import pandas as pd
from src.utility.file_utility import get_directory_files, create_directory, copy_file
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


def split_train_data(train_out_folder, validation_out_folder, dataset_path, validation_size=0.2, labels=43):
    dataframe = pd.read_csv(dataset_path)

    x_train, x_valid, y_train, y_valid = train_test_split(dataframe['image_path'].values, dataframe['label'].values,
                                                          test_size=validation_size, shuffle=True)

    for i in range(labels):
        if i < 10:
            folder = '0000' + str(i)
        else:
            folder = '000' + str(i)
        create_directory(train_out_folder + '/' + folder)
        create_directory(validation_out_folder + '/' + folder)

    copy_images(x_train, y_train, train_out_folder)
    print()
    copy_images(x_valid, y_valid, validation_out_folder)


def copy_images(x, y, output_path):
    for i in range(x.shape[0]):
        label = y[i]
        if label < 10:
            folder = '0000' + str(label)
        else:
            folder = '000' + str(label)
        file_name = x[i].split('/')[-1]
        copy_file(x[i], output_path + '/' + folder + '/' + file_name)
        progress_bar(i, x.shape[0], 'Copying ' + str(x.shape[0]) + ' images in: ' + output_path)


def prepare_test_data(starting_folder, output_folder, data_frame_path, sep=';', label_col='ClassId', labels=43):
    files = get_directory_files(starting_folder)
    files.sort()

    data_frame = pd.read_csv(data_frame_path, sep=sep)

    for i in range(labels):
        if i < 10:
            folder = '0000' + str(i)
        else:
            folder = '000' + str(i)
        create_directory(output_folder + '/' + folder)

    for i in range(data_frame.shape[0]):
        label = data_frame.iloc[i]
        label = label[label_col]

        if label < 10:
            folder = '0000' + str(label)
        else:
            folder = '000' + str(label)
        image_name = files[i]

        copy_file(starting_folder + '/' + image_name, output_folder + '/' + folder + '/' + image_name)
        progress_bar(i, data_frame.shape[0], 'Copying ' + str(data_frame.shape[0]) + ' images in: ' + output_folder)

    print()
