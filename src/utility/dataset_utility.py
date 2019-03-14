import pandas as pd
from src.utility.file_utility import get_directory_files, create_directory, copy_file
from src.utility.system_utility import progress_bar
from src.utility.image_utility import load_image, crop_roi, save_image
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

    datatable = pd.DataFrame(columns=['image_path', 'label', 'roi_x1', 'roi_y1', 'roi_x2', 'roi_y2'])
    total_count = 0

    for label in directories:
        current_directory = label
        path_label_folder = folder_path + '/' + current_directory

        images = [image for image in get_directory_files(path_label_folder) if img_ext in image]
        images.sort()

        category_df = pd.read_csv(path_label_folder + '/GT-' + current_directory + '.csv', sep=';')

        count = 0
        for img in images:
            img_path = path_label_folder + '/' + img
            category_df_row = category_df.iloc[count]

            datatable.loc[total_count] = [img_path, label, category_df_row['Roi.X1'], category_df_row['Roi.Y1'],
                                          category_df_row['Roi.X2'], category_df_row['Roi.Y2']]
            count += 1
            total_count += 1

            progress_bar(count, len(images), 'Processing label: ' + label + ' with ' + str(len(images)) + ' images')

        print()

    datatable.to_csv(output_path, index=False, header=True)


def split_train_data(train_out_folder, validation_out_folder, dataset_path, validation_size=0.25, labels=43, roi_folder_suffix='_roi'):
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

    # Simply move images
    copy_images(x_train, y_train, train_out_folder)
    print()
    copy_images(x_valid, y_valid, validation_out_folder)

    # Save images only ROI
    save_images_roi(x_train, y_train, train_out_folder + roi_folder_suffix, dataframe)
    print()
    save_images_roi(x_valid, y_valid, validation_out_folder + roi_folder_suffix, dataframe)


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


def prepare_test_data(starting_folder, output_folder, data_frame_path, sep=';', label_col='ClassId', labels=43, roi_folder_suffix='_roi'):
    files = get_directory_files(starting_folder)
    files.sort()

    data_frame = pd.read_csv(data_frame_path, sep=sep)

    for i in range(labels):
        if i < 10:
            folder = '0000' + str(i)
        else:
            folder = '000' + str(i)
        create_directory(output_folder + '/' + folder)
        create_directory(output_folder + roi_folder_suffix + '/' + folder)

    for i in range(data_frame.shape[0]):
        label = data_frame.iloc[i]
        label = label[label_col]

        if label < 10:
            folder = '0000' + str(label)
        else:
            folder = '000' + str(label)
        image_name = files[i]

        # Simply move images
        copy_file(starting_folder + '/' + image_name, output_folder + '/' + folder + '/' + image_name)

        # Save images only ROI
        roi = data_frame.iloc[i, 3: 7]
        image = load_image(starting_folder + '/' + image_name)

        roi_image = crop_roi(image, roi)
        save_image(output_folder + roi_folder_suffix + '/' + folder + '/' + image_name, roi_image)

        progress_bar(i, data_frame.shape[0], 'Copying ' + str(data_frame.shape[0]) + ' images in: ' + output_folder)

    print()


def save_images_roi(x, y, output_path, dataframe):
    for i in range(x.shape[0]):
        label = y[i]
        if label < 10:
            folder = '0000' + str(label)
        else:
            folder = '000' + str(label)

        create_directory(output_path + '/' + folder)
        file_name = x[i].split('/')[-1]

        image_row = dataframe.loc[dataframe['image_path'] == x[i]]
        image = load_image(x[i])

        roi_image = crop_roi(image, image_row.iloc[0, 2:].values)
        save_image(output_path + '/' + folder + '/' + file_name, roi_image)

        progress_bar(i, x.shape[0], 'Writing ' + str(x.shape[0]) + ' ROI images in: ' + output_path)