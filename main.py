import sys
import cv2
import numpy as np
import pandas as pd
import keras

from src.init import init_directories


def welcome_msg(random_seed):
    print("****************************************")
    print("******* TRAFFIC SIGN RECOGNITION *******")
    print("****************************************", end='\n\n\n')
    print("Welcome! The aim of this project is to use a Convolutional Neural Network (CNN) to classify the famous "
          "german traffic sign benchmark data set")
    print("_______________________________________", end='\n\n')
    print("PROJECT GOALS:")
    print("1) Classify the dataset using CNN (images contains only the ROI)")
    print("2) Find traffic signs from a real scene and than send sign image to the first point (OPTIONAL) ")
    print("_______________________________________", end='\n\n')
    print("Project main library version:")
    print("OpenCV: " + str(cv2.__version__))
    print("Numpy: " + str(np.__version__))
    print("Pandas: " + str(pd.__version__))
    print("Keras: " + str(keras.__version__), end='\n\n')
    print("To allow replication of the result will be used a specific random seed = " + str(random_seed))
    print("Random seed can be changed for more information run -h or --help command ")


def help_message():
    print('TRAFFIC SIGN RECOGNITION')
    print('python main.py [args]', end='\n\n')
    print('[args]\t\t\t\t\tis optional')
    print('-h or --help\t\t\t\tshows this message')
    print('-r or --random-seed [int_value]\t\tset the project random seed (default: 42)')
    print('-m or --mode [0 or 1]\t\t\tset the project mode 0 = interactive | 1 = script mode '
          'if mode is 1 action argument must be setted (default: 0)')
    print('-a or --action [actions]\t\tset the action or the list of action to execute '
          '(format: <action_1>,<action_2>,<action_3>,...)')
    print('--batch-size [size]\t\t\tsize of the batch (default: 400)')
    print('--epochs [number]\t\t\tnumber of epochs for training (default: 10)')
    print('--image-shape [shape]\t\t\tfinal size of all images after preprocessing '
          '(default: 46) images will be squared')
    print('--n-workers [number]\t\t\tnumber of workers to use to fit (default: 1)')
    print('--model-file [file_path]\t\tpath to the model json file that you want load (default: model/model.json)')
    print('--weights-file [file_path]\t\tpath to the weights h5 file that you want load '
          '(default: model/weights/weights.h5)')
    print('--split-factor [float_number]\t\tpercentage of training data to use as validation data (default: 0.25)')
    print('--n-train-samples [number]\t\tnumber of image in train dataset (defualt: 29406)')
    print('--n-validation-samples [number]\t\tnumber of image in validation dataset (defualt: 9803)', end='\n\n')
    print('Possible actions:')
    print('0) Download training and testing datasets')
    print('1) Prepare training datatable')
    print('2) Split training data')
    print('3) Prepare test data')
    print('4) Train all images')
    print('5) Load existing model from json')
    print('6) Test model performance')


def handle_arguments(args):
    random_seed = 42  # random seeds
    need_helps = False  # shows or not help message
    mode = 0  # mode of execution 0 = interactive | 1 = script mode
    actions = None  # action or list of actions to execute in script mode
    batch_size = 400
    epochs = 10
    image_shape = 46
    n_workers = 1
    model_file = 'model/model.json'
    weights_file = 'model/weights/weights.h5'
    split_factor = 0.25
    n_train_samples = 29406
    n_validation_samples = 9803
    i = 0
    while i < (len(args) - 1):
        arg = args[i]
        arg_val = args[i + 1]
        if arg == '-h' or arg == '--help':
            need_helps = True
        elif arg == '-r' or arg == '--random-seed':
            random_seed = int(arg_val)
        elif arg == '-m' or arg == '--mode':
            mode = int(arg_val)
        elif arg == '-a' or arg == '--action':
            actions = arg_val.split(',')
        elif arg == '--batch-size':
            batch_size = int(arg_val)
        elif arg == '--epochs':
            epochs = int(arg_val)
        elif arg == '--image-shape':
            image_shape = int(arg_val)
        elif arg == '--n-workers':
            n_workers = int(arg_val)
        elif arg == '--model-file':
            model_file = arg_val
        elif arg == '--weights-file':
            weights_file = arg_val
        elif arg == '--split-factor':
            split_factor = float(arg_val)
        elif arg == '--n-train-samples':
            n_train_samples = int(arg_val)
        elif arg == '--n-validation-samples':
            n_validation_samples = int(arg_val)

        i += 2

    if mode == 1 and actions is None:
        need_helps = True

    if len(args) == 1:
        need_helps = True

    return need_helps, random_seed, mode, actions, batch_size, epochs, image_shape, \
           n_workers, model_file, weights_file, split_factor, n_train_samples, n_validation_samples


def main(argv):
    need_helps, random_seed, mode, \
    actions, batch_size, epochs, \
    image_shape, n_workers, model_file, weigths_file, \
    split_factor, n_train_samples, n_validation_samples = handle_arguments(argv[1:])

    if need_helps:
        help_message()
        exit(1)

    try:
        from src.controllers.MenuController import MenuController
    except FileNotFoundError:
        init_directories()
        from src.controllers.MenuController import MenuController

    np.random.seed(random_seed)

    welcome_msg(random_seed)

    menu = MenuController(mode=mode, actions=actions, batch_size=batch_size, epochs=epochs, image_shape=image_shape,
                          num_workers=n_workers, model_path=model_file, weights_path=weigths_file,
                          split_factor=split_factor, n_train_samples=n_train_samples,
                          n_validation_samples=n_validation_samples)

    # df = pd.read_csv('data/training/training_table.csv')
    # print(df.info())


if __name__ == '__main__':
    main(sys.argv)
