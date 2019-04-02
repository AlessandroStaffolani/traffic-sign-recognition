import sys
import cv2
import numpy as np
import pandas as pd
import keras
import argparse


def progress_bar(value, endvalue, title, bar_length=40):
    percent = float(value) / endvalue
    arrow = '=' * int(round(percent * bar_length) - 1) + '>'
    spaces = ' ' * (bar_length - len(arrow))

    sys.stdout.write("\r{0}: [{1}] {2}%".format(title, arrow + spaces, int(round(percent * 100))))
    sys.stdout.flush()


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


def get_arguments():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description='''\
        TRAFFIC SIGN RECOGNITION
        The aim of this project is to use a Convolutional Neural Network (CNN) to classify the famous german traffic sign benchmark data set
        PROJECT GOALS:
            1) Classify the dataset using CNN (images contains only the ROI)
            2) Find traffic signs from a real scene and than send sign image to the first point (OPTIONAL)
        ''',
        epilog='''\
        Possible actions:
            0) Download training and testing datasets
            1) Prepare training datatable
            2) Split training data
            3) Prepare test data
            4) Train all images
            5) Load existing model from json
            6) Test model performance 
        Possibile models:
            0) Simple Model
            1) SGD Model
            2) Xception Model
            3) NASNet Mobile Model
    ''')
    parser.add_argument_group()
    parser.add_argument('-a', '--actions', help='set the action or the list of action to execute, check the list below',
                        type=int, nargs='*',
                        choices=[0, 1, 2, 3, 4, 5, 6])
    parser.add_argument('--batch-size', help='size of the batch (default: 32)', default=32, type=int)
    parser.add_argument('--color-mode', help='set the color mode used (default: grayscale)', default='grayscale',
                        type=str, choices=['grayscale', 'rgb'])
    parser.add_argument('-e', '--epochs', help='number of epochs for training (default: 30)', default=30, type=int)
    parser.add_argument('--image-shape',
                        help='final size of all images after preprocessing (default: 46) images will be squared',
                        default=46, type=int)
    parser.add_argument('-m', '--mode',
                        help='set the project mode, if mode = 1 action argument must be setted (default: 0)',
                        type=int, default=0, choices=[0, 1])

    parser.add_argument('--model-code', help='set the model to be used, check the list below (default: 0)', type=int,
                        choices=[0, 1, 2, 3], default=0)
    parser.add_argument('--model-file',
                        help='path to the model json file that you want load (default: model/model.json)',
                        default='model/simple-model.json', type=str)
    parser.add_argument('--n-train-samples', help='number of image in train dataset (defualt: 31367)', default=31367,
                        type=int)
    parser.add_argument('--n-validation-samples', help='number of image in validation dataset (defualt: 7842)',
                        default=7842,
                        type=int)
    parser.add_argument('--n-workers', help='number of workers to use to fit (default: 1)', default=1, type=int)
    parser.add_argument('-r', '--random-seed', help='set the project random seed (default: 42)', default=42, type=int)
    parser.add_argument('--run-file', help="use a json file containing the configs to run multiple instances",
                        default=None, type=str)
    parser.add_argument('--split-factor', help='percentage of training data to use as validation data (default: 0.2)',
                        default=0.2, type=float)
    parser.add_argument('--test-dir', help='Location of test images (default: data/test)', default='data/test',
                        type=str)
    parser.add_argument('--train-dir', help='Location of train images (default: data/train)', default='data/train',
                        type=str)
    parser.add_argument('--use-augmentation', help='Use augmentation on image dataset (default: 0)', default=0,
                        type=int, choices=[0, 1])
    parser.add_argument('--validation-dir', help='Location of validation images (default: data/validation)',
                        default='data/validation',
                        type=str)
    parser.add_argument('--weights-file',
                        help='path to the weights h5 file that you want load (default: model/weights/weights.h5)',
                        default='model/weights/weights.h5', type=str)

    args = parser.parse_args()

    if args.mode == 1 and args.actions is None:
        parser.error('--action is required when mode = 1')

    return args
