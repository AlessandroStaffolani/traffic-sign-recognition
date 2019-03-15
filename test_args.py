import argparse


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
    ''')
    parser.add_argument_group()
    parser.add_argument('-a', '--action', help='set the action or the list of action to execute, check the list below',
                        type=str, nargs='*',
                        choices=['0', '1', '2', '3', '4', '5', '6'])
    parser.add_argument('--color-mode', help='set the color mode used (default: grayscale)', default='grayscale',
                        type=str, choices=['grayscale', 'rgb'])
    parser.add_argument('-e', '--epochs', help='number of epochs for training (default: 30)', default=30, type=int)
    parser.add_argument('--image-shape',
                        help='final size of all images after preprocessing (default: 46) images will be squared',
                        default=46, type=int)
    parser.add_argument('-m', '--mode',
                        help='set the project mode, if mode = 1 action argument must be setted (default: 0)',
                        type=int, default=0, choices=[0, 1])

    parser.add_argument('--model-code', help='set the model to be used, check the list below (default: 0)', type=str,
                        choices=['0', '1', '2'], default='0')
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
    parser.add_argument('-s', '--size-batch', help='size of the batch (default: 32)', default=32, type=int)
    parser.add_argument('--split-factor', help='percentage of training data to use as validation data (default: 0.2)',
                        default=0.2, type=float)
    parser.add_argument('--weights-file',
                        help='path to the weights h5 file that you want load (default: model/weights/weights.h5)',
                        default='model/weights/weights.h5', type=str)

    args = parser.parse_args()

    if args.mode == 1 and args.action is None:
        parser.error('--action is required when mode = 1')

    return args


def main(args):
    print(args)


if __name__ == '__main__':
    args = get_arguments()
    main(args)
