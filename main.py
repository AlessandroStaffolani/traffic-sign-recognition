import cv2
import numpy as np
import pandas as pd
import sys

from src.controllers.MenuController import MenuController


def welcome_msg():
    print("***************************************")
    print("******* COMPUTER VISION PROJECT *******")
    print("***************************************", end='\n\n\n')
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
    print("Pandas: " + str(pd.__version__), end='\n\n')


def main(argv):
    welcome_msg()

    menu = MenuController('data/train_data_processed/train_46x46.csv', 'data/training/images')

    # preprocessor = Preprocessor('data/training/images/00000', 'data/train_data_processed/train_46x46.csv',
    #                             labels=labels)
    #
    # preprocessor.set_current_label('00000')
    #
    # while preprocessor.process_next():
    #     status = preprocessor.status()
    #     print('Image processed: ' + str(status['image_processed']) + '/' + str(status['image_to_process']))
    #
    # preprocessor.save_results()


if __name__ == '__main__':
    main(sys.argv)
