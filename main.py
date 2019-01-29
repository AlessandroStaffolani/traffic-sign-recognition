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

    menu = MenuController('data/train_data_processed/train_46x46.csv', 'data/test_data_processed/test_46x46.csv',
                          'data/training/images', 'data/testing/images')

    # df = pd.read_csv('data/test_data_processed/test_46x46.csv')
    # print(df.info())


if __name__ == '__main__':
    main(sys.argv)
