import sys
import cv2
import numpy as np
import pandas as pd
import keras

from src.controllers.MenuController import MenuController


def welcome_msg(random_seed):
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
    print("Pandas: " + str(pd.__version__))
    print("Keras: " + str(keras.__version__), end='\n\n')
    print("To allow replication of the result will be used a specific random seed = " + str(random_seed))
    print("Random seed can be changed passing another number as parameter of the program (python main.py <random_seed>)")


def main(argv):
    try:
        random_seed = int(argv[1])
    except IndexError:
        random_seed = 42

    welcome_msg(random_seed)

    menu = MenuController('data/training/images', 'data/testing/images')

    # df = pd.read_csv('data/training/training_table.csv')
    # print(df.info())


if __name__ == '__main__':
    main(sys.argv)
