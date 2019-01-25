import cv2
import numpy as np


class Preprocessor:

    def __init__(self, image_shape=46):
        """
        Image shape should be a squared image
        :param image_shape:
        """
        self.image_shape = image_shape

    def resize_image(self, image):
        return cv2.resize(image, (self.image_shape, self.image_shape))
