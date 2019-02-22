from src.preprocessors.Preprocessor import Preprocessor
from src.utility.image_utility import img_to_grayscale


class GrayScale(Preprocessor):

    def __init__(self, title):
        Preprocessor.__init__(self, title)

    def evaluate(self, value):
        return img_to_grayscale(value)
