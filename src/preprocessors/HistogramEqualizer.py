from src.preprocessors.Preprocessor import Preprocessor
from src.utility.image_utility import equalize_img


class HistogramEqualizer(Preprocessor):

    def __init__(self, titile='Histogram Equalizer'):
        Preprocessor.__init__(self, titile)

    def evaluate(self, value):
        return equalize_img(value)
