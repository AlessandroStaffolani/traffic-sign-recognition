import numpy as np
from src.preprocessors.Preprocessor import Preprocessor
from src.utility.image_utility import equalize_img


class HistogramEqualizer(Preprocessor):

    def __init__(self, titile='Histogram Equalizer'):
        Preprocessor.__init__(self, titile)

    def evaluate(self, value):
        try:
            if value.shape[2] == 1:
                image = np.squeeze(value, axis=2)
                equalized = equalize_img(image.astype(np.uint8))
                return np.array(equalized)[:, :, np.newaxis]
            else:
                return value
        except IndexError:
            return value
