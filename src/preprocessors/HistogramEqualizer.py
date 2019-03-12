import numpy as np
from src.preprocessors.Preprocessor import Preprocessor
from src.utility.image_utility import equalize_img, equalize_image_lightness, equalize_colored_image


class HistogramEqualizer(Preprocessor):

    def __init__(self, titile='Histogram Equalizer', mode=0):
        Preprocessor.__init__(self, titile)
        self.mode = mode  # 0 = equalize_image_lightness | 1 = equalize_colored_image valid only if image has 3 channels

    def evaluate(self, value):
        try:
            if value.shape[2] == 1:
                image = np.squeeze(value, axis=2)
                equalized = equalize_img(image.astype(np.uint8))
                return np.array(equalized)[:, :, np.newaxis]
            elif value.shape[2] == 3:
                if self.mode == 0:
                    equalized = equalize_image_lightness(value.astype(np.uint8))
                else:
                    equalized = equalize_colored_image(value.astype(np.uint8))
                return equalized
            else:
                return value
        except IndexError:
            return value
