from src.preprocessors.Preprocessor import Preprocessor
from src.utility.image_utility import resize_image


class Resizer(Preprocessor):

    def __init__(self, title, size=46):
        Preprocessor.__init__(self, title)
        self.size = int(size)

    def evaluate(self, value):
        return resize_image(value, (self.size, self.size))
