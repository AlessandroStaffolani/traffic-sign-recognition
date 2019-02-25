from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator


class DataGenerator:

    def __init__(self, data_dir, image_shape=(46, 46), batch_size=32, preprocessing_function=None):
        self.data_dir = data_dir
        self.image_shape = image_shape
        self.batch_size = batch_size
        self.preprocessing_function = preprocessing_function

        if K.image_data_format() == 'channels_first':
            self.input_shape = (1, self.image_shape[0], self.image_shape[1])
        else:
            self.input_shape = (self.image_shape[0], self.image_shape[1], 1)

        self.generator = None

    def create(self, featurewise_center=False, samplewise_center=False, featurewise_std_normalization=False,
               samplewise_std_normalization=False):

        self.generator = ImageDataGenerator(featurewise_center=featurewise_center,
                                            samplewise_center=samplewise_center,
                                            featurewise_std_normalization=featurewise_std_normalization,
                                            samplewise_std_normalization=samplewise_std_normalization,
                                            preprocessing_function=self.preprocessing_function
                                            )

    def get_generator(self, color_mode='grayscale', shuffle=True, seed=42, class_mode='categorical'):
        if self.generator is None:
            self.create()

        return self.generator.flow_from_directory(
            self.data_dir,
            target_size=self.image_shape,
            batch_size=self.batch_size,
            color_mode=color_mode,
            shuffle=shuffle,
            seed=seed,
            class_mode=class_mode
        )
