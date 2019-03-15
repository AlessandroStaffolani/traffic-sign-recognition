import logging
from time import gmtime, strftime

from src.models.Model import Model

from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPool2D, Dropout
from keras.optimizers import SGD
from keras.applications.xception import Xception as XceptionModel
from keras.callbacks import CSVLogger, ModelCheckpoint, LearningRateScheduler, TensorBoard
from src.models.callbacks.LogCallback import LogCallback


callback_table = {
    'LogCallback': {
        'class': LogCallback,
        'args': {
            'log_file': 'log/model.log',
            'level': logging.INFO,
            'format': '%(levelname)s: %(asctime)s: %(message)s'
        }
    },
    'CSVLogger': {
        'class': CSVLogger,
        'args': {
            'filename': 'log/last_history.csv',
            'separator': ',',
            'append': False
        }
    },
    'ModelCheckpoint': {
        'class': ModelCheckpoint,
        'args': {
            'filepath': 'model/checkpoints/weights-{epoch:02d}-{val_loss:.2f}.hdf5',
            'monitor': 'val_loss',
            'verbose': 0,
            'save_best_only': True,
            'save_weights_only': True,
            'mode': 'auto',
            'period': 1
        }
    },
}


class Xception(Model):

    def __init__(self, name='Xception Model', auto_save=True, layer_activation='relu', num_output=43,
                 output_activation='softmax', kernel_size=3, input_shape=(72, 72, 1), lr=0.01):

        self.lr = lr
        if input_shape[0] < 71 or input_shape[1] < 71:
            input_shape = (71, 71, input_shape[2])

        Model.__init__(self, name, auto_save, layer_activation, num_output, output_activation, kernel_size, input_shape)

    def create_model(self):
        self.model = XceptionModel(include_top=True, weights=None, input_shape=self.input_shape,
                                   pooling='max', classes=self.num_output)
