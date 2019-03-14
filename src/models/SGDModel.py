import logging
from time import gmtime, strftime

from src.models.Model import Model

from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPool2D, Dropout
from keras.optimizers import SGD
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


class SGDModel(Model):

    def __init__(self, name='SGD Model', auto_save=True, layer_activation='relu', num_output=43,
                 output_activation='softmax', kernel_size=3, input_shape=(46, 46), lr=0.01):

        self.lr = lr

        Model.__init__(self, name, auto_save, layer_activation, num_output, output_activation, kernel_size, input_shape)

    def create_model(self):
        model = Sequential()

        model.add(Conv2D(32, (3, 3), padding='same',
                         input_shape=self.input_shape,
                         activation=self.layers_activation))
        model.add(Conv2D(32, (3, 3), activation=self.layers_activation))
        model.add(MaxPool2D(pool_size=(2, 2)))
        model.add(Dropout(0.2))

        model.add(Conv2D(64, (3, 3), padding='same',
                         activation=self.layers_activation))
        model.add(Conv2D(64, (3, 3), activation=self.layers_activation))
        model.add(MaxPool2D(pool_size=(2, 2)))
        model.add(Dropout(0.2))

        model.add(Conv2D(128, (3, 3), padding='same',
                         activation=self.layers_activation))
        model.add(Conv2D(128, (3, 3), activation=self.layers_activation))
        model.add(MaxPool2D(pool_size=(2, 2)))
        model.add(Dropout(0.2))

        model.add(Flatten())
        model.add(Dense(512, activation=self.layers_activation))
        model.add(Dropout(0.5))
        model.add(Dense(self.num_output, activation=self.output_activation))

        self.model = model
        return self.model
