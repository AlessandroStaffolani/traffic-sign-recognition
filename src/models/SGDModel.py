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

    def compile(self, optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy']):
        sgd = SGD(lr=self.lr, decay=1e-6, momentum=0.9, nesterov=True)
        self.model.compile(loss=loss, optimizer=sgd, metrics=metrics)

    def fit(self, train_data, train_labels, validation_split=0.25, epochs=10, batch_size=100):
        if len(self.callbacks) == 0:
            callbacks = None
        else:
            lr = self.lr

            def lr_schedule(epoch):
                return lr * (0.1 ** int(epoch / 10))

            callbacks = self.callbacks
            now = strftime("%d-%m-%Y_%H-%M", gmtime())
            log_file = 'log/tensorboard-' + self.name_to_file() + '-logs-' + str(now)
            callbacks.append(TensorBoard(log_dir=log_file, write_grads=True,
                                         batch_size=batch_size, write_images=True))
            callbacks.append(LearningRateScheduler(lr_schedule))

        if self.auto_save:
            self.auto_save_model()
        history = self.model.fit(train_data, train_labels, validation_split=validation_split, epochs=epochs,
                                 batch_size=batch_size, callbacks=callbacks)
        if self.auto_save:
            self.auto_save_weights(epochs)
        return history

    def fit_generator(self, generator, steps_per_epoch=1000, epochs=10, validation_data=None, validation_steps=None,
                      workers=1, use_multiprocessing=False, initial_epoch=0):
        if len(self.callbacks) == 0:
            callbacks = None
        else:
            lr = self.lr

            def lr_schedule(epoch):
                return lr * (0.1 ** int(epoch / 10))

            callbacks = self.callbacks
            now = strftime("%d-%m-%Y_%H-%M", gmtime())
            log_file = 'log/tensorboard-' + self.name_to_file() + '-logs-' + str(now)
            callbacks.append(TensorBoard(log_dir=log_file, write_grads=True,
                                         batch_size=steps_per_epoch, write_images=True))
            callbacks.append(LearningRateScheduler(lr_schedule))

        if self.auto_save:
            self.auto_save_model()
        history = self.model.fit_generator(generator, steps_per_epoch=int(steps_per_epoch), epochs=int(epochs),
                                           validation_data=validation_data,
                                           validation_steps=validation_steps,
                                           callbacks=callbacks,
                                           workers=workers,
                                           use_multiprocessing=use_multiprocessing,
                                           initial_epoch=initial_epoch)
        if self.auto_save:
            self.auto_save_weights(epochs)
        return history

    def init_callbacks(self, callbacks_names=tuple(('LogCallback', 'CSVLogger', 'ModelCheckpoint', 'TensorBoard')),
                       callbacks=None):
        Model.init_callbacks(self, callbacks_names, callbacks)
