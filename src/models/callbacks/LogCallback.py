import logging

from keras.callbacks import Callback


class LogCallback(Callback):

    def __init__(self, log_file='log/model.log', level=logging.INFO, format='%(levelname)s: %(asctime)s: %(message)s'):
        Callback.__init__(self)

        logging.basicConfig(filename=log_file, level=level,
                            format=format)

    def on_train_begin(self, logs=None):
        logging.info('Training start - logs: ' + str(logs))

    def on_train_end(self, logs=None):
        logging.info('Training end - logs: ' + str(logs))

    def on_epoch_end(self, epoch, logs=None):
        logging.info('Epoch ' + str(epoch + 1) + ' completed - logs: ' + str(logs))
