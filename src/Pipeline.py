import logging

logging.basicConfig(filename='log/preprocessing.log', level=logging.INFO,
                    format='%(levelname)s: %(asctime)s: %(message)s')


class Pipeline:

    def __init__(self, verbose=False, log_active=False):
        self.pipeline = list()
        self.verbose = verbose
        self.log_active = log_active

    def add_preprocessors(self, preprocessors):
        self.pipeline += preprocessors

    def set_verbose(self, value):
        self.verbose = value

    def evaluate(self, value):
        result = value
        if self.verbose:
            print(value.shape)
        for preprocessor in self.pipeline:
            if self.verbose:
                print('Executing: ' + preprocessor.title)
            if self.log_active:
                logging.info('Executing action: ' + preprocessor.title)
            result = preprocessor.evaluate(result)
            if self.verbose:
                print('Image shape after action ' + preprocessor.title + ' is: ' + str(result.shape))
            if self.log_active:
                logging.info('Image shape after action ' + preprocessor.title + ' is: ' + str(result.shape))

        return result
