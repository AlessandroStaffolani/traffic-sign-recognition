

class Pipeline:

    def __init__(self, verbose=False):
        self.pipeline = list()
        self.verbose = verbose

    def add_preprocessors(self, preprocessors):
        self.pipeline += preprocessors

    def set_verbose(self, value):
        self.verbose = value

    def evaluate(self, value):
        result = value
        for preprocessor in self.pipeline:
            if self.verbose:
                print('Executing: ' + preprocessor.title)
            result = preprocessor.evaluate(result)

        return result
