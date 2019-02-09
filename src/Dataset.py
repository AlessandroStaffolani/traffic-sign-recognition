import pandas as pd
import numpy as np


class Dataset:

    def __init__(self, source_path, chunk_size=1000, labels=None):
        self._source_path = source_path
        self._chunk_size = chunk_size
        self.labels = labels
        self.data_frame = None

        self._init()

    def _init(self):
        columns = ['images']
        if self.labels is not None:
            columns += self.labels
        self.data_frame = pd.read_csv(self._source_path, chunksize=self._chunk_size, names=columns)

    def set_chunk_size(self, chunk):
        self._chunk_size = chunk

    def get_generator(self):
        for df in self.data_frame:
            images = df.iloc[:, 0].values  # all row, first column
            if self.labels is not None:
                labels = df.iloc[:, 1:].values  # all row, second column to the end

                yield images.astype(np.uint8), labels.astype(np.uint8)
            else:
                yield images.astype(np.uint8)
