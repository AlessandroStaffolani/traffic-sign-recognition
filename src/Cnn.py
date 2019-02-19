import keras
import json
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPool2D, Dropout
from keras.models import model_from_json


class Cnn:

    def __init__(self, num_layers=2, nodes_per_layer=100, layer_activation='relu', num_output=43,
                 output_activation='softmax', kernel_size=3, input_shape=(46, 46, 3)):
        self.model = None
        self.num_layers = num_layers
        self.nodes_per_layer = nodes_per_layer
        self.layers_activation = layer_activation
        self.num_output = num_output
        self.output_activation = output_activation
        self.kernel_size = kernel_size
        self.input_shape = input_shape

    def create_model(self):
        self.model = Sequential()

        self.model.add(
            Conv2D(filters=32, kernel_size=self.kernel_size, activation=self.layers_activation,
                   strides=(1, 1), padding='same', input_shape=self.input_shape, data_format='channels_last'))

        self.model.add(MaxPool2D(pool_size=(2, 2), strides=2))

        self.model.add(Dropout(0.2))

        self.model.add(
            Conv2D(filters=64, kernel_size=self.kernel_size, activation=self.layers_activation, strides=(1, 1),
                   padding='valid'))

        self.model.add(MaxPool2D(pool_size=(2, 2), strides=2))

        self.model.add(Flatten())

        # Add the output layer
        self.model.add(Dense(self.num_output, activation=self.output_activation))

    def compile(self, optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy']):
        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    def fit(self, train_data, train_labels, validation_split=0.25, epochs=10, batch_size=100):
        self.model.fit(train_data, train_labels, validation_split=validation_split, epochs=epochs,
                       batch_size=batch_size)

    def fit_generator(self, generator, steps_per_epoch=1000, epochs=10):
        self.model.fit_generator(generator, steps_per_epoch=int(steps_per_epoch), epochs=int(epochs),
                                 use_multiprocessing=True)

    def evaluate(self, test_data, test_labels, batch_size=100):
        self.model.evaluate(test_data, test_labels, batch_size=batch_size)

    def save_json_model(self, out_path):
        json_model = self.model.to_json()
        with open(out_path, 'w') as outfile:
            outfile.write(json_model)

    def load_json_model(self, file):
        json_file = open(file, 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        self.model = model_from_json(loaded_model_json)


def get_value_if_list_or_int(value, index):
    if type(value) is list:
        return value[index]
    else:
        return value
