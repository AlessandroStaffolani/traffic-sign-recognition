import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten


class Cnn:

    def __init__(self, num_layers=2, nodes_per_layer=100, layer_activation='relu', num_output=3,
                 output_activation='softmax', kernel_size=3, input_shape=(48, 48, 3)):
        self.model = None
        self.num_layers = num_layers
        self.nodes_per_layer = nodes_per_layer
        self.layers_activation = layer_activation
        self.num_output = num_output
        self.output_activation = output_activation
        self.kernel_size = kernel_size
        self.input_shape = input_shape

        self._create_model()

    def _create_model(self):
        self.model = Sequential()

        # Add Convolutional layers
        for layer in range(self.num_layers):
            nodes = get_value_if_list_or_int(self.nodes_per_layer, layer)
            activation = get_value_if_list_or_int(self.layers_activation, layer)
            kernel_size = get_value_if_list_or_int(self.kernel_size, layer)
            if layer == 0:
                # First layer need input_shape
                self.model.add(
                    Conv2D(nodes, activation=activation, kernel_size=kernel_size, input_shape=self.input_shape))
            else:
                self.model.add(Conv2D(nodes, activation=activation, kernel_size=kernel_size))

        # Add Flatten layer to translate between the image processing and classification part
        self.model.add(Flatten())

        # Add the output layer
        self.model.add(Dense(self.num_output, activation=self.output_activation))

    def compile(self, optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy']):
        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    def fit(self, train_data, train_labels, validation_split=0.25, epochs=10, batch_size=100):
        self.model.fit(train_data, train_labels, validation_split=validation_split, epochs=epochs, batch_size=batch_size)

    def evaluate(self, test_data, test_labels, batch_size=100):
        self.model.evaluate(test_data, test_labels, batch_size=batch_size)


def get_value_if_list_or_int(value, index):
    if type(value) is list:
        return value[index]
    else:
        return value
