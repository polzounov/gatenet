import pickle
# For module and submodule type
from graph.sublayer import *
from graph.module import *

class Parameters:
    def __init__(self):
        self.M = 2
        self.L = 2
        self.C = 10 # MNIST digits

        self.module_type = ConvModule
        self.sublayer_type = AdditionSublayerModule
        self.tensor_shape = (None, 1, 28, 28)
        self.image_shape = (None, 1, 28, 28)

        self.gamma = 2
        self.batch_size = 100
        self.num_batches = 1001
        self.learning_rate = 0.001
        self.output_file = 'test'

class OutputManager:

    def __init__(self):
        self.output_file = None
        self.parameter_dict = None
        self.data = None
        self.labels = None
        self.images = None


    def initialize(self, parameter_dict):
        self.parameter_dict = parameter_dict
        self.data = []
        self.labels = []
        self.images = []

    def addData(self, data, label, image):
        self.data.append(data)
        self.labels.append(label)
        self.images.append(image)

    def save(self):
        output = (self.data, self.labels, self.images, self.parameter_dict)
        with open('output/' + self.parameter_dict['output_file'] + '.pkl', 'wb') as f:
            pickle.dump(output, f , pickle.HIGHEST_PROTOCOL)

