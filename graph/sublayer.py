import tensorflow as tf
import numpy as np
#from parameters import Parameters
from tensorflow_utils import *

######################################################################
## Code for Sublayer
class Sublayer():
    def __init__(self, input_size, num_modules):
        self.input_size = input_size
        self.num_modules = num_modules


class AdditionSublayerModule(Sublayer):
    def __init__(self, input_size, num_modules):
        super(AdditionSublayerModule, self).__init__(input_size, num_modules)
        self.output_size = input_size

    def process_sublayer(self, module_tensors):
        return np.sum(module_tensors) / self.num_modules


class ConcatenationSublayerModule(Sublayer):
    def __init__(self, input_size, num_modules):
        super(ConcatenationSublayerModule, self).__init__(input_size, num_modules)
        self.output_size = input_size*num_modules

    def process_sublayer(self, module_tensors):
        output = module_tensors[0]
        for i in range(len(module_tensors)-1):
            output = tf.concat([output, module_tensors[i+1]], axis=1)
        return output
