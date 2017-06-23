import tensorflow as tf
import numpy as np
#from parameters import Parameters
from tensorflow_utils import *

######################################################################
## Code for Sublayer
class Sublayer:

    def __init__(self, input_size, num_modules, sublayer_module):
        self.input_size = input_size
        self.sublayer_module = sublayer_module
        self.num_modules = num_modules
        self.output_size = self.sublayer_module.output_size

    def processSublayer(self, module_tensors):
        return self.sublayer_module.processSublayerModule(module_tensors)
######################################################################

class AdditionSublayerModule:
    def __init__(self, input_size, num_modules):
        self.input_size = input_size
        self.output_size = input_size
        self.num_modules = num_modules

    def processSublayerModule(self, module_tensors):
        return np.sum(module_tensors) / self.num_modules


class ConcatenationSublayerModule:
    def __init__(self, input_size, num_modules):
        self.input_size = input_size
        self.output_size = input_size*num_modules
        self.num_modules = num_modules

    def processSublayerModule(self, module_tensors):
        output = module_tensors[0]
        for i in range(len(module_tensors)-1):
            output = tf.concat([output, module_tensors[i+1]], axis=1)
        return output
