import tensorflow as tf
import numpy as np
#from parameters import Parameters
from tensorflow_utils import *

from graph.module import *

######################################################################
## Code for Layers
class Layer:
    pass

class InputLayer(Layer):
    def __init__(self, modules):
        self.modules = modules

    def processLayer(self, input_tensors):
        output_tensors = np.zeros(len(self.modules),dtype=object)
        for i in range(len(self.modules)):
            output_tensors[i] = self.modules[i].processModule(input_tensors)
        return output_tensors


class GatedLayer(Layer):
    def __init__(self, modules, input_size, output_size, gamma=2.0):
        self.modules = modules
        self.input_size = input_size

        ## CHANGE NUMBERS !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        self.gate_module = LinearModule(weight_variable([self.input_size, len(modules)]),
                                              bias_variable([len(modules)]))
        self.gates = None
        self.gamma = gamma

    def computeGates(self, input_tensors):
        gates_unnormalized = self.gate_module.processModule(input_tensors)
        gates_pow = tf.pow(gates_unnormalized, self.gamma)

        gg = tf.reshape(tf.reduce_sum(gates_pow, axis = 1), [-1,1])
        num_cols = len(self.modules)
        gates_tiled = tf.tile(gg, [1, num_cols])

        gates_normalized = tf.nn.relu(gates_pow / gates_tiled)
        self.gates = tf.nn.relu(gates_normalized)
        return gates_normalized

    def processLayer(self, input_tensors):
        gates = self.computeGates(input_tensors)
        output_tensors = np.zeros(len(self.modules), dtype=object)

        for i in range(len(self.modules)):
            # Get the number of rows in the fed value at run-time.
            output_tensors[i] = self.modules[i].processModule(input_tensors)
            num_cols =  np.int32(output_tensors[i].get_shape()[1])

            gg = tf.reshape(gates[:,i], [-1,1])
            gates_tiled = tf.tile(gg, [1,num_cols])
            output_tensors[i] = tf.multiply(output_tensors[i], gates_tiled)

        return output_tensors


class OutputLayer(Layer):
    def __init__(self, modules):
        self.modules = modules

    def processLayer(self, input_tensors):
        output_tensors = np.zeros(len(self.modules), dtype=object)
        for i in range(len(self.modules)):
            output_tensors[i] = self.modules[i].processModule(input_tensors)
        return np.sum(output_tensors)/len(self.modules)
######################################################################