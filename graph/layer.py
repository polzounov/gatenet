import tensorflow as tf
import numpy as np
#from parameters import Parameters
from tensorflow_utils import *

from graph.module import *

######################################################################
## Code for Layers
class Layer():
    def __init__(self, layer_definition):
        # Todo 
        self.M = layer_definition['M']
        self.input_shape = layer_definition.get('input_shape')
        self.module_output_shape = layer_definition.get('module_output_shape')
        # TODO: Make module type work with lists (for module variation inside layer)
        self.ModuleType = layer_definition.get('module_type')
        self.SublayerType = layer_definition.get('sublayer_type')
        self.act = layer_definition.get('activations', tf.nn.relu)
        self._build_modules() # Init modules in layer
        self._build_sublayer()

    def _build_modules(self):
        self.modules = np.zeros(self.M, dtype=object)
        for i in range(self.M):
            self.modules[i] = self.ModuleType(self.input_shape, self.module_output_shape, activation=self.act)

    def _build_sublayer(self):
        self.sublayer = self.SublayerType(self.module_output_shape, self.M)
        self.layer_output_shape = self.sublayer.output_shape


class GatedLayer(Layer):
    def __init__(self, layer_definition, gamma=2.0):
        super(GatedLayer, self).__init__(layer_definition)
        self.gate_module = LinearModule(self.input_shape, (None, len(self.modules)))
        self.gates = None
        self.gamma = gamma

    def compute_gates(self, input_tensors):
        gates_unnormalized = self.gate_module.process_module(input_tensors)
        gates_pow = tf.pow(gates_unnormalized, self.gamma)

        gg = tf.reshape(tf.reduce_sum(gates_pow, axis = 1), [-1,1])
        num_cols = len(self.modules)
        gates_tiled = tf.tile(gg, [1, num_cols])

        gates_normalized = tf.nn.relu(gates_pow / gates_tiled)
        self.gates = tf.nn.relu(gates_normalized) # CHANGE !!! self.gates == same thing
        return gates_normalized

    def process_layer(self, input_tensors):
        gates = self.compute_gates(input_tensors) # CHANGE !!! self.gates == same thing
        output_tensors = np.zeros(len(self.modules), dtype=object)

        for i in range(len(self.modules)):
            # Get the number of rows in the fed value at run-time.
            output_tensors[i] = self.modules[i].process_module(input_tensors)
            num_cols = np.int32(output_tensors[i].get_shape()[1])

            gg = tf.reshape(gates[:,i], [-1,1])
            gates_tiled = tf.tile(gg, [1,num_cols])
            output_tensors[i] = tf.multiply(output_tensors[i], gates_tiled)

        return self.sublayer.process_sublayer(output_tensors)


class OutputLayer(Layer):
    def __init__(self, layer_definition):
        super(OutputLayer, self).__init__(layer_definition)
        ## TODO: Clean up code
        # Check that ModuleType makes sense
        if self.ModuleType not in set([LinearModule, PerceptronModule]):
            print('self.ModuleType is:', self.ModuleType)
            raise ValueError('Ouput layer has incorrect module type')
        ## END TODO

    def process_layer(self, input_tensors):
        print('flatten tensor, shape before:', input_tensors.shape)
        input_tensors = self._flatten_input(input_tensors) # force 2d ## TODO: Clean up code
        print('flatten tensor, shape after:', input_tensors.shape)
        output_tensors = np.zeros(len(self.modules), dtype=object)
        for i in range(len(self.modules)):
            output_tensors[i] = self.modules[i].process_module(input_tensors)
        return self.sublayer.process_sublayer(output_tensors)
######################################################################