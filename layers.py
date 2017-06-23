import tensorflow as tf
import numpy as np
#from parameters import Parameters
from tensorflow_utils import *


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


######################################################################
## Code for Graph construction
class Graph:

    def __init__(self):
        self.input_layer = None
        self.gated_layers = None
        self.output_layer = None
        self.sublayers = None
        
    ## Build graph to test code
    def buildTestGraph(self, input_images, parameter_dict):

        ##################################################
        ## Define Hyperparameters
        self.M = parameter_dict['M']
        self.L = parameter_dict['L']
        self.tensor_size = parameter_dict['tensor_size']
        self.gamma = parameter_dict['gamma']
        ##################################################


        ##################################################
        ## Define Sublayers
        self.sublayers = np.zeros(self.L + 1, dtype= object)
        input_size = self.tensor_size
        num_modules = self.M
        for i in range(self.L + 1):
            self.sublayers[i] = Sublayer(input_size, num_modules, ConcatenationSublayerModule(input_size,num_modules))

        ##################################################
        ## Define Modules        
        input_modules = np.zeros(self.M, dtype=object)
        gated_modules = np.zeros((self.L, self.M), dtype=object)
        output_modules = np.zeros(1, dtype=object)

        print(self.sublayers[-1].output_size)

        for i in range(self.M):
            input_modules[i] = PerceptronModule(weight_variable([784, self.sublayers[0].input_size]),
                                                bias_variable([self.sublayers[0].input_size]), activation=tf.nn.relu)
            
            for j in range(self.L):
                gated_modules[j][i] = PerceptronModule(weight_variable([self.sublayers[j].output_size, self.sublayers[j+1].input_size]),
                                                       bias_variable([self.sublayers[j+1].input_size]), activation=tf.nn.relu)
        output_modules[0] = LinearModule(weight_variable([self.sublayers[-1].output_size, 10]), bias_variable([10]))
        ##################################################
        

        ##################################################
        ## Define Layers
        self.input_layer = InputLayer(input_modules)
        self.gated_layers = []
        for i in range(self.L):
            gated_layer = GatedLayer(gated_modules[i], self.sublayers[i].output_size,  self.gamma)
            self.gated_layers.append(gated_layer)

        self.output_layer = OutputLayer(output_modules)
        ##################################################


        ##################################################
        ## Construct graph
        layer_output = self.input_layer.processLayer(input_images)
        sublayer_output = self.sublayers[0].processSublayer(layer_output)
        for i in range(self.L):
            layer_output = self.gated_layers[i].processLayer(sublayer_output)
            sublayer_output = self.sublayers[i+1].processSublayer(layer_output)

        logits = self.output_layer.processLayer(sublayer_output)
        ##################################################
        return logits


    def determineGates(self, image, x, sess):
        gates = np.zeros((self.L, self.M))
        for i in range(len(self.gated_layers)):
            g = sess.run([self.gated_layers[i].gates],
                         feed_dict={x:image})
            gates[i] = np.array(g)
        return gates
######################################################################


######################################################################
## Code for Modules
class Module:
    def processModule(self, inputTensor):
        pass


class PerceptronModule(Module):
    def __init__(self, weights, biases, activation):
        self.weights = weights
        self.biases = biases
        self.activation = activation

    def processModule(self, input_tensor):
        return self.activation(tf.matmul(input_tensor, self.weights) + self.biases)


class ResidualPerceptronModule(Module):
    def __init__(self, weights, biases, activation):
        self.weights = weights
        self.biases = biases
        self.activation = activation

    def processModule(self, input_tensor):
        return self.activation(tf.matmul(input_tensor, self.weights) + self.biases) + input_tensor


class LinearModule(Module):

    def __init__(self, weights, biases):
        self.weights = weights
        self.biases = biases

    def processModule(self, input_tensor):
        return tf.matmul(input_tensor, self.weights) + self.biases
######################################################################


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
    def __init__(self, modules, input_size, gamma=2.0,  ):
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