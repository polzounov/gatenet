import tensorflow as tf
import numpy as np
#from parameters import Parameters
from tensorflow_utils import *


######################################################################
## Code for Graph construction

class Graph:

    def __init__(self):
        self.input_layer = None
        self.gated_layers = None
        self.output_layer = None
        
    ## Build graph to test code
    def buildTestGraph(self, input_images):

        ##################################################
        ## Define Hyperparameters
        self.M = 10
        self.L = 3
        tensor_size = 20

        M = self.M
        L = self.L
        ##################################################
        
        ##################################################
        ## Define Modules
        
        input_modules = np.zeros(M, dtype=object)
        gated_modules = np.zeros((L, M), dtype=object)
        output_modules = np.zeros(1, dtype=object)

        for i in range(M):
            input_modules[i] = PerceptronModule(weight_variable([784, tensor_size]),
                                                bias_variable([tensor_size]), activation=tf.nn.relu)
            
            for j in range(L):
                gated_modules[j][i] = PerceptronModule(weight_variable([tensor_size, tensor_size]),
                                                       bias_variable([tensor_size]), activation=tf.nn.relu)
        output_modules[0] = LinearModule(weight_variable([tensor_size, 10]), bias_variable([10]))

        ##################################################
        

        ##################################################
        ## Define Layers

        self.input_layer = InputLayer(input_modules)
        self.gated_layers = []
        for i in range(L):
            gated_layer = GatedLayer(gated_modules[i])
            self.gated_layers.append(gated_layer)

        self.output_layer = OutputLayer(output_modules)

        ##################################################


        ##################################################
        ## Construct graph
        layer_output = self.input_layer.processLayer(input_images)
        for i in range(L):
            layer_output = self.gated_layers[i].processLayer(layer_output)

        logits = self.output_layer.processLayer(layer_output)
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
        return np.sum(output_tensors) / len(self.modules)



class GatedLayer(Layer):
    def __init__(self, modules):
        self.modules = modules

        ## CHANGE NUMBERS !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        self.gate_module = LinearModule(weight_variable([20, len(modules)]),
                                              bias_variable([len(modules)]))

        self.gates = None
        self.gamma = 2 #1.333

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

        return np.sum(output_tensors) / len(self.modules)

class OutputLayer(Layer):
    def __init__(self, modules):
        self.modules = modules

    def processLayer(self, input_tensors):
        output_tensors = np.zeros(len(self.modules), dtype=object)
        for i in range(len(self.modules)):
            output_tensors[i] = self.modules[i].processModule(input_tensors)
        return np.sum(output_tensors)/len(self.modules)

######################################################################