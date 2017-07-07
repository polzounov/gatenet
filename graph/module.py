import tensorflow as tf
import numpy as np
#from parameters import Parameters
from tensorflow_utils import *

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

    def __init__(self, weights, biases, activation=None):
        self.weights = weights
        self.biases = biases
        # Don't use the activation if it is given

    def processModule(self, input_tensor):
        return tf.matmul(input_tensor, self.weights) + self.biases
######################################################################
