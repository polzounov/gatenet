import tensorflow as tf
import numpy as np
import sonnet as snt
#from parameters import Parameters
from tensorflow_utils import *

######################################################################
## Code for Modules
###### TEMPORARY CODE ################################################
"""
  A naive implementation of the forward pass for a convolutional layer.
  The input consists of N data points, each with C channels, height H and width
  W. We convolve each input with F different filters, where each filter spans
  all C channels and has height HH and width HH.
  Input:
  - x: Input data of shape (N, C, H, W)
  - w: Filter weights of shape (F, C, HH, WW)
  - b: Biases, of shape (F,)
  - conv_param: A dictionary with the following keys:
    - 'stride': The number of pixels between adjacent receptive fields in the
      horizontal and vertical directions.
    - 'pad': The number of pixels that will be used to zero-pad the input.
  Returns a tuple of:
  - out: Output data, of shape (N, F, H', W') where H' and W' are given by
    H' = 1 + (H + 2 * pad - HH) / stride
    W' = 1 + (W + 2 * pad - WW) / stride
  - cache: (x, w, b, conv_param)
"""
def weight_helper(input_shape,
                  output_shape,
                  filters=(3,3)):
    '''Takes input shape and output shape, and decides the shapes of
    weights needed to convert from input to output shape
    '''
    #print('Weights init: input_shape', input_shape, 'output_shape', output_shape)
    dims_in = len(input_shape)
    dims_out = len(output_shape)
    if dims_in != dims_out:
        print('dims_in:', dims_in, ', dims_out:', dims_out)
        raise ValueError('Input and output dimensions do not match, \
                          (reshape the input to 2d)')
    if dims_in == 2:
        return (input_shape[1], output_shape[1])

    # For the 4d (convolutional) case
    N, C, W, H = input_shape
    Nout, F, Wout, Hout = output_shape
    HH, WW = filters
    if N != Nout:
        raise ValueError('Number of input and output datapoints are \
                          different')
    return (F, C, HH, WW)


def bias_helper(input_shape,
                  output_shape,
                  filters=(3,3)):
    '''Takes input shape and output shape, and decides the shapes of
    biases needed to convert from input to output shape
    '''
    #print('Bias inint: input_shape', input_shape, 'output_shape', output_shape)
    dims_in = len(input_shape)
    dims_out = len(output_shape)
    if dims_in != dims_out:
        print('dims_in:', dims_in, ', dims_out:', dims_out)
        raise ValueError('Input and output dimensions do not match, \
                          (reshape the input to 2d)')
    if dims_in == 2:
        return (output_shape[1],)

    # For the 4d (convolutional) case
    N, C, W, H = input_shape
    Nout, F, Wout, Hout = output_shape
    HH, WW = filters
    if N != Nout:
        raise ValueError('Number of input and output datapoints are \
                          different')
    return (F,)


class Module():
    def __init__(self, input_shape, output_shape, activation):
        self.weights = weight_variable(weight_helper(input_shape, output_shape))
        self.biases = bias_variable(bias_helper(input_shape, output_shape))
        #print('weights', self.weights.shape)
        self.activation = activation


class PerceptronModule(Module):
    # 2d only
    def __init__(self, input_shape, output_shape, activation):
        super(PerceptronModule, self).__init__(input_shape, output_shape, activation)

    def process_module(self, input_tensor):
        return self.activation(tf.matmul(input_tensor, self.weights) + self.biases)

class LinearModule(Module):
    # 2d only
    def __init__(self, input_shape, output_shape, activation=None):
        super(LinearModule, self).__init__(input_shape, output_shape, activation)
        # Ignore the activation function if given

    def process_module(self, input_tensor):
        return tf.matmul(input_tensor, self.weights) + self.biases
###### END TEMPORARY CODE ###########################################

'''
class ResidualPerceptronModule(Module):
    # 2d only
    def __init__(self, weights, biases, activation):
        super(ResidualPerceptronModule, self).__init__(weights, biases, activation)

    def process_module(self, input_tensor):
        return self.activation(tf.matmul(input_tensor, self.weights) + self.biases) + input_tensor


class LinearModule(Module):
    # 2d only
    def __init__(self, weights, biases, activation=None):
        super(LinearModule, self).__init__(weights, biases, activation)
        # Ignore the activation function if given

    def process_module(self, input_tensor):
        return tf.matmul(input_tensor, self.weights) + self.biases


class IdentityModule(Module):
    # Both 2d and 4d
    def __init__(self, *args):
        pass

    def process_module(self, input_tensor):
        return input_tensor


class ConvModule(Module):
    # 4D only
    def __init__(self, weights, activation, strides=[1, 1, 1, 1], padding='SAME'):
        super(ResidualConvModule, self).__init__(weights, biases, activation)
        self.strides = strides
        self.padding = padding

    def process_module(self, input_tensor):
        return self.activation(tf.nn.conv2d(input_tensor, 
                                            self.weights, 
                                            strides=self.strides, 
                                            padding=self.padding) + biases)


class ResidualConvModule(Module):
    # 4D only
    def __init__(self, weights, activation, strides=[1, 1, 1, 1], padding='SAME'):
        super(ResidualConvModule, self).__init__(weights, biases, activation)
        self.strides = strides
        self.padding = padding

    def process_module(self, input_tensor):
        return self.activation(tf.nn.conv2d(input_tensor, 
                                            self.weights, 
                                            strides=self.strides, 
                                            padding=self.padding) + biases) + input_tensor
'''