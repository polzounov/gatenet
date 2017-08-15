from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import sonnet as snt
import numpy as np

class FlatteningHelper():
    """This class takes in a variable scope and can flatten and unflatten them

    For a scope, it can return the summation of the flattened version of all the
    variables in the scope, and it can take in that flattened version and output
    the original variables and shapes 
    """
    def __init__(self, scope): 
        self.scope = scope
        self.vars_in_scope = self._get_vars_in_scope()
        self.flattened_shape = self._get_flattened_shape()
        self._second_derivatives = False

    def _get_vars_in_scope(self):
        '''Returns a list of trainable variables in `scope`'''
        scope = self.scope
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)
        # snt.get_variables_in_module(network)

    def _get_flattened_shape(self):
        '''Return the sum of all the flattened shapes of vars
        Eg. var1 is (10,5), var2 is (2,4,1), then return 50+8 = 58
        '''
        variables = self.vars_in_scope
        sum_flattened_shapes = 0
        for var in variables:
            var_shape = var.get_shape().as_list()
            flattened_shape = 1
            for el in var_shape:
               flattened_shape *= el
            sum_flattened_shapes+= flattened_shape
        return sum_flattened_shapes

    @staticmethod
    def _product_of_list(l):
        prod = 1
        for el in l:
            prod *= el
        return prod

    def matching_grads(self, list_of_grads):
         '''Takes a list of (gradient, variable) pairs for all trainable 
         variables and returns the same list type for the pairs corresponding to
         variable in scope (of self)
         '''
         current_vars = set(self.vars_in_scope)
         list_of_current_grads = []
         for (grad, var) in list_of_grads:
            if var in current_vars:
                list_of_current_grads.append((grad, var))


    def flatten(self, input_tensors):
        '''Flatten a list of (gradient, variable) pairs into a single tensor of
        shape (k,) to input into the meta optimizing RNN
        eg  [(g1, v1), (g2, v2)] -> [g1_flattened, g2_flattened, ...]
        '''
        with tf.name_scope('flatten'):
            flattened_tensors = [tf.reshape(var, [-1, 1]) for var in input_tensors]
            flattened_tensor = tf.concat(flattened_tensors, axis=0)

            if flattened_tensor.get_shape().as_list()[0] != self.flattened_shape:
                raise ValueError('self.flattened shape is',
                                 self.flattened_shape,
                                 ', but for these inputs the flattened shape is',
                                 flattened_tensor.get_shape().as_list()[0])
            else:
                return flattened_tensor

    def unflatten(self, flattened_deltas):
        '''The output of each of the meta optimizer's RNNs will give out a
        flattened output (k,), unflatten will take that tensor in and
        return a list of tuples representing the deltas and their respective 
        variables
        
        eg [g1_flattened, g2_flattened, ...] -> [(g1, v1), (g2, v2)]
        Returns in the form of a list of (delta, variable) 
        '''
        if flattened_deltas.get_shape().as_list()[0] != self.flattened_shape:
            raise ValueError('Incorrect input size to unflatten :',
                    '\n\t self.flattened shape is:', 
                    self.flattened_shape, 
                    ', but for this input the flattened shape is:',
                    flattened_deltas.get_shape().as_list()[0])

        # TODO clean this up
        with tf.name_scope('unflatten'):
            index = 0
            unflattened_grads = []
            for var in self.vars_in_scope:
                # Get the current shape delta is in, and the desired shape
                original_var_shape = var.get_shape().as_list()
                flattened_var_shape = self._product_of_list(original_var_shape)
                # Get the delta in its current shape
                flattened_delta = flattened_deltas[index:index+flattened_var_shape]
                index += flattened_var_shape
                # Reshape the delta to the original shape of the grad
                unflattened_delta = tf.reshape(flattened_delta, original_var_shape)
                delta_var_tuple = (unflattened_delta, var)
                unflattened_grads.append(delta_var_tuple)
            return unflattened_grads


def merge_var_lists(list_flattened_deltas, grads=None):
    list_deltas = []
    for flattened_deltas in list_flattened_deltas:
            list_deltas += delta

    if grads is not None: # Do we need to reorder to same order as grads???
        raise NotImplementedError

    return list_deltas



