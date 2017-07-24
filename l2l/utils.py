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
        self.vars_in_scope = tuple(self._get_vars_in_scope()) # Make immutable
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
        variables=self.variables_in_scope
        sum_flattened_shapes = 0
        for var in variables:
            var_shape = var.get_shape().as_list()
            flattened_shape = 1
            for el in var_shape:
               flattened_shape *= el
            sum_flattened_shapes+= flattened_shape
        return sum_flattened_shapes

    def matching_grads(self, list_of_grads):
         '''Takes a list of (gradient, variable) pairs for all trainable 
         variables and returns the same list type for the pairs corresponding to
         variable in scope (of self)
         '''
         current_vars = set(self.vars_in_scope)
         list_of_current_grads = []
         for (grad, var) in list_of_grads:
            if var is in current_vars:
                list_of_current_grads.append((grad, var))


    def flatten(self, input_tensors):
        '''Flatten the list (batch) of lists (vars) of (gradient, variable) 
        pairs into a single tensor of shape (batch_size, k) to input into the
        meta optimizing RNN
        '''
        # TODO make this work with batch sizes
        with tf.name_scope('flatten'):
            ###tf.contrib.layers.flatten(var)
            flattened_tensors = [tf.reshape(var, [-1, 1]) for var in input_tensors]
            flattened_tensor = tf.concat(flattened_tensors)

            if flattened_tensor.get_shape().as_list() is not self.flattened_shape:
                raise ValueError('self.flattened shape is {}, but for these \
                    inputs the flattened shape is {}'.format(self.flattened_shape,
                    flattened_tensor.get_shape().as_list()))
            else:
                return flattened_tensor

    def unflatten(self, flattened_deltas):
        '''The output of each of the meta optimizer's RNNs will give out a
        flattened output (batch_size, k), unflatten will take that tensor in and
        return a list of tuples representing the deltas and their respective 
        variables

        Returns in the form of a list of (delta, variable) 
        '''
        if input_tensors.get_shape().as_list() is not self.flattened_shape:
            raise ValueError('Incorrect input size to unflatten : \
                    self.flattened shape is {}, but for this input the \
                    flattened shape is {}'.format(self.flattened_shape,
                    input_tensors.get_shape().as_list()))

        index = 0
        for var in self.vars_in_scope:
            var_shape = var.get_shape().as_list()


merge_var_lists(list_flattened_deltas):
    raise NotImplementedError

