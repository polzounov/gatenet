from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
#from parameters import Parameters
from tensorflow_utils import *

######################################################################
## Code for Sublayer
class Sublayer():
  def __init__(self, num_modules):
    self.num_modules = num_modules


class AdditionSublayerModule(Sublayer):
  '''Averages the values of the sublayer (assume inputs are same size)'''
  def __init__(self, num_modules):
    super(AdditionSublayerModule, self).__init__(num_modules)

  def process_sublayer(self, module_tensors):
    return np.sum(module_tensors) / self.num_modules

"""
class ConcatenationSublayerModule(Sublayer):
  '''Concatenates the input modules along axis=1, (H and W should be the same)'''
  # For 4d tensors order is NCHW
  def __init__(self, input_shape, num_modules):
    super(ConcatenationSublayerModule, self).__init__(input_shape, num_modules)
    self.output_shape = input_shape*num_modules

  def process_sublayer(self, module_tensors):
    output = module_tensors[0]
    for i in range(len(module_tensors)-1):
      output = tf.concat([output, module_tensors[i+1]], axis=1)
    return output
"""

class IdentitySublayerModule(Sublayer):
  '''Returns input (can only be one input!)'''
  def __init__(self, num_modules):
    super(IdentitySublayerModule, self).__init__(num_modules)
    if self.num_modules is not 1:
      print('self.num_modules: ', self.num_modules)
      raise ValueError('Incorrect number of modules for IdentitySublayerModule, should be 1')

  def process_sublayer(self, module_tensors):
    return module_tensors
