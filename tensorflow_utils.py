from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

def flatten_to_2d(input_tensor):
    '''Flatten the input to 2d if input is in 4d'''
    if len(input_tensor.shape) == 2:
        return input_tensor
    N, H, W, C = input_tensor.get_shape().as_list()
    input_tensor = tf.reshape(input_tensor, [-1, H*W*C])
    return input_tensor

def weight_variable(shape):
  """Create a weight variable with appropriate initialization."""
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  """Create a bias variable with appropriate initialization."""
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def variable_summaries(var, options=None): # KEEP
  """Attach a lot of summaries to a Tensor (for TensorBoard visualization).
    - options is a dictionary with additional options
  """
  with tf.name_scope('summaries'):
    mean = tf.reduce_mean(var)
    tf.summary.scalar('mean', mean)
    with tf.name_scope('stddev'):
      stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.summary.scalar('stddev', stddev)
    tf.summary.scalar('max', tf.reduce_max(var))
    tf.summary.scalar('min', tf.reduce_min(var))
    tf.summary.histogram('histogram', var)

    if options is not None:
      # Get the total info flowing through the module (useful for understanding 
      # information flow through the overall network)
      if options.get('flow') is True:
        tensor_sum = tf.reduce_sum(var)
        tf.summary.scalar('flow', tensor_sum)