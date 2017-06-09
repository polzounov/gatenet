from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import tensorflow as tf
import numpy as np
import sys

def parameters_backup(var_list_to_learn): # KEEP
  # Copy each element of var_list_to_learn into var_list_backup
  var_list_backup = np.zeros(len(var_list_to_learn), dtype=object)
  for i in range(len(var_list_to_learn)):
    var_list_backup[i] = var_list_to_learn[i].eval()
  return var_list_backup

def parameters_update(sess, var_update_placeholders, var_update_ops, var_list_backup): # KEEP
  ######## TODO CHECK OVER
  # For each of the placeholders update the values to the new values
  # I'm assuming this is for setting the geopaths to another path
  # --- Check in other code
  for i in range(len(var_update_placeholders)):
    sess.run(var_update_ops[i], {var_update_placeholders[i]:var_list_backup[i]})

def weight_variable(shape): # KEEP
  """Create a weight variable with appropriate initialization."""
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape): # KEEP
  """Create a bias variable with appropriate initialization."""
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)


def variable_summaries(var): # KEEP
  """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
  with tf.name_scope('summaries'):
    mean = tf.reduce_mean(var)
    tf.summary.scalar('mean', mean)
    with tf.name_scope('stddev'):
      stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.summary.scalar('stddev', stddev)
    tf.summary.scalar('max', tf.reduce_max(var))
    tf.summary.scalar('min', tf.reduce_min(var))
    tf.summary.histogram('histogram', var)

#####################################################################################
###############            Initalize Params                 #########################
#####################################################################################
def init_params(graph_structure, classes=2):
  '''Initalizes all of the weights, biases, gating params, and reshaping params based
     on the graph definition given by graph_structure

  ARGS: graph_structure: a list of lists of tuples defining the graph_structure
  Example graph:
  graph_structure = [ [ ((2,2), identity_module) ],
                      [ ((2,2), identity_module), ((2,2), identity_module), ((3,3), identity_module) ],
                      [ ((2,2), identity_module), ((2,2), identity_module), ((3,3), identity_module) ],
                      [ ((2,2), identity_module), ((3,3), identity_module), ((4,4), identity_module) ],
                      [ ((2,2), identity_module), ((2,2), identity_module), ((3,3), identity_module) ],
                      [ ((2,2), identity_module) ]
                  ]
        classes: The number of classes to ouput in the final layer

  Notes: 
      - TODO: get working with 4D shapes

  Returns: weights_dict: A python dictionary with all of the needed params
  '''
  weights_dict = {}
  L = len(graph_structure)

  for l in range(1,L-1):
    prev_layer = graph_structure[l-1]
    current_layer = graph_structure[l]

    # Get shape shift weights and biases
    distinct_input_shapes  = list(set([module[0] for module in prev_layer]))
    distinct_output_shapes = list(set([module[0] for module in current_layer]))
    for input_shape in distinct_input_shapes:
      for output_shape in distinct_output_shapes:
        if input_shape != output_shape:
          # Get weights to go from shape (N, A) to shape (N, B), where N is batch size
          # Weights are shape (A, B), and biases shape (B)
          N, A = input_shape
          N, B = output_shape
          weights = weight_variable([A, B])
          biases = bias_variable([B,])
          weights_dict['shape_shift_weights_layer_'+ str(l-1) + '_to_' + str(l) + '_' + str(input_shape) + '_' + str(output_shape)] = weights
          weights_dict['shape_shift_biases_layer_' + str(l-1) + '_to_' + str(l) + '_' + str(input_shape) + '_' + str(output_shape)] = biases

    # Get the module parameters
    for m, module in enumerate(current_layer):
      shape, func = module
      # Get weights to go from shape (N, A) to shape (N, A), where N is batch size
      # Weights are shape (A, A), and biases shape (A)
      N, A = shape
      weights = weight_variable([A, A])
      biases  = bias_variable([A,])
      weights_dict['weights_'+ str(l) + '_' + str(m)] = weights
      weights_dict['biases_' + str(l) + '_' + str(m)] = biases

    # Get the gate parameters - very specific to the implementation of the current gating layer
    smallest_shape = prev_layer[0][0]
    for module in prev_layer:
      if module[0][1] < smallest_shape[1]:
        smallest_shape = module[0]
    # Get weights to go from shape (N, A) to shape (N, M), where N is batch size and M is the
    # number of modules in the current layer
    # Weights are shape (A, M), and biases shape (M)
    N, A = smallest_shape
    M = len(current_layer)
    weights = weight_variable([A, M])
    biases  = bias_variable([M,])
    weights_dict['weights_'+ str(l) + '_' + str(m)] = weights
    weights_dict['biases_' + str(l) + '_' + str(m)] = biases

  # Weights for the final output layer
  output_layer_shape = graph_structure[L-1][0][0]
  # Get weights to go from shape (N, A) to shape (N, C), where N is batch size, and C is the 
  # number of classes to output (eg binary mnist = 2)
  # Weights are shape (A, C), and biases shape (C)
  N, A = output_layer_shape
  output_weights = weight_variable([A, classes])
  output_biases  = bias_variable([classes,])
  weights_dict['output_weights_'+ str(L-1)] = output_weights
  weights_dict['output_biases_' + str(L-1)] = output_biases

  return weights_dict

#####################################################################################
###############             GATING LAYER                    #########################
#####################################################################################
def gating_layer(layer_input,
                 image_input,
                 layer_number,
                 gate_weights,
                 gate_biases,
                 gate_name=None,
                 prev_layer_structure=None,
                 gamma=1.333):

  '''Calculates the gating of the next layer based on the input image and previous layer
  ARGS: layer_input: A list of the outputs all of the modules in the previous layer, where
                     each is shape: [N, H_L, W_L, C_L] (where L is the layer where input tensor
                     is coming from, the list is length of M_pev (number of modules from 
                     the previous layer)
        image_input: The input images given to the network - [N, H, W, C]
        image_input: The number of the current layer
        gate_weights: The weights of the gates in the current layer (only one set 
                        of gate weights) - [H*W*C, M] (M is the modules in the current layer)
        gate_biases: The biases of the gates in the current layer - [M]
        gate_name : The name of the gates (for scoping)
        gamma (optional): The exponent term to use to encourage sparity

  Notes: 
      - Currently reshapes the outputs of the previous layer's modules into the shape of the
        smallest of the previous layer's module shapes
      - This solution (summing) is likely only good when the outputs of the modules are
        sparse, we should also consider other methods of gating


  Returns: gates: The gating outputs of the previous layer [M]
  '''

  # ONLY WORKS FOR 2D case - gets the smallest module shape in the layer
  smallest_shape = prev_layer_structure[0][0]
  for module in prev_layer_structure:
    if module[0][1] < smallest_shape[1]:
      smallest_shape = module[0]

  last_layer_module_summed = input_to_module(input_tensor,
                                             weights_dict,
                                             layer_number,
                                             prev_layer_structure,
                                             smallest_shape)

  # Adding a name scope ensures logical grouping of the layers in the graph.
  with tf.name_scope(gate_name):
    # This Variable will hold the state of the weights for the layer
    with tf.name_scope('gate_weights'):
      variable_summaries(gate_weights)
    with tf.name_scope('gate_biases'):
      variable_summaries(gate_biases)
    with tf.name_scope('gates_unnormalized'):
      gates_unnormalized = tf.matmul(last_layer_module_summed, gate_weights) + gate_biases
      variable_summaries(gates_unnormalized)
    with tf.name_scope('gates_normalized'):
      gates_pow = tf.pow(gates_unnormalized, gamma)
      gates_normalized = gates_pow / tf.reduce_sum(gates_pow)
      variable_summaries(gates_normalized)

  return tf.nn.softmax(gates_normalized)


#####################################################################################
###############           Function types                    #########################
#####################################################################################

def identity_module(input_tensor, weights, biases, act):
  return input_tensor

def perceptron_module(input_tensor, weights, biases, act):
  return act(tf.matmul(input_tensor, weights) + biases)

def residual_perceptron_module(input_tensor, weights, biases, act):
  return act(tf.matmul(input_tensor, weights) + biases) + input_tensor

def conv_module(input_tensor, weights, biases, act, params=None):
  if params is not None:
    strides = params.get('strides')
    padding = params.get('padding')
  else:
    strides = [1,1,1,1]
    padding = "SAME"
  return act(tf.nn.conv2d(input_tensor, weights, strides=strides, padding=padding))

def residual_conv_module(input_tensor, weights, biases, act, params=None):
  if params is not None:
    strides = params.get('strides')
    padding = params.get('padding')
  else:
    strides = [1,1,1,1]
    padding = "SAME"
  return act(tf.nn.conv2d(input_tensor, weights, strides=strides, padding=padding)) + input_tensor

#####################################################################################
###############                  MODULES                    #########################
#####################################################################################
def module(input_tensor,
           weights,
           biases,
           module_name=None,
           act=tf.nn.relu,
           func=perceptron_module):
  ''' ***This is for modules in the first layer of the net
     Returns the module's calculation, module can be a simple function, or a 
     traditional NN layer, or even a NN itself
     Here it's the equivalent of a traditional NN layer

  ARGS: input_tensor  : The reshaped and summer tensor output from the previous layer,
                        shape [N,H,W,C]
        weights       : The weights for the module (only the current module's weights)
        biases        : The biases for the module
        module_name (optional) : Name for scoping
        act(optional) : Activation function
  
  Returns: The modules output (calculation of the function)
  '''

  # Adding a name scope ensures logical grouping of the layers in the graph.
  with tf.name_scope(module_name):
    # This Variable will hold the state of the weights for the layer
    with tf.name_scope('weights'):
      variable_summaries(weights)
    with tf.name_scope('biases'):
      variable_summaries(biases)
    activations = func(input_tensor, weights, biases, act)
    tf.summary.histogram('activations', activations)
  return activations

#####################################################################################
###############                 SUB LAYERS                  #########################
#####################################################################################
def input_to_module(input_tensor,
                    weights_dict,
                    layer_number,
                    prev_layer_structure,
                    output_shape):
  '''Reshapes (w convs) and sums up the previous layer's modules from the input_tensor
     and returns the sum that will feed into the next layer's modules

  ARGS: input_tensor: A list of tensors that contain the outputs of the previous 
                      layer's modules, shape: [M, N, H, W, C]
        weights_dict: The weights and biases (used for the reshaping convolutions) 
        layer_number: The current layer number
                    - the number of weights/biases = (n - 1) * n, n = number of sublayers
        prev_layer_structure: A list of shapes from the previous layer
        output_shape: The wanted shape for the output

  Returns: output_tensor: Reshaped to the desired output_shape and then  summed up into
                          one tensor [N, H_n, W_n, C_n], where _n corresponds the 
                          desired output shape
  '''
  tensor_to_sum = []
  for i, module in enumerate(prev_layer_structure):
    if module[0] == output_shape:
      tensor_to_sum.append(input_tensor[i])
    else:
      shape_shift_weights = weights_dict['shape_shift_weights_layer_'+ str(layer_number) + '_to_' + str(layer_number-1) + '_' + str(shape) + '_' + str(output_shape)]
      shape_shift_biases  = weights_dict['shape_shift_biases_layer_' + str(layer_number) + '_to_' + str(layer_number-1) + '_' + str(shape) + '_' + str(output_shape)]
      tensor_to_sum.append(
          reshape_connection(input_tensor[i], shape_shift_weights, shape_shift_biases)
      )
  output_tensor = np.sum(tensor_to_sum) # TODO Make sure this sum works
  return output_tensor

def reshape_connection(input_tensor,
                       weights,
                       biases,
                       func=perceptron_module,
                       func_params=None, # CHANGE TO SOMETHING VALID
                       output_shape=None):
  '''Reshapes a tensor into the the desired shape 

  ARGS: input_tensor: The tensor you want to reshape using func (usually conv)
        weights     : The weights for the reshape (usually conv weights)
        biases      : The biases
        func        : A lambda function of what function you want to use to reshape
                      the input tensor (conv, perceptron, etc.)
        func_params : (opt) The params for a conv/other (put into lambda func???)
        output_shape: (opt) If no func params are given then automatically infer 
                      from output_shape

  Returns: The tensor reshaped using func
  '''

  # The following is for the 4d case
  '''
  if func_params is None:
    if output_shape is None:
      raise Exception('Function: reshape_connection, need one of func_params or output_shape')
    ## ---------------------------------------------------------------
    ## TODO implement getting func_params from the output_shape
    func_params = [0] # NOT IMPLEMENTED
    raise Exception('Not implemented in function: reshape_connection')
    ## ---------------------------------------------------------------
  '''
  return func(input_tensor, weights, biases, tf.nn.relu)


#####################################################################################
###############                  LAYERS                     #########################
#####################################################################################
def layer(input_tensor,
          input_image,
          weights_dict,
          layer_number,
          prev_layer_structure=None,
          current_layer_structure=None):
  '''Returns the output of a layer of modules

  ARGS: input_tensor: The input to the layer, should be list of tensors each of shape
                      [N, H_i, W_i, C_i] (_i means elements in list don't have to match)
        input_image : The original input image to the network, shape [N,H,W,C] of image
        weights_dict: A dictionary with the weights and biases for the whole network,
                      we only use a subset of these (the ones for the current layer)
        layer_number: the current layer number

        NOT FULLY IMPLEMENTED: *SUB-LAYERS*
          prev_layer_structure   : The shapes of all the modules of the previous layer
          current_layer_structure: The shapes of all the modules in the current layer


  Returns: output_tensor: The outputs of all of the tensors in the previous layer
  '''
  M = len(current_layer_structure)

  # Get a list of weights, biases, etc for the current layer from the weights_dict
  weights_list = [weights_dict['weights_'+ str(layer_number) + '_' + module] for module in range(M)]
  biases_list  = [weights_dict['biases_' + str(layer_number) + '_' + module] for module in range(M)]
  gate_weights = weights_dict['gate_weights_'+ str(layer_number)]
  gate_biases  = weights_dict['gate_biases_' + str(layer_number)]

  # Get the gating values for the layer
  gates = gating_layer(input_tensor, input_image, layer_number, 
                       gate_weights, gate_biases, gate_name='gate_'+str(layer_number), 
                       prev_layer_structure=prev_layer_structure, gamma=1)

  tensor_output = []
  input_to_shape = {}
  for i, current_module in enumerate(current_layer_structure):
    module_shape, module_func = current_module

    # If there are multiple modules that need the same inputs
    if input_to_shape.get(module_shape) is None:
      # Get the input of the module into the right shape (reshape to proper shape and average over previous module outputs)
      input_to_shape[module_shape] = input_to_module(input_tensor, weights_dict, layer_number, prev_layer_structure, module_shape)

    # Multiply gate and module values
    module = gates[i] * module(input_to_shape[module_shape],
                               weights_list[i],
                               biases_list[i],
                               module_name='module_'+str(layer_number)+'_'+str(i),
                               act=tf.nn.relu,
                               func=module_func)
    tensor_output.append(module)
  return (tensor_output, gates)

#####################################################################################
###############               BUILD GRAPH                   #########################
#####################################################################################
def build_pathnet_graph(X, weights_dict, graph_structure):
  '''builds the graph from the variable initialization function and returns the predictions
  ARGS: X: The input image in the shape of [N, H, W, C]
        weights_dict: Dictionary with the keys being the names of the weights
        graph_structure: Describes the modules and shapes of the graph, (all of the modules,
                         shapes, etc.) (First el is the input shape)

  Returns: logits: The predictions of the network, same shape as y (batch size N by classes)
           gates : The gate activations for each layer: gate[l] = 1xM tensor with gates for the layer
  '''
  next_tensor_input = X
  L = len(graph_structure)

  # Save the gate outputs
  gates = []

  # Build graph up to the last hidden layer
  for l in range(1,L-1):
    next_tensor_input, layer_gates = layer(next_tensor_input, X, weights_dict, l,
                                           prev_layer_structure=graph_structure[l-1],
                                           current_layer_structure=graph_structure[l])
    gates.append(layer_gates)

  # The final layer that combines all of the modules
  output_shape = graph_structure[L-1][0][0]
  output_func = graph_structure[L-1][0][1]
  # Get the average for the last layer's modules
  last_input = input_to_module(next_tensor_input, weights_dict, L, graph_structure[L-2], output_shape)
  output_weights = weights_dict['output_weights_'+ str(L-1)]
  output_biases  = weights_dict['output_biases_' + str(L-1)]
  logits = module(last_input, output_weights, output_biases, layer_name='output_layer', act=tf.nn.relu, func=output_func)

  return (logits, gates)


