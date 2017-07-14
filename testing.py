from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pickle
# For module and submodule type
from graph.sublayer import *
from graph.module import *

class Parameters:
  def __init__(self):
    '''The defaults for creating a gatenet graph

    All three below create same graph:

    1. Using graph_structure:
            self.graph_structure = [[ConvModule, ConvModule],
                                    [ConvModule, ConvModule],
                                    [ConvModule, ConvModule]]

    2. Using layer_structure:
            self.L = 3
            self.layer_structure = [ConvModule, ConvModule]
    
    3. Using module_type:
            self.L = 3
            self.M = 2
            self.module_type = ConvModule
    '''
    self.C = 10 # MNIST digits
    self.sublayer_type = AdditionSublayerModule
    self.hidden_size = 2 # Hidden size or # of conv filters
    self.gamma = 2
    self.batch_size = 100
    self.num_batches = 201
    self.learning_rate = 0.001
    self.output_file = 'test'

    self.M = 2
    self.L = 3
    self.module_type = ConvModule


class OutputManager:
  def __init__(self):
    self.output_file = None
    self.parameter_dict = None
    self.data = None
    self.labels = None
    self.images = None


  def initialize(self, parameter_dict):
    self.parameter_dict = parameter_dict
    self.data = []
    self.labels = []
    self.images = []

  def addData(self, data, label, image):
    self.data.append(data)
    self.labels.append(label)
    self.images.append(image)

  def save(self):
    output = (self.data, self.labels, self.images, self.parameter_dict)
    with open('output/' + self.parameter_dict['output_file'] + '.pkl', 'wb') as f:
      pickle.dump(output, f , pickle.HIGHEST_PROTOCOL)
