import tensorflow as tf
import numpy as np
#from parameters import Parameters
from tensorflow_utils import *

from graph.layer import *
from graph.sublayer import *
from graph.module import *

######################################################################
## Code for Gatenet Graph construction
class Graph():

    def __init__(self, parameter_dict):
        self.input_layer = None
        self.gated_layers = None
        self.output_layer = None
        self.sublayers = None

        # Build new graph from param dict
        self.build_graph(parameter_dict)

    ## Build graph to test code
    def build_graph(self, parameter_dict):

        ## Define Hyperparameters #########################################################
        self.M = parameter_dict['M']
        self.L = parameter_dict['L']
        self.tensor_shape = parameter_dict['tensor_shape']
        self.gamma = parameter_dict['gamma']        

        ## Define Layers #################################################################
        prev_output_shape = 784 # Image size (first input)
        self.gated_layers = []
        for i in range(self.L):
            gated_layer_defn = {'M': self.M,
                                'input_shape': prev_output_shape,
                                'module_output_shape': self.tensor_shape,
                                'module_type': PerceptronModule,
                                'sublayer_type': ConcatenationSublayerModule}
            gated_layer = GatedLayer(gated_layer_defn, self.gamma)
            prev_output_shape = gated_layer.layer_output_shape

            self.gated_layers.append(gated_layer)

        output_layer_defn = {'M': 1,
                             'input_shape': prev_output_shape,
                             'module_output_shape': 10,
                             'module_type': LinearModule,
                             'sublayer_type': AdditionSublayerModule}
        self.output_layer = GatedLayer(output_layer_defn, gamma=0)
        ##################################################################################


    ## Build graph to test code
    def return_logits(self, input_images):
        ## Construct graph ###############################
        next_input = input_images
        for i in range(self.L):
            next_input = self.gated_layers[i].process_layer(next_input)
        logits = self.output_layer.process_layer(next_input)
        ##################################################
        return logits


    def determine_gates(self, image, x, sess):
        gates = np.zeros((self.L, self.M))
        for i in range(len(self.gated_layers)):
            g = sess.run([self.gated_layers[i].gates],
                         feed_dict={x:image})
            gates[i] = np.array(g)
        return gates
######################################################################
