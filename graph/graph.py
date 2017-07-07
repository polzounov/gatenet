import tensorflow as tf
import numpy as np
#from parameters import Parameters
from tensorflow_utils import *

from graph.layer import *
from graph.sublayer import *
from graph.module import *

######################################################################
## Code for Graph construction
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
        self.tensor_size = parameter_dict['tensor_size']
        self.gamma = parameter_dict['gamma']        

        ## Define Layers #################################################################
        input_layer_defn = {'M': self.M,
                            'input_size': 784,
                            'module_output_size': self.tensor_size,
                            'module_type': PerceptronModule,
                            'sublayer_type': ConcatenationSublayerModule}
        self.input_layer = InputLayer(input_layer_defn)

        prev_output_size = self.input_layer.layer_output_size
        #prev_output_size = 784 # Image size
        self.gated_layers = []
        for i in range(self.L):
            gated_layer_defn = {'M': self.M,
                                'input_size': prev_output_size,
                                'module_output_size': self.tensor_size,
                                'module_type': PerceptronModule,
                                'sublayer_type': ConcatenationSublayerModule}
            gated_layer = GatedLayer(gated_layer_defn, self.gamma)
            prev_output_size = gated_layer.layer_output_size

            self.gated_layers.append(gated_layer)

        output_layer_defn = {'M': 1,
                             'input_size': prev_output_size,
                             'module_output_size': 10,
                             'module_type': LinearModule,
                             'sublayer_type': AdditionSublayerModule}
        self.output_layer = OutputLayer(output_layer_defn)
        ##################################################################################


    ## Build graph to test code
    def return_logits(self, input_images):
        ## Construct graph ###############################
        layer_output = self.input_layer.process_layer(input_images)
        for i in range(self.L):
            layer_output = self.gated_layers[i].process_layer(layer_output)
        logits = self.output_layer.process_layer(layer_output)
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





######################################################################
'''
    GRAPH API DOCUMENTATION:
        • Takes in:
            -  

    SUBLAYER API DOC:
        • Takes in:
            - Number of modules

    LAYER API DOCUMENTATION:
        • Takes in:
            - Input size
            - Layer definition (future: [M0 M1 M3 ... Mn])
                - Number of modules
                - Type of module (only use one type for now)
            - Sublayer Types (future is several)







































''' 
