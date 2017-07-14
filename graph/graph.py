import tensorflow as tf
import numpy as np
#from parameters import Parameters
from tensorflow_utils import *

from graph.layer import *
from graph.sublayer import *
from graph.module import *

#from matplotlib import pyplot as plt

##############################################################################
## Code for Gatenet Graph construction
class Graph():

    def __init__(self, parameter_dict):
        self.gated_layers = None
        self.output_layer = None

        ## Define Hyperparameters
        self.M = parameter_dict['M'] # Number of modules per layer
        self.L = parameter_dict['L'] # Number of layers
        self.C = parameter_dict['C'] # Number of output classes
        self.gamma = parameter_dict['gamma'] # Strength of gating
        self.module_type = parameter_dict['module_type']
        self.sublayer_type = parameter_dict['sublayer_type']

        self.hidden_size = parameter_dict['hidden_size']

        # Build new graph
        self.build_graph()


    def build_graph(self):
        ## Define Layers #####################################################
        self.gated_layers = []
        for i in range(self.L):
            with tf.variable_scope('gated_layer_'+str(i)):
                gated_layer_defn = {'M': self.M,
                                    'hidden_size': self.hidden_size,
                                    'module_type': self.module_type,
                                    'sublayer_type': self.sublayer_type}
                gated_layer = GatedLayer(gated_layer_defn, gamma=self.gamma)
                self.gated_layers.append(gated_layer)
            with tf.variable_scope('output_layer'):
                self.output_layer = OutputLayer(C=self.C)
        ######################################################################


    def return_logits(self, input_images):
        '''Return the output of graph based off of input_images'''
        ## Construct graph ###################################################
        next_input = input_images
        for i in range(self.L):
            next_input = self.gated_layers[i].process_layer(next_input)
        logits = self.output_layer.process_layer(next_input)
        ######################################################################
        return logits


    def determine_gates(self, image, x, sess):
        '''Output the values of the gates for image'''
        gates = np.zeros((self.L, self.M))
        for i in range(len(self.gated_layers)):
            im = np.reshape(image, [28, 28])
            #plt.imshow(im)
            #plt.show()
            g = sess.run([self.gated_layers[i].gates],
                         feed_dict={x:image})
            gates[i] = np.array(g)
        return gates
##############################################################################
