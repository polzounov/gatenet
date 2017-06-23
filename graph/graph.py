import tensorflow as tf
import numpy as np
#from parameters import Parameters
from tensorflow_utils import *

from graph.layer import *
from graph.sublayer import *
from graph.module import *

######################################################################
## Code for Graph construction
class Graph:

    def __init__(self):
        self.input_layer = None
        self.gated_layers = None
        self.output_layer = None
        self.sublayers = None
        
    ## Build graph to test code
    def buildTestGraph(self, input_images, parameter_dict):

        ## Define Hyperparameters ########################
        self.M = parameter_dict['M']
        self.L = parameter_dict['L']
        self.tensor_size = parameter_dict['tensor_size']
        self.gamma = parameter_dict['gamma']
        ##################################################


        ## Define Sublayers ##############################
        self.sublayers = np.zeros(self.L + 1, dtype= object)
        input_size = self.tensor_size
        num_modules = self.M
        for i in range(self.L + 1):
            self.sublayers[i] = Sublayer(input_size, num_modules, ConcatenationSublayerModule(input_size,num_modules))
        ##################################################


        ## Define Modules ################################
        input_modules = np.zeros(self.M, dtype=object)
        gated_modules = np.zeros((self.L, self.M), dtype=object)
        output_modules = np.zeros(1, dtype=object)

        print(self.sublayers[-1].output_size)

        for i in range(self.M):
            input_modules[i] = PerceptronModule(weight_variable([784, self.sublayers[0].input_size]),
                                                bias_variable([self.sublayers[0].input_size]), activation=tf.nn.relu)
            
            for j in range(self.L):
                gated_modules[j][i] = PerceptronModule(weight_variable([self.sublayers[j].output_size, self.sublayers[j+1].input_size]),
                                                       bias_variable([self.sublayers[j+1].input_size]), activation=tf.nn.relu)
        output_modules[0] = LinearModule(weight_variable([self.sublayers[-1].output_size, 10]), bias_variable([10]))
        ##################################################


        ## Define Layers #################################
        self.input_layer = InputLayer(input_modules)
        self.gated_layers = []
        for i in range(self.L):
            gated_layer = GatedLayer(gated_modules[i], self.sublayers[i].output_size,  self.gamma)
            self.gated_layers.append(gated_layer)

        self.output_layer = OutputLayer(output_modules)
        ##################################################


        ## Construct graph ###############################
        layer_output = self.input_layer.processLayer(input_images)
        sublayer_output = self.sublayers[0].processSublayer(layer_output)
        for i in range(self.L):
            layer_output = self.gated_layers[i].processLayer(sublayer_output)
            sublayer_output = self.sublayers[i+1].processSublayer(layer_output)

        logits = self.output_layer.processLayer(sublayer_output)
        ##################################################
        return logits


    def determineGates(self, image, x, sess):
        gates = np.zeros((self.L, self.M))
        for i in range(len(self.gated_layers)):
            g = sess.run([self.gated_layers[i].gates],
                         feed_dict={x:image})
            gates[i] = np.array(g)
        return gates
######################################################################