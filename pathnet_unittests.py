from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import tensorflow as tf
import numpy as np
import sys
from pathnet import *


### All of the unit tests for pathnet
class ModuleFunctionsTest(tf.test.TestCase):

	def unit_test_module_functions(self):
	''' Tests all of the lambda functions that are inputed into the module function
	Includes: 
						- Identity module (2d and 4d)
						- Perceptron module (2d only)
						- Residual Perceptron module (2d only)
	TODO:
						- Conv module (4d only)
						- Residual Conv module (4d only)

	'''
  # Define inputs
  act = tf.nn.relu
  input_2d = np.array([[ 5.21725234,  7.86804288,  7.99242374,  4.45774655, -5.50443247], # Shape (4,5)
       				     		 [ 5.07227163,  0.67308993, -3.9649117 ,  9.85479189,  9.05849196],
       					 			 [-7.78440314,  6.60532817,  7.31782079, -1.52107527,  4.55220238],
       					 			 [ 7.70451637,  9.13639315,  6.29414844,  0.37114561, -7.22741304]])
  input_4d = np.array([[[[ 6.79197047,  5.64171543], # Shape (2,2,3,2)
         								 [ 4.04310649,  6.54305578],
         							   [ 5.96901693,  4.42749864]],
        						    [[ 0.08351978,  2.41839058],
         							   [ 9.48140987,  5.68764355],
         							   [ 0.05218624,  5.71626734]]],

       						     [[[ 5.17492141,  8.64053361],
         						     [ 6.39615003,  4.93051958],
         						     [ 3.84808429,  8.98262799]],
        						    [[ 4.17638444,  4.81738749],
         						     [ 9.17608388,  7.17350660],
         						     [ 4.75977022,  2.8040811]]]])
  weights_2d = np.eye(4,5)
  biases_2d  = np.ones([1,5])
 
  # Identity module
	with self.test_session():
  	identity_module_output_2d = identity_module(input_2d, weights_2d, biases_2d)
  	identity_module_output_4d = identity_module(input_4d)
		self.assertAllEqual(identity_module_output_2d.eval(), input_2d)
		self.assertAllEqual(identity_module_output_4d.eval(), input_4d)
  	#print('identity_module works for 2d case: ', identity_module_output_2d.eval() == input_2d)
  	#print('identity_module works for 4d case: ', identity_module_output_4d.eval() == input_4d)

  # Perceptron Module
  correct_perceptron_output_2d = np.array( # Shape (4,5) - like input
  					[[6.217252340,  8.86804288,  8.99242374,  5.457746550,  0.000000000],
    				 [6.072271630,  1.67308993,  0.00000000,  10.85479189,  10.05849196],
        		 [0.000000000,  7.60532817,  8.31782079,  0.000000000,  5.552202380], 
         		 [8.70451637,  10.13639315,  7.29414844,  1.371145610,  0.000000000]])
	with self.test_session():
  	perceptron_module_output_2d = perceptron_module(input_2d)
		self.assertAllEqual(perceptron_module_output_2d.eval(), correct_perceptron_output_2d)
  	#print('perceptron_module works for 2d case: ', perceptron_module_output_2d.eval() == correct_perceptron_output_2d)

  # Residual Perceptron Module
  correct_residual_perceptron_output_2d = correct_perceptron_output_2d + input_2d
	with self.test_session():
  	residual_perceptron_module_output_2d = residual_perceptron_module(input_2d)
		self.assertAllEqual(residual_perceptron_module_output_2d.eval(), correct_residual_perceptron_output_2d)
  	#print('residual_perceptron_module works for 2d case: ', residual_perceptron_module_output_2d.eval() == correct_residual_perceptron_output_2d)



class ModuleUnit(tf.test.TestCase):

	def module_unit_test(self):

  	# 2d test
  	input_2d = np.array([[ 5.21725234,  7.86804288,  7.99242374,  4.45774655, -5.50443247], # Shape (4,5)
       				     		   [ 5.07227163,  0.67308993, -3.9649117 ,  9.85479189,  9.05849196],
       					 			   [-7.78440314,  6.60532817,  7.31782079, -1.52107527,  4.55220238],
       					 			   [ 7.70451637,  9.13639315,  6.29414844,  0.37114561, -7.22741304]])
  	weights_2d = np.eye(4,5)
  	biases_2d  = np.ones([1,5])
  	correct_module_output_2d = np.array( # Shape (4,5) - like input
  					[[6.217252340,  8.86804288,  8.99242374,  5.457746550,  0.000000000],
    				 [6.072271630,  1.67308993,  0.00000000,  10.85479189,  10.05849196],
        		 [0.000000000,  7.60532817,  8.31782079,  0.000000000,  5.552202380], 
         		 [8.70451637,  10.13639315,  7.29414844,  1.371145610,  0.000000000]])

 	  with self.test_session():
  	  # use default act and func: tf.nn.relu and perceptron_module
  		module_output_2d = module(input_2d, weights, biases)
		  self.assertAllEqual(module_output_2d.eval(), correct_module_output_2d)
		  #print('module works for 2d case: ', module_output_2d.eval() == correct_module_output_2d)



 graph_structure = [ [ ((2,2), identity_module) ],
                     [ ((2,2), identity_module), ((2,2), identity_module), ((3,3), identity_module) ],
                     [ ((2,2), identity_module), ((2,2), identity_module), ((3,3), identity_module) ],
                     [ ((2,2), identity_module), ((3,3), identity_module), ((4,4), identity_module) ],
                     [ ((2,2), identity_module), ((2,2), identity_module), ((3,3), identity_module) ],
                     [ ((2,2), identity_module) ]
                  ]

if __name__ == '__main__':
    tf.test.main()

