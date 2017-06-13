#Test
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pathnet
from parameters import Parameters
import data_manager
import numpy as np
import time
import tensorflow as tf

from pathnet import *


def train():
  # Import data
  tr_data1, tr_label1 = data_manager.get_next_batch()

  sess = tf.InteractiveSession()

  # Input placeholders
  with tf.name_scope('input'):
    x = tf.placeholder(tf.float32, [None, 784], name='x-input')
    y_ = tf.placeholder(tf.float32, [None, 2], name='y-input')

  with tf.name_scope('input_reshape'):
    image_shaped_input = tf.reshape(x, [-1, 28, 28, 1])
    tf.summary.image('input', image_shaped_input, 2)



  graph_structure = []
  weights_dict = {}

  # FILL IN
  tensor_sizes = {}
  graph_io_sizes = {}
  num_sublayers = 3

  ############################################ New Code


      # Hidden Layers
  for i in range(Parameters.L):
    for j in range(Parameters.M):
      if (i == 0):
        tensor_sizes['weights_' + str(i) + '_' + str(j)] = [28*28, 20]
        tensor_sizes['gate_weights_' + str(i) + '_' + str(j)] = [28*28, 20]
      else:
        tensor_sizes['weights_' + str(i) + '_' + str(j)] = [20,20]
        tensor_sizes['gate_weights_' + str(i) + '_' + str(j)] = [20,20]
      tensor_sizes['biases_' + str(i) + '_' + str(j)] = [20]
      tensor_sizes['gate_biases_' + str(i) + '_' + str(j)] = [20]

    for s1 in range(num_sublayers):
      for s2 in range(num_sublayers):
        tensor_sizes['shape_shift_weights_' + str(i) + '_' + str(s1) + '_' + str(s2)] = [20,20]
        tensor_sizes['shape_shift_biases_' + str(i) + '_' + str(s1) + '_' + str(s2)] = [20]



      graph_structure = [[((2, 2), identity_module)],
                         [((2, 2), identity_module), ((2, 2), identity_module), ((3, 3), identity_module)],
                         [((2, 2), identity_module), ((2, 2), identity_module), ((3, 3), identity_module)],
                         [((2, 2), identity_module), ((3, 3), identity_module), ((4, 4), identity_module)],
                         [((2, 2), identity_module), ((2, 2), identity_module), ((3, 3), identity_module)],
                         [((2, 2), identity_module)]
                         ]

  ############################################




      # Hidden Layers
  for i in range(Parameters.L):
    for j in range(Parameters.M):
          weights_dict['weights_' + str(i) + '_' + str(j)] = pathnet.weight_variable(tensor_sizes['weights_' + str(i) + '_' + str(j)])
          weights_dict['biases_' + str(i) + '_' + str(j)] = pathnet.bias_variable(tensor_sizes['biases_' + str(i) + '_' + str(j)])
          weights_dict['gate_weights_' + str(i) + '_' + str(j)] = pathnet.weight_variable(tensor_sizes['gate_weights_' + str(i) + '_' + str(j)])
          weights_dict['gate_biases_' + str(i) + '_' + str(j)] = pathnet.weight_variable(tensor_sizes['gate_biases_' + str(i) + '_' + str(j)])

    for s1 in range(num_sublayers):
        for s2 in range(num_sublayers):
            weights_dict['shape_shift_weights_' + str(i) + '_' + str(s1) + '_' + str(s2)]\
                = pathnet.weight_variable(tensor_sizes['shape_shift_weights_' + str(i) + '_' + str(s1) + '_' + str(s2)])
            weights_dict['shape_shift_biases_' + str(i) + '_' + str(s1) + '_' + str(s2)] \
                = pathnet.bias_variable(
                tensor_sizes['shape_shift_biases_' + str(i) + '_' + str(s1) + '_' + str(s2)])

  '''
    graph_structure.append((graph_io_sizes[(0,0)], pathnet.identity_module))
    for i in range(Parameters.L): # Change indexing
      for j in range(Parameters.M):
          graph_structure.append((graph_io_sizes[(i,j)], pathnet.identity_module))
  
    graph_structure.append((graph_io_sizes[(Parameters.L, 0)], pathnet.identity_module))
  '''

  y, gates = build_pathnet_graph(x, weights_dict, graph_structure)

  # Cross Entropy
  with tf.name_scope('cross_entropy'):
    diff = tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y)
    with tf.name_scope('total'):
      cross_entropy = tf.reduce_mean(diff)
  tf.summary.scalar('cross_entropy', cross_entropy)

  # GradientDescent 
  with tf.name_scope('train'):
    train_step = tf.train.GradientDescentOptimizer(Parameters.learning_rate).minimize(cross_entropy);

  # Accuracy 
  with tf.name_scope('accuracy'):
    with tf.name_scope('correct_prediction'):
      correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    with tf.name_scope('accuracy'):
      accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  tf.summary.scalar('accuracy', accuracy)

  # Merge all the summaries and write them out to /tmp/tensorflow/mnist/logs/mnist_with_summaries (by default)
  merged = tf.summary.merge_all()
  train_writer = tf.summary.FileWriter(Parameters.log_dir + '/train1', sess.graph)
  test_writer = tf.summary.FileWriter(Parameters.log_dir + '/test1')
  tf.global_variables_initializer().run()



  idx = list(range(len(tr_data1)));
  np.random.shuffle(idx);
  tr_data1 = tr_data1[idx];
  tr_label1 = tr_label1[idx];

  for i in range(1000):
    summary_geo_tr, _, acc = sess.run([merged, train_step, accuracy], feed_dict={
      x: tr_data1,
      y_: tr_label1});


    print('Training Accuracy at step %s: %s' % (i, acc));
    '''if (acc >= 0.99):
      print('Learning Done!!');
      print('Optimal Path is as followed.');
      break;
      '''


def main(_):
  Parameters.log_dir+=str(int(time.time()));
  if tf.gfile.Exists(Parameters.log_dir):
    tf.gfile.DeleteRecursively(Parameters.log_dir)
  tf.gfile.MakeDirs(Parameters.log_dir)
  train()


if __name__ == '__main__':
  tf.app.run(main=main)
