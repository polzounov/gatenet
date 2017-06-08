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


      # Hidden Layers
  weights_list = np.zeros((Parameters.L, Parameters.M), dtype=object);
  biases_list = np.zeros((Parameters.L, Parameters.M), dtype=object);
  for i in range(Parameters.L):
    for j in range(Parameters.M):
      if (i == 0):
        weights_list[i, j] = pathnet.module_weight_variable([784, Parameters.filt]);
        biases_list[i, j] = pathnet.module_bias_variable([Parameters.filt]);
      else:
        weights_list[i, j] = pathnet.module_weight_variable([Parameters.filt, Parameters.filt]);
        biases_list[i, j] = pathnet.module_bias_variable([Parameters.filt]);

  for i in range(Parameters.L):
    layer_modules_list = np.zeros(Parameters.M, dtype=object);
    for j in range(Parameters.M):
      if (i == 0):
        layer_modules_list[j] = pathnet.module(x, weights_list[i, j], biases_list[i, j],
                                               'layer' + str(i + 1) + "_" + str(j + 1))  # *geopath[i,j];
      else:
        layer_modules_list[j] = pathnet.module2(j, net, weights_list[i, j], biases_list[i, j],
                                                'layer' + str(i + 1) + "_" + str(j + 1))  # *geopath[i,j];
    net = np.sum(layer_modules_list) / Parameters.M;


  output_weights = pathnet.module_weight_variable([Parameters.filt, 2]);
  output_biases = pathnet.module_bias_variable([2]);
  y = pathnet.nn_layer(net, output_weights, output_biases, 'output_layer');

  # Cross Entropy
  with tf.name_scope('cross_entropy'):
    diff = tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y)
    with tf.name_scope('total'):
      cross_entropy = tf.reduce_mean(diff)
  tf.summary.scalar('cross_entropy', cross_entropy)

  # Need to learn variables
  var_list_to_learn = [] + output_weights + output_biases;
  for i in range(Parameters.L):
    for j in range(Parameters.M):
      var_list_to_learn += weights_list[i, j] + biases_list[i, j];

  # GradientDescent 
  with tf.name_scope('train'):
    train_step = tf.train.GradientDescentOptimizer(Parameters.learning_rate).minimize(cross_entropy,
                                                                                 var_list=var_list_to_learn);

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
