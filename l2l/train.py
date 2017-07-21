from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from sklearn.neighbors import KNeighborsClassifier

from graph.graph import Graph
import l2l.metaoptimizer
from tensorflow_utils import *
from testing import *

def train(parameter_dict=None, skip_digits=[7,8], num_gate_vectors_output=100):
  if parameter_dict is None:
    print('Use test params')
    parameter_dict = Parameters().__dict__

  mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
  test_images = mnist.test.images[:1000]
  test_labels = mnist.test.labels[:1000]

  # Probably switch to Session rather than InteractiveSession later on
  # Start session
  sess = tf.InteractiveSession()

  # Input placeholders
  x = tf.placeholder(tf.float32, [None, 28, 28, 1], name='x-input')
  y_ = tf.placeholder(tf.float32, [None, 10], name='y-input')

  # Build computation graph
  graph = Graph(parameter_dict)
  y = graph.return_logits(x)

  with tf.name_scope('softmax_predictions'):
    predictions = tf.nn.softmax(y)
  # Cross Entropy
  with tf.name_scope('cross_entropy'):
    diff = tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y)
    cross_entropy = tf.reduce_mean(diff)

  # GradientDescent
  train_step = metaoptimizer.optimizer(optimizer_type='Adam',
                                       lr=parameter_dict['learning_rate'],
                                       loss_func=cross_entropy)

  # Accuracy
  with tf.name_scope('accuracy'):
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

  # Initialize Variables
  tf.global_variables_initializer().run()

  writer = tf.summary.FileWriter('./logs', graph=tf.get_default_graph())
  #Command to run: tensorboard --logdir=logs

  for i in range(parameter_dict['num_batches']):
    tr_data, tr_label = mnist.train.next_batch(parameter_dict['batch_size'])

    # Get the indices of elements in skip list
    elems_list = []
    for skip_digit in skip_digits:
      elems = (np.where(np.argmax(tr_label, axis=1) == skip_digit)[0])
      elems_list.append(elems)
    elems = np.sort(np.concatenate(elems_list, axis=0))
    elems = np.delete(np.arange(tr_label.shape[0]), elems) # Invert elems

    tr_data = tr_data[elems,:]
    tr_label = tr_label[elems,:]

    tr_data = np.reshape(tr_data, (-1,28,28,1))

    if len(tr_data) < 50:
      continue

    if i % 10 == 0:
      acc = sess.run(accuracy, feed_dict={x: tr_data, y_: tr_label})
      print('training %d, accuracy %g' % (i, acc))

    acc = sess.run(accuracy, feed_dict={x: tr_data, y_: tr_label})
    sess.run(train_step, feed_dict={x: tr_data, y_: tr_label})


def main(_):
  train(skip_digits=[0], num_gate_vectors_output=1000)


if __name__ == '__main__':
  tf.app.run(main=main)
