from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
sns.set_style("dark")

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)


from layers import *

from parameters import Parameters

import data_manager

from tensorflow_utils import *

def train():
  # Start session
  sess = tf.InteractiveSession()

  use_prebuilt_model = 0
  #if use_prebuilt_model != 1:

  # Import data
  tr_data, tr_label = data_manager.get_next_batch()



  # Input placeholders
  x = tf.placeholder(tf.float32, [None, 784], name='x-input')
  y_ = tf.placeholder(tf.float32, [None, 10], name='y-input')

  # Build computation graph
  graph = Graph()
  y = graph.buildTestGraph(x)


  # Cross Entropy
  diff = tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y)
  cross_entropy = tf.reduce_mean(diff)

  # GradientDescent
  train_step = tf.train.AdamOptimizer().minimize(cross_entropy)

  # Accuracy
  correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

  # Initialize Variables
  tf.global_variables_initializer().run()

  writer = tf.summary.FileWriter('/tmp/tensorflow_logs', graph=tf.get_default_graph())

  for i in range(10000):
    tr_data, tr_label = mnist.train.next_batch(100)

    if i % 100 == 0:
      acc = sess.run(accuracy, feed_dict={x: tr_data, y_: tr_label})
      print('training %d, accuracy %g' % (i, acc))

    sess.run(train_step, feed_dict={x: tr_data, y_: tr_label})


  #saver = tf.train.Saver()
  # Now, save the graph
  #saver.save(sess, 'C:\Gatenet\my_test_model', global_step=1000)

  #saver = tf.train.import_meta_graph('C:\Gatenet\my_test_model-1000.meta')
  #saver.restore(sess, tf.train.latest_checkpoint('./'))


  test_data = mnist.test.images
  test_labels = mnist.test.labels
  gates = np.zeros((3,10,11))


  digit = 4
  elem = np.where(test_labels[:, digit] == 1)
  elem = elem[0][0]
  test_image = test_data[elem, :]
  image1 = np.reshape(test_image, (1, 28 * 28))
  gates[:, :, 10] = graph.determineGates(image1, x, sess)



  tr_data, tr_label = mnist.train.next_batch(50000)

  for i in range(10):
    elem = np.where(tr_label[:,i] == 1)
    elem = elem[0][0]
    test_image = tr_data[elem,:]
    image1 = np.reshape(test_image, (1, 28 * 28))
    gates[:,:,i] = graph.determineGates(image1, x, sess)

  np.save('gates.txt', gates)

def main(_):
  train()


if __name__ == '__main__':
  tf.app.run(main=main)
