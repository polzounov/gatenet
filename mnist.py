from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from graph.graph import Graph
from tensorflow_utils import *
from testing import *

from tensorflow.examples.tutorials.mnist import input_data
from experiments import *

from sklearn.neighbors import KNeighborsClassifier

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

  # Cross Entropy
  diff = tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y)
  predictions = tf.nn.softmax(y)
  cross_entropy = tf.reduce_mean(diff)

  # GradientDescent
  train_step = tf.train.AdamOptimizer(parameter_dict['learning_rate']).minimize(cross_entropy)

  # Accuracy
  correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

  # Initialize Variables
  tf.global_variables_initializer().run()

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

  tr_data, tr_label = mnist.train.next_batch(num_gate_vectors_output)


  # Initialize and add param dict to outputManager
  outputManager = OutputManager()
  outputManager.initialize(parameter_dict)
  # Compute gate vectors
  for i in range(len(tr_label)):
    image = np.reshape(tr_data[i,:], (1,28, 28,1))
    output_image = np.reshape(image,(28,28))
    gates = graph.determine_gates(image, x, sess)
    outputManager.addData(gates, np.argmax(tr_label[i]), output_image)
  # Save to file
  outputManager.save()
  

  # Save outputs of the final pre-softmax layer (features)
  features_param_dict = parameter_dict.copy()
  features_param_dict['output_file'] = features_param_dict['output_file']+'__feature_layer_outputs'

  features_output_manager = OutputManager()
  features_output_manager.initialize(features_param_dict)
  # Compute the features (y)
  for i in range(len(tr_label)):
    image = np.reshape(tr_data[i,:], (1,28, 28,1))
    output_image = np.reshape(image, (28, 28))
    pred = sess.run(y, feed_dict={x: image})
    features_output_manager.addData(pred, np.argmax(tr_label[i]), output_image)
  # Save feature vectors to file
  features_output_manager.save()


  # Save outputs of the final output layer (softmax)
  predictions_param_dict = parameter_dict.copy()
  predictions_param_dict['output_file'] = predictions_param_dict['output_file']+'__softmax_outputs'

  predictions_output_manager = OutputManager()
  predictions_output_manager.initialize(predictions_param_dict)
  # Compute predictions
  for i in range(len(tr_label)):
    image = np.reshape(tr_data[i,:], (1,28, 28,1))
    output_image = np.reshape(image, (28, 28))
    pred = sess.run(predictions, feed_dict={x: image})
    predictions_output_manager.addData(pred, np.argmax(tr_label[i]), output_image)
  # Save softmax output vectors to file
  predictions_output_manager.save()

  '''
  print('Starting test')


  tr_data, tr_label = mnist.train.next_batch(num_gate_vectors_output)
  # Get the indices of elements in skip list
  skip_digits_test = np.delete([0,1,2,3,4,5,6,7,8,9], skip_digits)
  elems_list = []
  for skip_digit in skip_digits_test:
    elems = (np.where(np.argmax(tr_label, axis=1) == skip_digit)[0])
    elems_list.append(elems)
  elems = np.sort(np.concatenate(elems_list, axis=0))
  elems = np.delete(np.arange(tr_label.shape[0]), elems)  # Invert elems
  tr_data = tr_data[elems, :]
  tr_label = tr_label[elems, :]

  test_images1, test_labels1 = test_images, test_labels
  # Get the indices of elements in skip list
  skip_digits_test = np.delete([0,1,2,3,4,5,6,7,8,9], skip_digits)
  elems_list = []
  for skip_digit in skip_digits_test:
    elems = (np.where(np.argmax(test_labels, axis=1) == skip_digit)[0])
    elems_list.append(elems)
  elems = np.sort(np.concatenate(elems_list, axis=0))
  elems = np.delete(np.arange(test_labels.shape[0]), elems)  # Invert elems
  test_images = test_images[elems, :]
  test_labels = test_labels[elems, :]



  gates_train = np.zeros((len(tr_data), parameter_dict['L'] * parameter_dict['M']))
  for i in range(len(tr_label)):
    image = np.reshape(tr_data[i, :], (1, 28 * 28))
    gates = graph.determine_gates(image, x, sess)
    gates_train[i] = np.reshape(gates, (parameter_dict['L'] * parameter_dict['M']))

  # Determine accuracy of regular image input network and gate input network on test data
  gates_test = np.zeros((len(test_images), parameter_dict['L'] * parameter_dict['M']))
  for i in range(len(test_labels)):
    image = np.reshape(test_images[i, :], (1, 28 * 28))
    gates = graph.determine_gates(image, x, sess)
    gates_test[i] = np.reshape(gates, (parameter_dict['L'] * parameter_dict['M']))



  # mini NN

  image_test_accuracy = sess.run(accuracy, feed_dict={x: test_images, y_: test_labels})
  image_test_accuracy1 = sess.run(accuracy, feed_dict={x: test_images1, y_: test_labels1})





  # Comparing prediction accuracy of image fed neural network
  # With knn using gate vectors
  nbrs = KNeighborsClassifier(n_neighbors=5)
  nbrs.fit(gates_train, np.argmax(tr_label,1))
  g_labels = nbrs.predict(gates_test)
  correct_prediction = np.equal(g_labels, np.argmax(test_labels, 1))
  knn_gate_accuracy = np.mean(correct_prediction)


  # Compute prediction accuracy using gate vector fed neural network
  gate_test_accuracy = train_gate_network(parameter_dict, gates_train, tr_label, gates_test, test_labels, sess)

  print('classification accuracy from neural network with image inputs %g' %(image_test_accuracy))
  print('classification accuracy from neural network 1! with image inputs %g' %(image_test_accuracy1))
  print('classiciation accuracy from knn with gate vectors %g' %(knn_gate_accuracy))
  print('classification accuracy from neural network with gate vector inputs %g' % (gate_test_accuracy))
  '''


def main(_):
  train(skip_digits=[0,1,2], num_gate_vectors_output=1000)


if __name__ == '__main__':
  tf.app.run(main=main)
