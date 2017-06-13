from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from layers import *
from tensorflow_utils import *
from testing import *

from tensorflow.examples.tutorials.mnist import input_data


def train(parameter_dict=None, skip_digit=8, num_gate_vectors_output=100):
  if parameter_dict is None:
    print('Use test params')
    parameter_dict = Parameters().__dict__

  mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

  # Probably switch to Session rather than InteractiveSession later on
  # Start session
  sess = tf.InteractiveSession()

  # Input placeholders
  x = tf.placeholder(tf.float32, [None, 784], name='x-input')
  y_ = tf.placeholder(tf.float32, [None, 10], name='y-input')

  # Build computation graph
  graph = Graph()
  y = graph.buildTestGraph(x, parameter_dict)

  # Cross Entropy
  diff = tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y)
  cross_entropy = tf.reduce_mean(diff)

  # GradientDescent
  train_step = tf.train.AdamOptimizer(parameter_dict['learning_rate']).minimize(cross_entropy)

  # Accuracy
  correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

  # Initialize Variables
  tf.global_variables_initializer().run()

  skip_digit = 8 # Skip over this digits in training

  for i in range(parameter_dict['num_batches']):
    tr_data, tr_label = mnist.train.next_batch(parameter_dict['batch_size'])
    elems = np.where(np.argmax(tr_label, axis=1) != skip_digit)[0]
    tr_data = tr_data[elems,:]
    tr_label = tr_label[elems,:]

    if len(tr_data) < 50:
      continue

    if i % 100 == 0:
      acc = sess.run(accuracy, feed_dict={x: tr_data, y_: tr_label})
      print('training %d, accuracy %g' % (i, acc))

    sess.run(train_step, feed_dict={x: tr_data, y_: tr_label})

  tr_data, tr_label = mnist.train.next_batch(num_gate_vectors_output)

  # Initialize and add param dict to outputManager
  outputManager = OutputManager()
  outputManager.initialize(parameter_dict)
  # Compute gate vectors
  for i in range(len(tr_label)):
    image = np.reshape(tr_data[i,:], (1, 28 * 28))
    output_image = np.reshape(image,(28,28))
    gates = graph.determineGates(image, x, sess)
    outputManager.addData(gates, np.argmax(tr_label[i]), output_image)
  # Save to file
  outputManager.save()
  

  ''' Need to have run predictions function for this
  # Save outputs of the final output layer (softmax)
  param_dict2 = parameter_dict.copy()
  param_dict2['output_file'] = 'softmax_outputs'
  # Initialize output manager to save gating data
  outputManager = outputManager()
  outputManager.initialize(parameter_dict)
  # Compute gate vectors
  for i in range(len(tr_label)):
    image = np.reshape(tr_data[i,:], (1, 28 * 28))
    pred = 
    gates = graph.determineGates(image, x, sess)
    outputManager.addData(gates, np.argmax(tr_label[i]), output_image)

  # Save gate vectors to file
  outputManager.save()
  '''

def main(_):
  train(skip_digit=8, num_gate_vectors_output=1000)


if __name__ == '__main__':
  tf.app.run(main=main)
