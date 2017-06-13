from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


import data_manager
import pathnet
import pprint as pp

def train():

  # Import data
  tr_data, tr_label = data_manager.get_next_batch()

  # Start session
  sess = tf.InteractiveSession()

  # Input placeholders
  X = tf.placeholder(tf.float32, [None, 784], name='x-input')
  y_ = tf.placeholder(tf.float32, [None, 10], name='y-input')

  # Build computation graph
  graph_structure = [ [ ((None,784), pathnet.identity_module) ],
                      [ ((None,20), pathnet.perceptron_module), ((None,20), pathnet.perceptron_module) ],
                      [ ((None,20), pathnet.perceptron_module) ],
                      [ ((None,20), pathnet.perceptron_module) ],
                      [ ((None,20), pathnet.perceptron_module) ],
                      [ ((None,10), pathnet.perceptron_module) ]
                    ]
  weights_dict = pathnet.init_params(graph_structure, classes=10)
  print('\n\nKeys:\n')
  pp.pprint(str(weights_dict.keys()))
  print('\n\n')
  y = pathnet.build_pathnet_graph(X, weights_dict, graph_structure)

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
  correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

  # Initialize Variables
  tf.global_variables_initializer().run()

  writer = tf.summary.FileWriter('./logs', graph=tf.get_default_graph())

  for i in range(100):

    # Shuffle the data
    idx=list(range(len(tr_data)))
    np.random.shuffle(idx)
    tr_data=tr_data[idx]
    tr_label=tr_label[idx]

    # Insert Candidate
    acc = 0
    for k in range(Parameters.T):
      _, acc_epoc = sess.run([train_step,accuracy],
                             feed_dict={X:tr_data[k*Parameters.batch_num:(k+1)*Parameters.batch_num,:],
                                        y_:tr_label[k*Parameters.batch_num:(k+1)*Parameters.batch_num,:]})
      acc +=acc_epoc

    acc=acc/Parameters.T
    print('Training Accuracy at step %s: %s' % (i, acc));

def main(_):
  train()


if __name__ == '__main__':
  tf.app.run(main=main)
