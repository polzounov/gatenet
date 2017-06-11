from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from layers import *

from parameters import Parameters

import data_manager

from tensorflow_utils import *

def train():

  # Import data
  tr_data, tr_label = data_manager.get_next_batch()

  # Start session
  sess = tf.InteractiveSession()

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
                             feed_dict={x:tr_data[k*Parameters.batch_num:(k+1)*Parameters.batch_num,:],
                                        y_:tr_label[k*Parameters.batch_num:(k+1)*Parameters.batch_num,:]})
      acc +=acc_epoc

    acc=acc/Parameters.T
    print('Training Accuracy at step %s: %s' % (i, acc));

def main(_):
  train()


if __name__ == '__main__':
  tf.app.run(main=main)
