from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from graph.graph import Graph
from graph.module import *
from graph.sublayer import *
from l2l.metaoptimizer import *


# A simple (deterministic) 1D problem
def simple_problem(batch_size):
    x = np.random.rand(batch_size, 1) * 5
    y = np.sqrt(x)
    return (x, y)

################################################################################
######                        MAIN PROGRAM                                ######
################################################################################
def train(parameter_dict):
    sess = tf.InteractiveSession()

    # Input placeholders
    x = tf.placeholder(tf.float32, [None, 1], name='x-input')
    y_ = tf.placeholder(tf.float32, [None, 1], name='y-input')

    # Build computation graph
    graph = Graph(parameter_dict)
    y = graph.return_logits(x)

    def loss_func(x=x, y_=y_, fx=graph.return_logits):
        def build():
            y = graph.return_logits(x)
            with tf.name_scope('loss'):
                return tf.reduce_mean(tf.abs(y_ - y))
        return build

    loss = loss_func()

    with tf.name_scope('accuracy'):
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # Get layer wise variable sharing for the meta optimizer
    shared_scopes = graph.scopes('layers')

    # Meta optimization
    print('Actually going to be doing stuff now')
    optimizer = MetaOptimizer(shared_scopes, name='MetaOptSimple')
    train_step = optimizer.minimize(loss_func=loss_func)

    # Initialize Variables
    tf.global_variables_initializer().run()

    writer = tf.summary.FileWriter('./logs/simple', graph=tf.get_default_graph())
    #Command to run: tensorboard --logdir=l2l/logs/simple

    for i in range(parameter_dict['num_batches']):
        tr_data, tr_label = simple_problem(parameter_dict['batch_size'])

        if i % parameter_dict['print_every'] == 0:
            acc = sess.run(accuracy, feed_dict={x: tr_data, y_: tr_label})
            print('\ntraining %d, accuracy %g' % (i, acc))

            predicted, loss_ = sess.run([y, loss], feed_dict={x: tr_data, y_: tr_label})
            actual = tr_label
            print('Predictions & Answers')
            for i in range(min(len(actual), 10)):
                print('Pred: {}, Actual: {}'.format(predicted[i], actual[i]))
            print('Loss: {}'.format(loss_))

        acc = sess.run(accuracy, feed_dict={x: tr_data, y_: tr_label})
        sess.run(train_step, feed_dict={x: tr_data, y_: tr_label})



if __name__ == "__main__":

    parameter_dict = {
        'C': 1,
        'sublayer_type': AdditionSublayerModule,
        'hidden_size': 10,
        'gamma': 0,
        'batch_size': 20,
        'num_batches': 100,
        'learning_rate': 0.001,
        'print_every': 10,
        'M': 1,
        'L': 2,
        'module_type': PerceptronModule,
    }

    train(parameter_dict)
