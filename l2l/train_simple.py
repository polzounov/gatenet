from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from graph.graph import Graph
from graph.module import *
from graph.sublayer import *
from l2l.metaoptimizer import *
from tensorflow_utils import variable_summaries


# A simple (deterministic) 1D problem
def simple_problem(batch_size):
    x = np.random.rand(batch_size, 1) * 10
    #y = np.multiply(x, x)
    #y = np.sqrt(x)
    y = x * 10
    return (x, y)


################################################################################
######                        MAIN PROGRAM                                ######
################################################################################
def train(parameter_dict):
    with tf.Session() as sess:

        # Input placeholders
        x = tf.placeholder(tf.float32, [None, 1], name='x-input')
        y_ = tf.placeholder(tf.float32, [None, 1], name='y-input')

        ########## Define simple graph #########################################
        graph = Graph(parameter_dict)

        ########## Build the rest of the functions in the graph ################
        def loss_func(x=x, y_=y_, fx=graph.return_logits, mock_func=None, var_dict=None):
            def build():
                y = fx(x)#, custom_getter=custom_getter)
                with tf.name_scope('loss'):
                    return tf.reduce_mean(tf.abs(y_ - y))
            # Run through the mock func (w fake vars)
            if (mock_func is not None) and (var_dict is not None):
                return mock_func(build, var_dict)
            # Return using real vars
            return build

        # Get the y, loss, and accuracy to use in printing out stuff later
        y = graph.return_logits(x)

        with tf.name_scope('loss'):
            loss = loss_func()()
            tf.summary.scalar('loss', loss)
        with tf.name_scope('accuracy'):
            # Using cosine similarity for the sqaure root estimator
            # Both need to be normalized (use label norms for both preds & labels)
            norm_y_ = tf.sqrt(tf.reduce_sum(tf.multiply(y_,y_)))
            normalized_y_ = y_ / norm_y_
            normalized_y = y / norm_y_

            accuracy = tf.losses.cosine_distance(normalized_y_, normalized_y, 0)
            tf.summary.scalar('accuracy', accuracy)


        # Get layer wise variable sharing for the meta optimizer
        shared_scopes = ['init_graph']

        # Meta optimization
        optimizer = MetaOptimizer(shared_scopes, name='MetaOptSimple')
        train_step, train_step_meta = optimizer.minimize(loss_func=loss_func)
        ###optimizer = tf.train.AdamOptimizer(0.001)
        ###train_step = optimizer.minimize(loss); train_step_meta = loss

        merged = tf.summary.merge_all()
        writer = tf.summary.FileWriter('./logs/simple', graph=sess.graph)
        # Command to run: tensorboard --logdir=l2l/logs/simple

        # Initialize Variables
        tf.global_variables_initializer().run()

        ################ Run the graph #########################################
        for i in range(parameter_dict['num_batches']):
            tr_data, tr_label = simple_problem(parameter_dict['batch_size'])

            # Save summaries and print
            if i % parameter_dict['print_every'] == 0:
                # Print out variables to paste into script to test easily
                print('\nvariables = {')
                for var in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='init_graph'):
                    print("\t'{}': {},".format(
                        var.name.split(':')[0].split('/')[-1], var.eval().tolist()))
                print('}')

                # Run all the stuff
                summary, acc, _, _, predicted, loss_ = sess.run(
                    [merged, accuracy, train_step, train_step_meta, y, loss],
                    feed_dict={x: tr_data, y_: tr_label})

                # Write out summary at current time_step
                #writer.add_summary(summary, i)

                # Print stuff out
                print('\nIteration: {}, accuracy: {}, loss: {}'.format(i, acc, loss_))
                print('Predictions & Answers')
                for i in range(min(len(tr_label), 10)):
                    print('Pred: {}, Actual: {} -- Input: {}'.format(predicted[i], tr_label[i], tr_data[i]))

            else:
                acc, _, _, predicted, loss_ = sess.run(
                            [accuracy, train_step, train_step_meta, y, loss], 
                            feed_dict={x: tr_data, y_: tr_label})

        # Save the parameters of the metaoptimizer
        optimizer.save(sess)





        """
        ########################################################################
        ################# Run the pretrained optimizer #########################
        ########################################################################
        # Meta optimization
        optimizer2 = MetaOptimizer(shared_scopes, name='MetaOptSimple', load_from_file=['save/meta_opt_network_0.l2l'])
        train_step, train_step_meta = optimizer2.minimize(loss_func=loss_func)

        # Initialize Variables
        tf.global_variables_initializer().run()

        # Run the graph
        for i in range(parameter_dict['num_batches']):
            tr_data, tr_label = simple_problem(parameter_dict['batch_size'])

            # Save summaries and print
            if i % parameter_dict['print_every'] == 0:
                # Print out variables to paste into script to test easily
                print('\nvariables = {')
                for var in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='init_graph'):
                    print("\t'{}': {},".format(
                        var.name.split(':')[0].split('/')[-1], var.eval().tolist()))
                print('}')

                # Run all the stuff
                summary, acc, _, _, predicted, loss_ = sess.run(
                    [merged, accuracy, train_step, train_step_meta, y, loss],
                    feed_dict={x: tr_data, y_: tr_label})

                # Write out summary at current time_step
                #writer.add_summary(summary, i)

                # Print stuff out
                print('\nIteration: {}, accuracy: {}, loss: {}'.format(i, acc, loss_))
                print('Predictions & Answers')
                for i in range(min(len(tr_label), 10)):
                    print('Pred: {}, Actual: {} -- Input: {}'.format(predicted[i], tr_label[i], tr_data[i]))

            else:
                acc, _, _, predicted, loss_ = sess.run(
                    [accuracy, train_step, train_step_meta, y, loss],
                    feed_dict={x: tr_data, y_: tr_label})
        ########################################################################
        ########################################################################
        ########################################################################
        """

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
