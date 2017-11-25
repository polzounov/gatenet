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
from dataset_loading.data_managers import *
from timing import *


################################################################################
######                        MAIN PROGRAM                                ######
################################################################################
def train(parameter_dict):
    with tf.Session() as sess:

        k_shot = 2
        num_classes = parameter_dict['C']
        num_test_images = 5
        mnist_path = '../datasets/mnist'
        metaDataManager = MetaDataManager(mnist_path, dataset='mnist', load_images_in_memory=False)
        metaDataManager.build_dataset(num_classes, k_shot, num_test_images)

        # Input placeholders
        x = tf.placeholder(tf.float32, [None, metaDataManager.dataset.image_size_matrix[0],
                                        metaDataManager.dataset.image_size_matrix[1],
                                        metaDataManager.dataset.image_size_matrix[2]], name='x-input')
        y_ = tf.placeholder(tf.float32, [None, num_classes], name='y-input')


        ########## Define simple graph #########################################
        graph = Graph(parameter_dict)

        # Get layer wise variable sharing for the meta optimizer
        shared_scopes = graph.scopes(scope_type='graph')

        ########## Build the rest of the functions in the graph ################
        def loss_func(x=x, y_=y_, fx=graph.return_logits, mock_func=None, var_dict=None):
            def build():
                y = fx(x)  # , custom_getter=custom_getter)
                with tf.name_scope('loss'):
                    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))

                    # Add regularization to the weights in init_graph
                    '''
                    weights_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='init_graph')
                    reg = tf.contrib.layers.l2_regularizer(1.)
                    loss += tf.contrib.layers.apply_regularization(reg, weights_list=weights_list)
                    '''
                    return loss

        def loss_func(x=x, y_=y_, fx=graph.return_logits, mock_func=None, var_dict=None):
            def build():
                y = fx(x)  # , custom_getter=custom_getter)
                with tf.name_scope('loss'):
                    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))

                    # Add regularization to the weights in init_graph
                    '''
                    weights_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='init_graph')
                    reg = tf.contrib.layers.l2_regularizer(1.)
                    loss += tf.contrib.layers.apply_regularization(reg, weights_list=weights_list)
                    '''
                    return loss

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
            # Both need to be normalized (use label norms for both preds & labe
            # ls)
            correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            tf.summary.scalar('accuracy', accuracy)


        # Meta optimization
        optimizer = MetaOptimizer(shared_scopes, name='MetaOptSimple')
        train_step, train_step_meta = optimizer.minimize(loss_func=loss_func)
        #optimizer = tf.train.AdamOptimizer(0.001)
        #train_step = optimizer.minimize(loss); train_step_meta = loss

        merged = tf.summary.merge_all()
        writer = tf.summary.FileWriter('./logs/mnist', graph=sess.graph)
        # Command to run: tensorboard --logdir=l2l/logs/mnist

        # Initialize Variables
        tf.global_variables_initializer().run()

        ################ Run the graph #########################################
        for i in range(parameter_dict['num_batches']):
            images, labels = metaDataManager.get_train_batch()

            tr_data = np.reshape(images, (-1, 28,28,1))

            tr_label = np.zeros((len(labels), num_classes))
            tr_label[np.arange(len(labels)), labels] = 1


            # Save summaries and print
            if i % parameter_dict['print_every'] == 0:

                
                # Print out variables to paste into script to test easily
                print('\nvariables = {')
                for var in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='init_graph'):
                    if 'w' in var.name:
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
                print('Predictions: {}\n'.format(predicted))


                #print('Predictions & Answers')
                #for i in range(min(len(tr_label), 10)):
                    #print('Accuracy: {}'.format(accuracy))

            else:
                acc, _, _, predicted, loss_ = sess.run(
                            [accuracy, train_step, train_step_meta, y, loss],
                            feed_dict={x: tr_data, y_: tr_label})



if __name__ == "__main__":

    parameter_dict = {
        'C': 5,
        'sublayer_type': AdditionSublayerModule,
        'hidden_size': 2,
        'gamma': 0,
        'batch_size': 10,
        'num_batches': 5001,
        'learning_rate': 0.001,
        'print_every': 1,
        'M': 1,
        'L': 1,
        'module_type': ConvModule,
    }

    train(parameter_dict)
