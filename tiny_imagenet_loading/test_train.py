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
from tiny_imagenet_loading.dManager import *
from tiny_imagenet_loading.timing import *

def train(parameter_dict=None):
    timer = Timer()
    timer.reset_time()

    if parameter_dict is None:
        print('Use test params')
        parameter_dict = Parameters().__dict__

    # Probably switch to Session rather than InteractiveSession later on
    # Start session
    sess = tf.InteractiveSession()

    omniglot_path = "/home/chris/images_background"
    mnist_path = "/home/chris/mnist_png"

    k_shot = 10
    num_classes = 10
    num_test_images = 5
    metaDataManager = MetaDataManager(omniglot_path, dataset='omniglot', load_images_in_memory=False)
    metaDataManager.build_dataset(num_classes, k_shot, num_test_images)

    # Input placeholders
    x = tf.placeholder(tf.float32, [None, metaDataManager.dataset.image_size_matrix[0] ,
                                    metaDataManager.dataset.image_size_matrix[1],
                                    metaDataManager.dataset.image_size_matrix[2]], name='x-input')
    y_ = tf.placeholder(tf.float32, [None, num_classes], name='y-input')

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
    train_step = tf.train.AdamOptimizer(parameter_dict['learning_rate']).minimize(cross_entropy)

    # Accuracy
    with tf.name_scope('accuracy'):
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    i = 0
    for v in tf.trainable_variables():
        print(i+1)
        i = i+1
        variable_summaries(v)

    merged = tf.summary.merge_all()
    #train_writer = tf.summary.FileWriter('./logs')

    writer = tf.summary.FileWriter('./logs', graph=tf.get_default_graph())
    # Initialize Variables
    tf.global_variables_initializer().run()


    # Command to run: tensorboard --logdir=logs


    timer.log_time('setup')
    for i in range(5):

        timer.reset_time()
        images, labels = metaDataManager.get_train_batch()
        timer.log_time('get train batch')


        tr_data = np.reshape(images, (-1, 105,105,1))

        #plt.imshow(np.reshape(images[0], (28,28)))
        #print(labels[0])
        #plt.show()

        tr_label = np.zeros((len(labels), num_classes))
        tr_label[np.arange(len(labels)), labels] = 1


        if i % 10 == 0:
            acc = sess.run(accuracy, feed_dict={x: tr_data, y_: tr_label})
            print('training %d, accuracy %g' % (i, acc))

        timer.reset_time()
        summary, _ = sess.run([merged, train_step], feed_dict={x: tr_data, y_: tr_label})
        timer.log_time('training step')

        writer.add_summary(summary, i)

def main(_):
    train()


if __name__ == '__main__':
    tf.app.run(main=main)
