import numpy as np
import tensorflow as tf
from graph.graph import Graph
from graph.module import *
from graph.sublayer import *
from l2l.metaoptimizer import *


'''
# All of the default options
parameter_dict = {
        'C': 1,
        'sublayer_type': AdditionSublayerModule,
        'hidden_size': 1,
        'gamma': 2,
        'M': 10,
        'L': 2,
        'module_type': PerceptronModule,
}
MO_options = {
        'optimizer_type':'CoordinateWiseLSTM',
        'second_derivatives':False,
        'params_as_state':False,
        'rnn_layers':(20,20),
        'len_unroll':3,
        'w_ts':[0.33 for _ in range(3)],
        'lr':0.001,
        'meta_lr':0.01,
        'load_from_file':None,
        'save_summaries':{},
        'name':'MetaOptimizer'
}
training_info = {
        'batch_size':20,
        'num_batches':100,
        'print_every':10
}
'''


def get_callable_loss_func(x, y_,  fx, loss_type):
    '''Returns the callable loss function with the required variables set'''
    def loss_func(x=x,
                  y_=y_, 
                  fx=fx,
                  loss_type=loss_type,
                  mock_func=None,
                  var_dict=None):
        '''Create a loss function callable that can be passed into the 
        meta optimizer. This can then 'mock' variables that are created to get 
        to the loss function (i.e. the variables in fx)
        Args:
            x: The input/data placeholder
            y_: The output/label placeholder
            fx: A function that takes input `x` and predicts a label `y`
            loss_type: The actual loss func to use for prediction & label
        Args for Internal use:
            mock_func: A function that replaces tf.get_variable
            var_dict : The variable dict for mock_func
        '''
        def build():
            y = fx(x)
            with tf.name_scope('loss'):
                return loss_type(y_, y)
        # If the function is being mocked then return `fake variables`
        if (mock_func is not None) and (var_dict is not None):
            return mock_func(build, var_dict)
        # Else return the real variables (normal loss function)
        return build
    return loss_func


def training_setup(sess,
                   parameter_dict=None,#parameter_dict,
                   MO_options=None,#MO_options,
                   training_info=None,#training_info,
                   additional_train=False,
                   summaries=None,
                   loss_type=lambda y_, y: tf.reduce_mean(tf.abs(y_ - y)),
                   accuracy_func=lambda y_, y: tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1)), tf.float32)),
                   optimizer_sharing='m',
                   load_prev_meta_opt=None,
                   save_optimizer=False):
    '''All the setup for training that is invariant to problem type.
    Return all the variables needed for training.
    '''
    ########## Graph setup #####################################################
    # Input placeholders
    x = tf.placeholder(tf.float32, [None, 2], name='x-input')
    y_ = tf.placeholder(tf.float32, [None, 1], name='y-input')
    if additional_train:
        a_x = tf.placeholder(tf.float32, [None, 2], name='additional-x-input')
        a_y_ = tf.placeholder(tf.float32, [None, 1], name='additional-y-input')

    # Initialize the graph
    graph = Graph(parameter_dict)

    loss_func = get_callable_loss_func(x, y_, graph.return_logits, loss_type)
    if additional_train:
        additional_loss_func = get_callable_loss_func(a_x, a_y_,
                                                      graph.return_logits,
                                                      loss_type)
    # Get the prediction, loss, and accuracy
    y = graph.return_logits(x)

    with tf.name_scope('loss'):
        loss = loss_func()()
        if summaries == 'all':
            tf.summary.scalar('loss', loss)
    if additional_train:
            additional_loss = additional_loss_func()()

    if accuracy_func is not None:
        with tf.name_scope('accuracy'):
            accuracy = accuracy_func(y_, y)
    else:
        accuracy = None

    ########## Meta Optimizer Setup ############################################
    # Get the previous meta optimizer's variables
    MO_options['load_from_file'] = load_prev_meta_opt

    # Get module wise variable sharing for the meta optimizer
    shared_scopes = graph.scopes(scope_type=optimizer_sharing)

    # Meta optimization
    optimizer = MetaOptimizer(shared_scopes=shared_scopes, **MO_options)
    if additional_train:
        train_step, train_step_meta, meta_loss = optimizer.minimize(
                                loss_func=loss_func,
                                additional_loss_func=additional_loss_func)
    else:
        train_step, train_step_meta, meta_loss = optimizer.minimize(loss_func)

    ########## Other Setup #####################################################
    if (summaries == 'all') or (summaries == 'graph'):
        merged = tf.summary.merge_all()
        writer = tf.summary.FileWriter('./logs/simple', graph=sess.graph)
        # Command to run: tensorboard --logdir=l2l/logs/simple
    else:
        writer = None
        merged = None

    # Initialize Variables
    tf.global_variables_initializer().run()

    ########## Pack Needed Vars ################################################
    packed_vars = {
        #'optimizer': optimizer,
        'additional_train': additional_train,
        'save_optimizer': save_optimizer,
        'sess': sess,
        'x': x,
        'y_': y_,
        'y': y,
        'batch_size': training_info['batch_size'],
        'num_batches': training_info['num_batches'],
        'print_every': training_info['print_every'],
        'merged': merged,
        'writer': writer,
        'accuracy': accuracy,
        'train_step': train_step,
        'train_step_meta': train_step_meta,
        'loss': loss,
        'meta_loss': meta_loss,
    }
    if additional_train:
        packed_vars['a_x'] = a_x
        packed_vars['a_y_'] = a_y_
        packed_vars['additional_loss'] = additional_loss

    return packed_vars



def training(additional_train=None,
             save_optimizer=None,
             sess=None,
             x=None,
             y_=None,
             y=None,
             a_x=None,
             a_y_=None,
             batch_size=None,
             num_batches=None,
             print_every=None,
             data_getter=None,
             data_getter_additional=None,
             merged=None,
             writer=None,
             accuracy=None,
             train_step=None,
             train_step_meta=None,
             loss=None,
             meta_loss=None,
             additional_loss=None):
    '''Run the actual training from packed vars'''

    if additional_train:
        train_loop_additional(sess,
                              x=x,
                              y_=y_,
                              y=y,
                              a_x=a_x,
                              a_y_=a_y_,
                              batch_size=batch_size,
                              num_batches=num_batches,
                              print_every=print_every,
                              data_getter=data_getter,
                              data_getter_additional=data_getter_additional,
                              merged=merged,
                              writer=writer,
                              accuracy=accuracy,
                              train_step=train_step,
                              train_step_meta=train_step_meta,
                              loss=loss,
                              meta_loss=meta_loss,
                              additional_loss=additional_loss)
    else:
        train_loop(sess,
                   x=x,
                   y_=y_,
                   y=y,
                   batch_size=batch_size,
                   num_batches=num_batches,
                   print_every=print_every,
                   data_getter=data_getter,
                   merged=merged,
                   writer=writer,
                   accuracy=accuracy,
                   train_step=train_step,
                   train_step_meta=train_step_meta,
                   loss=loss,
                   meta_loss=meta_loss)

    if save_optimizer:
        # Save the parameters of the metaoptimizer
        results = optimizer.save(sess, path='save/meta_opt_a')
        filenames = list(results.keys())
        list(print(filenames))
        return filenames


def train_loop(sess,
               x,
               y_,
               y,
               batch_size,
               num_batches,
               print_every,
               data_getter,
               merged,
               writer,
               accuracy,
               train_step,
               train_step_meta,
               loss,
               meta_loss):
    for i in range(num_batches):
        tr_data, tr_label = data_getter(batch_size)

        # Save summaries and print
        if i % print_every == 0:

            ################## Run graph based on what is given ################
            if (accuracy is not None) and (merged is not None):
                summary, accuracy_, _, _, predicted, loss_, meta_loss_ = sess.run(
                    [merged, accuracy, train_step, train_step_meta, y, loss, meta_loss],
                    feed_dict={x: tr_data, y_: tr_label})
                # Write out summary at current time_step
                writer.add_summary(summary, i)

            elif (accuracy is not None) and (merged is None):
                accuracy_, _, _, predicted, loss_, meta_loss_ = sess.run(
                    [accuracy, train_step, train_step_meta, y, loss, meta_loss],
                    feed_dict={x: tr_data, y_: tr_label})

            elif (accuracy is None) and (merged is not None):
                summary, _, _, predicted, loss_, meta_loss_ = sess.run(
                    [merged, train_step, train_step_meta, y, loss, meta_loss],
                    feed_dict={x: tr_data, y_: tr_label})
                # Write out summary at current time_step
                writer.add_summary(summary, i)

            else:
                _, _, predicted, loss_, meta_loss_ = sess.run(
                    [train_step, train_step_meta, y, loss, meta_loss],
                    feed_dict={x: tr_data, y_: tr_label})
            ####################################################################

            # Print out function depending on whether accuracy is given
            if accuracy is not None:
                print('\nIteration:', i,
                      ', loss:', loss_,
                      ', meta_l:', meta_loss_,
                      ', acc:', accuracy_)
            else:
                print('\nIteration:', i,
                      ', loss:', loss_,
                      ', meta_l:', meta_loss_)
            print('Predictions & Answers')
            for i in range(min(len(tr_label), 10)):
                print('Pred: {}, Actual: {} -- Input: {}'.format(predicted[i], tr_label[i], tr_data[i]))
        else:
            _, _ = sess.run([train_step, train_step_meta], feed_dict={x: tr_data, y_: tr_label})


def train_loop_additional(sess,
                          x,
                          y_,
                          y,
                          a_x,
                          a_y_,
                          batch_size,
                          num_batches,
                          print_every,
                          data_getter,
                          data_getter_additional,
                          merged,
                          writer,
                          accuracy,
                          train_step,
                          train_step_meta,
                          loss,
                          meta_loss,
                          additional_loss):
    for i in range(num_batches):
        tr_data, tr_label = data_getter(batch_size)
        a_tr_data, a_tr_label = data_getter_additional(batch_size)

        # Save summaries and print
        if i % print_every == 0:

            ################## Run graph based on what is given ################
            if (accuracy is not None) and (merged is not None):
                summary, accuracy_, _, _, predicted, loss_, meta_loss_, a_loss_ = sess.run(
                    [merged, accuracy, train_step, train_step_meta, y, loss, meta_loss, additional_loss],
                    feed_dict={x: tr_data, y_: tr_label, a_x: a_tr_data, a_y_: a_tr_label})
                # Write out summary at current time_step
                writer.add_summary(summary, i)

            elif (accuracy is not None) and (merged is None):
                accuracy_, _, _, predicted, loss_, meta_loss_, a_loss_ = sess.run(
                    [accuracy, train_step, train_step_meta, y, loss, meta_loss, additional_loss],
                    feed_dict={x: tr_data, y_: tr_label, a_x: a_tr_data, a_y_: a_tr_label})

            elif (accuracy is None) and (merged is not None):
                summary, _, _, predicted, loss_, meta_loss_, a_loss_ = sess.run(
                    [merged, train_step, train_step_meta, y, loss, meta_loss, additional_loss],
                    feed_dict={x: tr_data, y_: tr_label, a_x: a_tr_data, a_y_: a_tr_label})
                # Write out summary at current time_step
                writer.add_summary(summary, i)

            else:
                _, _, predicted, loss_, meta_loss_, a_loss_ = sess.run(
                    [train_step, train_step_meta, y, loss, meta_loss, additional_loss],
                    feed_dict={x: tr_data, y_: tr_label, a_x: a_tr_data, a_y_: a_tr_label})
            ####################################################################

            # Print out function depending on whether accuracy is given
            if accuracy is not None:
                print('\nIteration:', i,
                      ', loss:', loss_,
                      ', meta_l:', meta_loss_,
                      ', addi_l:', a_loss_,
                      ', acc:', accuracy_)
            else:
                print('\nIteration:', i,
                      ', loss:', loss_,
                      ', meta_l:', meta_loss_,
                      ', addi_l:', a_loss_)
            print('Predictions & Answers')
            for i in range(min(len(tr_label), 10)):
                print('Pred: {}, Actual: {} -- Input: {}'.format(predicted[i], tr_label[i], tr_data[i]))
        else:
            _, _ = sess.run([train_step, train_step_meta], 
                    feed_dict={x: tr_data, y_: tr_label,
                               a_x: a_tr_data, a_y_: a_tr_label})