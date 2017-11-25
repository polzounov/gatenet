from random import shuffle
import numpy as np
import tensorflow as tf

from graph.graph import Graph
from graph.module import *
from graph.sublayer import *
from l2l.training import training_setup, training


parameter_dict = {
                'C': 1,
                'sublayer_type': 'AdditionSublayerModule',
                'hidden_size': 4,  # Hidden size of each module (or # of filters if using conv)
                'gamma': 3.,  # Gamma is the strength of gating in Gatenet
                'M': 6,  # The number of 'modules' in each layer of Gatenet
                'L': 2,  # The number of layers
                'module_type': 'PerceptronModule',  # The type of module used
}
MO_options = {
                'optimizer_type': 'CoordinateWiseLSTM',  # Type of RNN used for the meta optimizer
                'second_derivatives': False,
                'rnn_layers': (3,3),  # Hidden sizes of the layers in the RNN
                'len_unroll': 3,
                'w_ts': [0.33 for _ in range(3)],
                'lr': 0.0006,  # Multiply the output of the RNN
                'meta_lr': 0.002,  # LR for the meta optimizer
                'additional_loss_scale': 1.,  # How to weight the 'additional' loss
}
training_info = {
                'batch_size': 16,
                'num_batches': 101,
                'print_every': 10,
}
training_iters = 100


# A simple (deterministic) 1D problem
def simple_problem(batch_size, problem='mult'):
    x = np.random.rand(batch_size, 1) * 10
    if problem == 'mult' or problem == 'multiply':
        y = x * 4
        x = np.concatenate((x, np.ones_like(x)), axis=1)
    elif problem == 'square':
        y = np.multiply(x, x)
        x = np.concatenate((x, -1*np.ones_like(x)), axis=1)
    elif problem == 'sqrt':
        y = np.sqrt(x) * 10
        x = np.concatenate((x, np.zeros_like(x)), axis=1)
    else:
        raise ValueError('Invalid problem type: {}'.format(problem))
    return (x, y)


################################################################################
def train(parameter_dict, MO_options, training_info, training_iters=10):
    # Callable data getter where you can pick the problem
    def data_getter(problem):
        def func(batch_size):
            return simple_problem(batch_size, problem=problem)
        return func

    def random_data_getter(problems):
        def func(batch_size):
            prob = problems[np.random.randint(0, len(problems))]
            return simple_problem(batch_size, problem=prob)
        return func

    with tf.Session() as sess:
        filenames = None#['./save/meta_opt_a_2.l2l', './save/meta_opt_a_4.l2l', './save/meta_opt_a_0.l2l', './save/meta_opt_a_11.l2l', './save/meta_opt_a_7.l2l', './save/meta_opt_a_9.l2l', './save/meta_opt_a_10.l2l', './save/meta_opt_a_13.l2l', './save/meta_opt_a_1.l2l', './save/meta_opt_a_5.l2l', './save/meta_opt_a_8.l2l', './save/meta_opt_a_12.l2l', './save/meta_opt_a_14.l2l', './save/meta_opt_a_6.l2l', './save/meta_opt_a_3.l2l']

        packed_vars = training_setup(sess,
                                     parameter_dict=parameter_dict,
                                     MO_options=MO_options,
                                     training_info=training_info,
                                     additional_train=True,
                                     summaries=None,#'graph',
                                     accuracy_func=None,
                                     optimizer_sharing='m',
                                     load_prev_meta_opt=filenames,
                                     save_optimizer=False)

        #packed_vars['train_step_meta'] = packed_vars['train_step']

        # Run a round of training for multiplication problem
        print('First Iteration: {}. Problem: {}.'.format(0, 'mult'))
        training(data_getter=data_getter('mult'),
                 data_getter_additional=random_data_getter(['mult']),
                 **packed_vars)

        # List of problems
        problems = ['mult', 'square', 'sqrt']

        # Run a round of training with a random problem
        for i in range(int(training_iters)):
            print('TRAINING ITERATION NUMBER: {}'.format(i))

            # Select problem to use
            current_p = np.random.randint(0, len(problems))
            prob = problems[current_p]
            a_probs = [p for j, p in enumerate(problems) if j != current_p]

            # Run training
            print('\n\n\nIteration: {}. Next problem: {}.'.format(i, prob))
            training(data_getter=data_getter(prob),
                     data_getter_additional=random_data_getter(a_probs),
                     **packed_vars)


        # To save the optimizer
        packed_vars['save_optimizer'] = True
        prev_num_batches = packed_vars['num_batches']
        packed_vars['num_batches'] = 0
        filenames = training(**packed_vars)
        packed_vars['save_optimizer'] = None
        packed_vars['num_batches'] = prev_num_batches
        print(filenames)

        '''
        ########################################################################
        print('\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n \
               \n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n \
               \n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n')
        print(filenames)
        print('Load from prev opt - no meta-train')
        # Don't do MO updates
        packed_vars['train_step_meta'] = packed_vars['train_step']
        packed_vars['num_batches'] = int(packed_vars['num_batches']/2)

        for i in range(int(training_iters*1)):
            print('TRAINING ITERATION NUMBER: {}'.format(i))

            # Select problem to use
            current_p = np.random.randint(0, len(problems))
            prob = problems[current_p]
            a_probs = [p for j, p in enumerate(problems) if j != current_p]

            # Run training
            print('\n\n\nIteration: {}. Next problem: {}.'.format(i, prob))
            training(data_getter=data_getter(prob),
                     data_getter_additional=random_data_getter(a_probs),
                     **packed_vars)
        '''

        
train(parameter_dict, MO_options, training_info, training_iters)



















"""
################################################################################
################################################################################

def training_setup_no_mo(sess,
             parameter_dict=None,
             training_info=None,
             summaries=None,
             lr=0.001,
             optimizer_type=tf.train.AdamOptimizer,
             loss_type=lambda y_, y: tf.reduce_mean(tf.abs(y_ - y)),
             accuracy_func=lambda y_, y: tf.reduce_mean(
             tf.cast(tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1)), tf.float32)),
             ):
  '''All the setup for training that is invariant to problem type.
  Return all the variables needed for training.
  '''
  # Input placeholders
  x = tf.placeholder(tf.float32, [None, 2], name='x-input')
  y_ = tf.placeholder(tf.float32, [None, 1], name='y-input')

  # Initialize the graph
  graph = Graph(parameter_dict)
    
  # Get the prediction, loss, and accuracy
  y = graph.return_logits(x)
    
  with tf.name_scope('loss'):
    loss = loss_type(y_, y)
    if summaries == 'all':
      tf.summary.scalar('loss', loss)

  if accuracy_func is not None:
    with tf.name_scope('accuracy'):
        accuracy = accuracy_func(y_, y)
  else:
    accuracy = None

  # Optimization setup
  optimizer = optimizer_type(learning_rate=lr)
  train_step = optimizer.minimize(loss)

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
      'loss': loss,
  }
  return packed_vars


def training_no_mo(sess,
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
           loss):
    for i in range(num_batches):
        tr_data, tr_label = data_getter(batch_size)

        # Save summaries and print
        if i % print_every == 0:

            ################## Run graph based on what is given ################
            if (accuracy is not None) and (merged is not None):
                summary, accuracy_, _, predicted, loss_, = sess.run(
                    [merged, accuracy, train_step, y, loss],
                    feed_dict={x: tr_data, y_: tr_label})
                # Write out summary at current time_step
                writer.add_summary(summary, i)

            elif (accuracy is not None) and (merged is None):
                accuracy_, _, predicted, loss_ = sess.run(
                    [accuracy, train_step, y, loss],
                    feed_dict={x: tr_data, y_: tr_label})

            elif (accuracy is None) and (merged is not None):
                summary, _, predicted, loss_ = sess.run(
                    [merged, train_step, y, loss],
                    feed_dict={x: tr_data, y_: tr_label})
                # Write out summary at current time_step
                writer.add_summary(summary, i)

            else:
                _, predicted, loss_ = sess.run(
                    [train_step, y, loss],
                    feed_dict={x: tr_data, y_: tr_label})
            ####################################################################

            # Print out function depending on whether accuracy is given
            if accuracy is not None:
                print('\nIteration:', i,
                      ', loss:', loss_,
                      ', acc:', accuracy_)
            else:
                print('\nIteration:', i,
                      ', loss:', loss_)
            print('Predictions & Answers')
            for i in range(min(len(tr_label), 10)):
                print('Pred: {}, Actual: {} -- Input: {}'.format(predicted[i], tr_label[i], tr_data[i]))
        else:
            _ = sess.run([train_step], feed_dict={x: tr_data, y_: tr_label})




def train_no_mo(parameter_dict, MO_options, training_info, training_iters=10):
    # Callable data getter where you can pick the problem
    def data_getter(problem):
        def func(batch_size):
            return simple_problem(batch_size, problem=problem)
        return func

    def random_data_getter(problems):
        def func(batch_size):
            prob = problems[np.random.randint(0, len(problems))]
            return simple_problem(batch_size, problem=prob)
        return func

    tf.reset_default_graph()
    with tf.Graph().as_default():

        with tf.Session() as sess:
            packed_vars = training_setup_no_mo(sess,
                                        parameter_dict=parameter_dict,
                                        lr=MO_options['lr'],
                                        training_info=training_info,
                                        summaries=None,#'graph',
                                        accuracy_func=None)

            # Run a round of training for multiplication problem
            print('First Iteration: {}. Problem: {}.'.format(0, 'mult'))
            training_no_mo(data_getter=data_getter('mult'), **packed_vars)

            # List of problems
            problems = ['mult', 'square', 'sqrt']

            # Run a round of training with a random problem
            for i in range(int(training_iters)):
                print('TRAINING ITERATION NUMBER: {}'.format(i))

                # Select problem to use
                current_p = np.random.randint(0, len(problems))
                prob = problems[current_p]

                # Run training
                print('\n\n\nIteration: {}. Next problem: {}.'.format(i, prob))
                training_no_mo(data_getter=data_getter(prob), **packed_vars)
            '''
            for i in range(int(training_iters/2)):
                print('TRAINING ITERATION NUMBER: {}'.format(int(i/2)))
                print('\n\n\nIteration: {}. Next problem: {}.'.format(i*2+1, 'square'))
                training(data_getter=data_getter('square'), **packed_vars)
                
                print('TRAINING ITERATION NUMBER: {}'.format(1+int(i/2)))
                print('\n\n\nIteration: {}. Next problem: {}.'.format(i*2+2, 'mult'))
                training(data_getter=data_getter('mult'), **packed_vars)'''



train_no_mo(parameter_dict, MO_options, training_info, training_iters)

################################################################################
"""