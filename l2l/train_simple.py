from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from random import shuffle
import numpy as np
import tensorflow as tf
from graph.graph import Graph
from graph.module import *
from graph.sublayer import *
from l2l.metaoptimizer import *
from tensorflow_utils import variable_summaries
from l2l.training import training_setup, training


# A simple (deterministic) 1D problem
def simple_problem(batch_size, problem='mult'):
    x = np.random.rand(batch_size, 1) * 10
    if problem == 'mult' or problem == 'multiply':
        y = x * 5
        x = np.concatenate((x, np.ones_like(x)), axis=1)
    elif problem == 'square':
        y = np.multiply(x, x)
        x = np.concatenate((x, -1*np.ones_like(x)), axis=1)
    elif problem == 'sqrt':
        y = np.sqrt(x)
        x = np.concatenate((x, np.zeros_like(x)), axis=1)
    else:
        raise ValueError('Invalid problem type: {}'.format(problem))
    return (x, y)


################################################################################
######                        MAIN PROGRAM                                ######
################################################################################
def train(parameter_dict, MO_options, training_info):
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
        packed_vars = training_setup(sess,
                                     parameter_dict=parameter_dict,
                                     MO_options=MO_options,
                                     training_info=training_info,
                                     additional_train=True,
                                     summaries=None,#'graph',
                                     accuracy_func=None,
                                     optimizer_sharing='m',
                                     load_prev_meta_opt=None,
                                     save_optimizer=False)

        # Run a round of training for multiplication problem
        training(data_getter=data_getter('mult'),
                 data_getter_additional=random_data_getter(['mult']),
                 **packed_vars)

        '''# List of problems
        problems = ['mult', 'square', 'sqrt']

        # Run a round of training with a random problem
        for i in range(500):
            # Select problem to use
            current_p = np.random.randint(0, len(problems))
            prob = problems[current_p]
            a_probs = [p for j, p in enumerate(problems) if j != current_p]

            # Run training
            print('\n\n\nIteration: {}. Next problem: {}.'.format(i, prob))
            training(data_getter=data_getter(prob),
                     data_getter_additional=random_data_getter(a_probs),
                     **packed_vars)'''
                # List of problems

        for i in range(200):
            print('\n\n\nIteration: {}. Next problem: {}.'.format(i*2, 'square'))
            training(data_getter=data_getter('square'),
                     data_getter_additional=data_getter('mult'),
                     **packed_vars)
            print('\n\n\nIteration: {}. Next problem: {}.'.format(i*2+1, 'mult'))
            training(data_getter=data_getter('mult'),
                     data_getter_additional=data_getter('square'),
                     **packed_vars)


if __name__ == "__main__":

    parameter_dict = {
        'C': 1,
        'sublayer_type': AdditionSublayerModule,
        'hidden_size': 4,
        'gamma': 3.,
        'M': 6,
        'L': 2,
        'module_type': PerceptronModule
    }
    MO_options = {
        'optimizer_type': 'CoordinateWiseLSTM',
        'second_derivatives': False,
        'rnn_layers': (3,3),
        'len_unroll': 3,
        'w_ts': [0.33 for _ in range(3)],
        'lr': 0.001,
        'meta_lr': 0.005,
        'additional_loss_scale': 1.
    }
    training_info = {
        'batch_size': 16,
        'num_batches': 100,
        'print_every': 10
    }

    train(parameter_dict, MO_options, training_info)
