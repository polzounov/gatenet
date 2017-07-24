from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import namedtuple
import tensorflow as tf
import numpy as np

from l2l.utils import *
import l2l.networks


MetaOptimizerRNN = namedtuple('MetaOptimizerRNN', 'rnn, flat_helper')


class MetaOptimizer():
    def __init__(self, 
                 optimizer_type='CoordinateWiseLSTM',
                 shared_scopes=['init_graph'],
                 second_derivatives=False,
                 params_as_state=False,
                 rnn_layers=(5,5),
                 len_unroll=5,
                 w_ts=None,
                 lr=0.001, # The lr for the meta optimizer (not for fx)
                 name='MetaOptimizer'):
        '''An optimizer that mimics the API of tf.train.Optimizer
        ARGS:
            - optimizer_type: The type of RNN you want to use for the metaoptimizer
                - options: CoordinatewiseLSTM & LSTM
            - shared_scopes: Scopes across which to create a metaoptimizer
                eg. If you use just 1 with the outermost scope you get the
                original learning to learn result. If you use a coordinatwise 
                LSTM and scope across layers you get param sharing within each
                layer but seperate between layers
                - TODO: MAKE THIS DESCRIPTION MORE CLEAR #######################
            - name: The name of the MetaOptimizer
                - # TODO: Scope the entire MetaOptimizer under name
        '''
        self._OptimizerType = self._get_optimizer(optimizer_type)
        self._scope = name # The meta optimizer's scope
        self._second_derivatives = second_derivatives
        #self._params_as_state = params_as_state # TODO: Implement
        self._rnn_layers = rnn_layers
        self._len_unroll = len_unroll
        self._w_ts = w_ts
        self._lr = lr

        self._opt_helper = tf.train.GradientDescentOptimizer(0.001)
        #self._optimizee_vars = [self._get_vars_in_scope(scope) for scope in shared_scopes]

        with tf.variable_scope(self._scope):
            self._optimizers = []
            for scope in shared_scopes:
                ### TODO check to make sure this scope doesn't contain vars form
                ### the meta optimizer itself
                self._verify_scope(scope)
                # Get all trainable variables in the given scope and create an 
                # independent optimizer to optimize all variables in that scope
                optimizer = self._init_optimizer(optimizer_options, scope)
                self._optimizers.append(optimizer)
                print('\n Optimizer:')
                print(optimizer)

        # For scoping the variables to train with a step of ADAM
        # (make sure that you only update optimizer vars and not optimizee vars)
        self._meta_optimizer_vars = self._get_vars_in_scope(self._scope)


    ##### SIMPLE HELPER FUNCTIONS ##############################################
    def _get_vars_in_scope(self, scope):
        '''Returns a list of trainable variables in `scope`'''
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)
        # snt.get_variables_in_module(network)
    
    def _get_optimizer(optimizer_name):
        '''Returns the optimizer class, works for strings too'''
        if isinstance(optimizer_name, str): # PYTHON 3 ONLY, TODO FIX FOR PY2
            optimizer_mapping = {
                'CoordinateWiseLSTM': networks.CoordinateWiseDeepLSTM,
                'StandardLSTM': networks.StandardDeepLSTM,
                'KernelLSTM': networks.KernelDeepLSTM
            }
            return optimizer_mapping[optimizer_name]
        return optimizer_name # Assume it's the class itself

    def _verify_scope(self, scope):
        ##  if scope contain meta_optimizer vars:
        ##      raise ValueError('You cannot use a scope containing the meta optimizer')
        print('TODO Implement _verify_scope')
        pass

    def _get_meta_optimizer_vars(self):
        return self._get_vars_in_scope(self._scope)


    ##### OPTIMIZER FUNCTIONS ##################################################
    def _init_optimizer(self, optimizer_options, scope, name='optimizer'):
        '''Creates a named tuple of FlatteningHelper and Network objects'''
        flat_helper = FlatteningHelper(scope)
        with tf.variable_scope(name):
            rnn = self._OptimizerType(output_size=flat_helper.flattened_shape,
                                      batch_size=1, #batch_size = ...,
                                      layers=(5, 5),#layers=self.rnn_layers,
                                      scale=1.0,    #scale= ...,                        
                                      name='LSTM')  #name='Something else')
        return MetaOptimizerRNN(rnn, flat_helper)

    def _meta_loss(self, optimizee_loss_func):
        '''Takes `optimizee_loss_func` applies a gradient step (to the optimizee
        network) for the original variables and returns the loss for the meta
        optimizer itself

        - optimizee_loss_func is the loss you want to minimize for the optimizee 
        function (eg. cross-entropy between predictions and labels on mnist)

        https://www.tensorflow.org/api_docs/python/tf/train/GradientDescentOptimizer
        USE tf.train.GradientDescentOptimizer (for now) to:
            - Get the gradients for certain variables:
                - tf.train.GradientDescentOptimizer().compute_gradients()
            - Apply the deltas from the meta_learner:
                - tf.train.GradientDescentOptimizer().apply_gradients()
        '''
        if self._w_ts is None:
            # Time step weights are all equal as in paper
            self._w_ts = [1. for _ in range(self._len_unroll)] 

        with tf.name_scope('meta_loss'):
            meta_loss = 0

            for t in range(self._len_unroll):
                deltas = self._update_calc(fx, x)
                self._update_step(deltas)

                # Add the loss of the optmizee after the update step to the meta
                # loss weighted by w_t for the current time step
                meta_loss += tf.reduce_sum(optimizee_loss_func) * self._w_ts[t]

        return meta_loss

    def _update_calc(self, fx, x):
        '''Single step of the meta optimizer to calculate the delta updates for
        the params of the optimizee network

        fx is the optimizee loss func
        x are all the variables in the network
        '''
        with tf.name_scope("gradients"):
            gradients = tf.gradients(fx, x)

            # However it looks like things like BatchNorm, etc. don't support 
            # second-derivatives so we still need this term.
            if not self._second_derivatives:
                gradients = [tf.stop_gradient(g) for g in gradients]

        with tf.name_scope('meta_optmizer_step'):
            for optimizer in self._optimizers
                list_deltas = []
                with tf.name_scope('deltas'):
                    RNN = optimizer.rnn # same as optimizer[0]
                    flat_helper = optimizer.flat_helper # same as optimizer[1]
                    flattened_grads = flat_helper.flatten(gradients)
                    flattened_deltas = RNN(flattened_grads)
                    deltas = flat_helper.unflatten(flattened_deltas)
                    list_deltas.append(deltas)
            merged_deltas = merge_var_lists(list_deltas)

        return deltas

    def _update_step(self, deltas):
        '''Performs the actual update of the optimizee's params'''
        # TODO: Do this properly
        # This is the lazy version of code commented out below 
        '''
        with tf.name_scope("dx"):
            for subset, key, s_i in zip(subsets, net_keys, state):
                x_i = [x[j] for j in subset]
                deltas, s_i_next = update(nets[key], fx, x_i, s_i)
                for idx, j in enumerate(subset):
                    x_next[j] += deltas[idx]
                state_next.append(s_i_next)
        '''
        if self._tf_optim is None: # Init first time it's called
            self._tf_optim = tf.train.Optimizer()

        with scope('update_optimizee_params'):
            self._tf_optim.apply_gradients(deltas)


    ##### PUBLIC API ###########################################################
    def minimize(self, loss_func):
        '''A single step of optimization for the meta optimizer
        '''
        #============== Call a series of rnn update steps here =================
        ##meta_loss = self._meta_loss(loss_func)
        meta_loss = self._meta_loss

        # Get all trainable variables from the meta_optimizer itself
        meta_vars = self._meta_optimizer_vars

        # Update step of adam to (only) the meta optimizer's variables
        optimizer = tf.train.AdamOptimizer(self._lr)
        train_step = optimizer.minimize(meta_vars)

        return train_step

