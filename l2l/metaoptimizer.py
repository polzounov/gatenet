from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import namedtuple
import tensorflow as tf
import numpy as np
import mock

from l2l.utils import *
from l2l import networks


#MetaOptimizerRNN = namedtuple('MetaOptimizerRNN', 'rnn, rnn_hidden_state, flat_helper')


def _custom_getter(name=None,
                   var_dict=None,
                   shape=None,
                   dtype=None,
                   initializer=None,
                   regularizer=None,
                   reuse=None,
                   trainable=None,
                   collections=None,
                   caching_device=None,
                   partitioner=None,
                   validate_shape=None,
                   use_resource=None):
    kws = [shape, dtype, initializer, regularizer, reuse, trainable, 
           collections, caching_device, partitioner, validate_shape, 
           use_resource]
    kw_names = ['shape', 'dtype', 'initializer', 'regularizer', 'reuse', 
                'trainable', 'collections', 'caching_device', 'partitioner', 
                'validate_shape', 'use_resource']
    for i, kw in enumerate(kws):
        if kw is not None:
            raise AttributeError('The meta opt\'s custom getter does not'
                                 'support the keyword argument:', kw_names[i])
    if (name is None) or (var_dict is None):
        raise AttributeError('Need name and var dict for meta opt custom getter')
    # Return the var or tensor
    return var_dict[name]


def _wrap_variable_creation(func, var_dict):
    '''Provides a custom getter for all variable creations.'''

    def custom_getter(*args, var_dict=var_dict, **kwargs):
        return _custom_getter(*args, var_dict=var_dict, **kwargs)

    original_get_variable = tf.get_variable
    def custom_get_variable(*args, **kwargs):
        if hasattr(kwargs, 'custom_getter'):
            raise AttributeError('Custom getters are not supported for optimizee variables.')
        return original_get_variable(*args, custom_getter=custom_getter, **kwargs)

    # Mock the get_variable method.
    with mock.patch('tensorflow.get_variable', custom_get_variable):
        return func()


class MetaOptimizer():
    def __init__(self, 
                 shared_scopes=['init_graph'],
                 optimizer_type=networks.StandardDeepLSTM,
                 second_derivatives=True,
                 params_as_state=False,
                 rnn_layers=(5,5),
                 len_unroll=3,
                 w_ts=None,
                 lr=1, # Scale the deltas from the optimizer
                 meta_lr=0.001, # The lr for the meta optimizer (not for fx)
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
        self._meta_lr = meta_lr
        
        # Get all of the variables of the optimizee network ## TODO improve
        self._optimizee_vars = []
        for scope in shared_scopes:
            self._optimizee_vars += self._get_vars_in_scope(scope)
        self._optimizee_vars = tuple(self._optimizee_vars) # Freeze values


        self._fake_optimizee_vars = np.zeros(len(self._optimizee_vars), dtype=object)
        self._fake_optimizee_var_dict = {}
        for i, var in enumerate(self._optimizee_vars):
            print('Var :', var)
            fake_var_tensor = tf.identity(var) # Returns a _TENSOR_ not Variable
            fake_var = tf.Variable(fake_var_tensor)
            self._fake_optimizee_vars[i] = fake_var
            print('Fake:', self._fake_optimizee_vars[i])

            # Dictionary for the custom getter
            self._custom_var_dict = {}
            self._custom_var_dict[var.name] = self._fake_optimizee_vars[i]
        self._fake_optimizee_vars = tuple(self._fake_optimizee_vars) # Freeze values



        with tf.variable_scope(self._scope):
            self._optimizers = []
            for i, scope in enumerate(shared_scopes):
                # Make sure scope doesn't contain vars from the meta optimizer
                self._verify_scope(scope)

                # Get all trainable variables in the given scope and create an 
                # independent optimizer to optimize all variables in that scope
                optimizer = self._init_optimizer({}, scope, name='optimizer'+str(i))
                self._optimizers.append(optimizer)

                print('\n Optimizer:')
                print(optimizer)


    ##### SIMPLE HELPER FUNCTIONS ##############################################
    def _get_vars_in_scope(self, scope):
        '''Returns a list of trainable variables in `scope`'''
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)
    
    def _get_optimizer(self, optimizer_name):
        '''Returns the optimizer class, works for strings too'''
        if isinstance(optimizer_name, str): # PYTHON 3 ONLY, TODO FIX FOR PY2
            optimizer_mapping = {
                'CoordinateWiseLSTM': networks.CoordinateWiseDeepLSTM,
                'StandardLSTM': networks.StandardDeepLSTM,
                'KernelLSTM': networks.KernelDeepLSTM
            }
            return optimizer_mapping[optimizer_name]
        return optimizer_name # If not string assume it's the class itself

    def _verify_scope(self, scope):
        ##if scope contain meta_optimizer vars:
        ##  raise ValueError('You cannot use a scope containing the meta optimizer')
        print('TODO Implement _verify_scope')


    ##### OPTIMIZER FUNCTIONS ##################################################
    def _init_optimizer(self, optimizer_options, scope, name='optimizer'):
        '''Creates a named tuple of FlatteningHelper and Network objects'''
        flat_helper = FlatteningHelper(scope)
        with tf.variable_scope(name):
            rnn = self._OptimizerType(output_size=flat_helper.flattened_shape,
                                      layers=(5, 5),#layers=self.rnn_layers,
                                      scale=1.0,    #scale= ...,
                                      name='LSTM')  #name='Something else')
            intitial_hidden_state = None
        return [rnn, intitial_hidden_state, flat_helper]#MetaOptimizerRNN(rnn, intitial_hidden_state, flat_helper)

    def _meta_loss(self, loss_func):
        '''Takes `optimizee_loss_func` applies a gradient step (to the optimizee
        network) for the original variables and returns the loss for the meta
        optimizer itself

        - optimizee_loss_func is the loss you want to minimize for the optimizee 
        function (eg. cross-entropy between predictions and labels on mnist)
        '''

        if self._w_ts is None:
            # Time step weights are all equal as in paper
            self._w_ts = [1. for _ in range(self._len_unroll)] 

        with tf.name_scope('meta_loss'):
            meta_loss = 0
            prev_loss = _wrap_variable_creation(loss_func, self._fake_optimizee_var_dict)
            prev_loss = prev_loss()

            for t in range(self._len_unroll):
                x = list(self._fake_optimizee_vars)
                deltas = self._update_calc(prev_loss, x)
                self._fake_update_step(deltas)

                prev_loss = _wrap_variable_creation(loss_func, self._fake_optimizee_var_dict)()
                # Add the loss of the optmizee after the update step to the meta
                # loss weighted by w_t for the current time step
                meta_loss += tf.reduce_sum(prev_loss) * self._w_ts[t]

        return meta_loss

    """
    ############################################################################
    ####### This does only one tf.gradients call and attempts to split those 
    ####### grads  into multiple 'pieces'
    ############################################################################
    def _update_calc(self, fx):
        '''Single step of the meta optimizer to calculate the delta updates for
        the params of the optimizee network

        fx is the optimizee loss func
        x are all the variables in the network
        '''
        x = self._optimizee_vars
        gradients = tf.gradients(fx, x)

        # However it looks like things like BatchNorm, etc. don't support
        # second-derivatives so we still need this term.
        if not self._second_derivatives:
            gradients = [tf.stop_gradient(g) for g in gradients]

        with tf.name_scope('meta_optmizer_step'):
            for optimizer in self._optimizers:
                list_deltas = []
                with tf.name_scope('deltas'):
                    RNN = optimizer.rnn # same as optimizer[0]
                    prev_state = optimizer.rnn_hidden_state # same as optimizer[1]
                    flat_helper = optimizer.flat_helper # same as optimizer[2]

                    # Get the gradients that match the current RNN
                    matching_grads = flat_helper.matching_grads(gradients)

                    # Flatten the gradients from list of (gradient, variable) 
                    # into single tensor (k,)
                    flattened_grads = flat_helper.flatten(matching_grads)

                    # If first run set initial intputs ########## This is hacky!!! Fix this ###############
                    if prev_state is None:
                        prev_state = RNN.initial_state_for_inputs(flattened_grads)

                    # Run step for the RNN optimizer
                    flattened_deltas, next_state = RNN(flattened_grads, prev_state)

                    # Set the new hidden state for the optimizer
                    optimizer.rnn_hidden_state = next_state ############### TEST THIS !!!! ################

                    # Get deltas back into original form
                    deltas = flat_helper.unflatten(flattened_deltas)
                    list_deltas.append(deltas)

            # Get `deltas` into form that `gradients` are in
            merged_deltas = merge_var_lists(list_deltas)

        return deltas
    ############################################################################
    """

    ###### This one call tf.gradients several times (since it's a static graph 
    # it doesn't actually recalculate grads multiple times but combines them???)
    def _update_calc(self, fx, x):
        '''Single step of the meta optimizer to calculate the delta updates for
        the params of the optimizee network

        fx is the optimizee loss func
        x are all the variables in the network
        '''
        with tf.name_scope('meta_optmizer_step'):
            list_deltas = []
            for i, optimizer in enumerate(self._optimizers):
                with tf.name_scope('deltas'):
                    RNN = optimizer[0]
                    prev_state = optimizer[1]
                    flat_helper = optimizer[2]

                    # Gradients for ONLY the current RNNs vars
                    gradients = tf.gradients(fx, x)

                    # It looks like things like BatchNorm, etc. don't support
                    # second-derivatives so we still need this term.
                    if not self._second_derivatives:
                        gradients = [tf.stop_gradient(g) for g in gradients]

                    # Flatten the gradients from list of (gradient, variable) 
                    # into single tensor (k,)
                    flattened_grads = flat_helper.flatten(gradients)

                    # If first run set initial intputs ########## This is hacky!!! Fix this ###############
                    if prev_state is None:
                        prev_state = RNN.initial_state_for_inputs(flattened_grads)

                    # Run step for the RNN optimizer
                    flattened_deltas, next_state = RNN(flattened_grads, prev_state)

                    # Set the new hidden state for the optimizer
                    self._optimizers[i][1] = next_state ########## TEST THIS !!!! #############

                    # Get deltas back into original form
                    deltas = flat_helper.unflatten(flattened_deltas)
                    list_deltas += deltas
        return list_deltas

    def _fake_update_step(self, deltas):
        '''Performs the actual update of the optimizee's params'''
        with tf.name_scope('update_optimizee_params'):
            for grad, var in deltas:
                print('\nvar : {}\ngrad: {}'.format(var, grad))
                tf.assign_sub(var, grad) # Update the variable with the delta

    def _real_update_step(self):
        for var in self._optimizee_vars:
            fake_var = self._fake_optimizee_var_dict[var.name]
            var.assign(fake_var)


    def minimize(self, loss_func=None):
        '''A series of updates for the optmizee network and a single step of 
        optimization for the meta optimizer
        '''
        if loss_func is None:
            raise ValueError('loss_func must not be none')
        meta_loss = self._meta_loss(loss_func)

        # Get all trainable variables from the meta_optimizer itself
        # For scoping the variables to train with a step of ADAM
        # (make sure that you only update optimizer vars and not optimizee vars)
        meta_optimizer_vars = self._get_vars_in_scope(scope=self._scope)

        # Update step of adam to (only) the meta optimizer's variables
        optimizer = tf.train.AdamOptimizer(self._meta_lr)
        #train_step = optimizer.minimize(meta_loss, var_list=meta_optimizer_vars)

        # Update the original variables with the updates to the fake ones
        self._real_update_step()

        # This is actually multiple steps of update to the optimizee and one 
        # step of optimization to the optimizer itself
        return 0#train_step

