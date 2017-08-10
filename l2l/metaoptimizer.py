from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import namedtuple
import tensorflow as tf
import numpy as np
import mock

from l2l.utils import *
from l2l import networks
from tensorflow_utils import variable_summaries


#MetaOptimizerRNN = namedtuple('MetaOptimizerRNN', 'rnn, rnn_hidden_state, flat_helper')


def _custom_getter(name, *args, var_dict=None, **kwargs):
    if var_dict is None:
        raise AttributeError('No var dictionary is given')
    # Return the var or tensor
    return var_dict[name+':0']


def _wrap_variable_creation(func, var_dict):
    '''Provides a custom getter for all variable creations.'''
    def custom_get_variable(*args, **kwargs):
        if hasattr(kwargs, 'custom_getter'):
            raise AttributeError('Custom getters are not supported for optimizee variables.')
        return _custom_getter(*args, var_dict=var_dict, **kwargs)

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

        with tf.variable_scope(self._scope+'/init'):
            # Create fake variables that will be called with a custom getter for
            # the network instead of the real variables
            self._fake_optimizee_var_dict, self._fake_optimizee_vars = self._create_fakes(self._optimizee_vars)

            self._optimizers = []
            for i, scope in enumerate(shared_scopes):
                # Make sure scope doesn't contain vars from the meta-opt
                self._verify_scope(scope)

                # Get all trainable variables in the given scope and create 
                # an independent optimizer to optimize all vars in scope
                optimizer = self._init_optimizer({}, scope, name='optimizer'+str(i))
                self._optimizers.append(optimizer)


    ##### WEIRD HACKS & STUFF ##################################################
    def _create_fakes(self, prev_vars):
        '''This creates `fake` tensors which act as the variables in the
        optimizee function but can actually be backpropogated through (removes 
        cycles in the graph). This func can also 
        '''
        # If no dict is given, assume that prev_vars are the real variables
        # and should be of type tf.Variable
        with tf.name_scope('fake_variables'):
            fake_var_dict = {}
            fake_vars = []
            for i, var in enumerate(prev_vars):
                # 'Create' a fake var by passing the real one through an 
                # identity. Note: the fake var is a tensor NOT a variable
                fake_var = tf.identity(var, name='identity_'+str(i))
                fake_vars.append(fake_var)

                fake_var_dict[var.name] = fake_var
            return (fake_var_dict, fake_vars)

    def _update_step(self, deltas, prev_dict):
        ###### This code is dirty and unclear, TODO clean this section up ######

        # Do an update step of the real variables
        
        with tf.name_scope('update_real_vars'):
            for g, v in deltas:
                v -= g
                n = lambda x: x.split(':')[0]
                print(n(v.name))
                variable_summaries(g, name='gradient'+str(n(v.name)))


        '''with tf.name_scope('update_fake_vars'):
            # New dictionary for the custom getter
            fake_var_dict = {}
            fake_vars = []'''


        # Do a mock update step to the fake var tensors. 'Create' a new fake var
        # by passing the prev fake var through an identity (which we should make
        # into a non-differentiable edge so you won't be able to backprop
        # through it) and then subtract the deltas from that new fake var
        with tf.name_scope('update_fake_vars'):
            # New dictionary for the custom getter
            fake_var_dict = {}
            fake_vars = []
            real_var_names = prev_dict.keys()
            for real_var_name in real_var_names:
                matching_prev_var = prev_dict[real_var_name]

                # 'Create' a new fake var
                new_fake_var = tf.identity(matching_prev_var)
                
                # Add the gradient to that fake var
                matching_delta = None
                for i, (g, v) in enumerate(deltas):
                    if v.name == real_var_name:
                        new_fake_var -= g

                # Reassign the value in the dict to be its 'copy' with the grad 
                # subtracted
                fake_var_dict[real_var_name] = new_fake_var
                fake_vars.append(new_fake_var)

            return (fake_var_dict, fake_vars)

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

        # Makes the custom getter callable!!!!
        def callable_custom_getter(*args, **kwargs):
            return _custom_getter(*args, var_dict=self._fake_optimizee_var_dict, **kwargs)

        if self._w_ts is None:
            # Time step weights are all equal as in paper
            self._w_ts = [1. for _ in range(self._len_unroll)] 

        meta_loss = 0
        prev_loss = loss_func(custom_getter=callable_custom_getter)()
        ###prev_loss = _wrap_variable_creation(
        ###    loss_func, self._fake_optimizee_var_dict)()

        for t in range(self._len_unroll):
            # Calculate the updates from the rnn
            deltas = self._update_calc(prev_loss, self._fake_optimizee_vars)

            # Run an update step to the real and fake variables
            fake_var_dict, fake_vars =  self._update_step(deltas, self._fake_optimizee_var_dict)
            # Update the fake var dict and list
            self._fake_optimizee_var_dict = fake_var_dict
            self._fake_optimizee_vars = fake_vars

            ###prev_loss = _wrap_variable_creation(
            ###    loss_func, self._fake_optimizee_var_dict)()
            prev_loss = loss_func(custom_getter=callable_custom_getter)()

            # Add the loss of the optmizee after the update step to the meta
            # loss weighted by w_t for the current time step
            meta_loss += prev_loss * self._w_ts[t]

        with tf.name_scope('final_meta_loss'):
            return tf.reduce_sum(meta_loss)

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
                var =- grad # Update the variable with the delta

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
        with tf.name_scope(self._scope+'/meta_loss'):
            meta_loss = self._meta_loss(loss_func)

        # Get all trainable variables from the meta_optimizer itself
        # For scoping the variables to train with a step of ADAM (make sure
        # that you only update optimizer vars and not optimizee vars)
        meta_optimizer_vars = self._get_vars_in_scope(scope=self._scope)

        # Update step of adam to (only) the meta optimizer's variables
        optimizer = tf.train.AdamOptimizer(self._meta_lr)
        train_step = optimizer.minimize(meta_loss, var_list=meta_optimizer_vars)

        # Update the original variables with the updates to the fake ones
        with tf.name_scope(self._scope+'/update_real_vars'):
            pass#self._real_update_step()

        # This is actually multiple steps of update to the optimizee and one 
        # step of optimization to the optimizer itself
        return train_step

