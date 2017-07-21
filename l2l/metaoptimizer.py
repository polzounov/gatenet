from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from l2l.networks import *

def choose_optimizer(optimizer_type, loss_func, lr=0.001, **kwargs):
    if optimizer_type is 'Adam':
        return tf.train.AdamOptimizer(lr).minimize(loss_func)
    else:
        raise NotImplementedError


class MetaOptimizer(object):
    def __init__(self, params, use_locking, name):
        pass

    def apply_gradients(grads_and_vars, global_step=None, name=None):
        pass

    def compute_gradients(loss, var_list=None,
                          gate_gradients=GATE_OP,
                          aggregation_method=None,
                          colocate_gradients_with_ops=False,
                          grad_loss=None):
        pass


    def minimize(self,
                 loss,
                 global_step=None,
                 var_list=None,
                 gate_gradients=GATE_OP,
                 aggregation_method=None,
                 colocate_gradients_with_ops=False,
                 name=None,
                 grad_loss=None):
        pass
        
