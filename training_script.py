import numpy as np
import tensorflow as tf
from graph.graph import Graph
from l2l.metaoptimizer import *


# All of the default options
parameter_dict = {
        'C': 1,
        'sublayer_type': 'AdditionSublayerModule',
        'hidden_size': 1,
        'gamma': 2,
        'M': 10,
        'L': 2,
        'module_type': 'ConvModule',
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
meta = {
    'init': {
        'dataset': 'mnist',
        'load_images_in_memory': False,
        'seed': 0, # Random seed to keep train/val/test datasets the same
        'splits': {
            'train': 0.6,
            'val'  : 0.0,
            'test' : 0.4
        }
    },
    'build': {
        'num_classes': 2, # Number of classes per meta dataset
        'k_shot': 5, # Number of training examples to give per class
        'num_testing_images': 10
    }
}


################################################################################


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


######################### HELPER FUNCTIONS #####################################


def _run_sess(sess, feed_dict, **kwargs):
    '''Runs the session and returns all given vars in a dictionary with
    the same name as the inputs. Skips all `None` inputs.'''
    names = []
    args = []
    for name, arg in kwargs.items():
        if arg is not None:
            names.append(name)
            args.append(arg)
    outputs = sess.run(*args, feed_dict=feed_dict)
    vals = {}
    for (name, output) in zip(names, outputs):
        vals[name] = output
    return vals


def _print_progress(meta_type=None,
                    x=None,
                    y_=None,
                    **kwargs):
    '''Prints out desired info. TODO: Refactor this ugly code.'''
    if iteration_steps == 0:
        print('\n\n#######################################################')
        print('\nEpoch {}'.format(self.epoch_steps))

    print('Meta type: {}, Iteration: {}, Total steps: {}'.format(
        meta_type, self.iteration_steps, self.total_steps))

    # Print the losses
    loss_str = ''
    for l in ('loss', 'meta_loss', 'a_loss'):
        if kwargs.get(l) is not None:
            loss_str += l + ': ' + str(kwargs.get(l)) + ', '
    print(loss_str[:-2])

    # Print the results
    y = kwargs.get('y')
    if (y is not None) and (y_ is not None) and (x is not None):
        print('Actual: {}, Pred: {}, Input: {}'.format(y_, y, x))
    elif (y is not None) and (y_ is not None):
        print('Actual: {}, Pred: {}'.format(y_, y))
    else:
        pass


################################################################################


class MetaTraining(object):
    
    def __init__(self,
                 sess,

                 # Options for intialization
                 parameter_dict=None,
                 MO_options=None,
                 training_info=None,
                 optimizer_sharing='m',
                 prev_meta_opt=None,

                 # For the meta_learning problem (# of classes, k-shot, etc.)
                 meta=None,

                 # Printing, graphs, etc.
                 save_summaries=None,
                 print_progress=None,

                 # Options for the specific problem type
                 loss_type=lambda y_, y: tf.reduce_mean(tf.abs(y_ - y)),
                 accuracy_func=lambda y_, y: tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1)), tf.float32)),
                 ):

        ##### Misc Vars
        self.epoch_steps = 0
        self.iteration_steps = 0
        self.total_steps = 0
        self.loss_history = []
        self.training_info = training_info

        ##### Save Class Vars ##################################################
        self.meta_build = meta['build']

        ##### Set up the trainer ###############################################
        # Instantiate MetaDataManager class
        self.metaDataManager = MetaDataManager(**meta['init'])

        ##### Set up the graph #################################################
        # Input placeholders
        input_shape, label_shape = self._get_placeholder_shapes()
        self.input = tf.placeholder(tf.float32, input_shape, name='x-input')
        self.label = tf.placeholder(tf.float32, label_shape, name='y-input')
        self.a_input = tf.placeholder(tf.float32, input_shape, name='additional-x-input')
        self.a_label = tf.placeholder(tf.float32, label_shape, name='additional-y-input')

        # Initialize the graph
        graph = Graph(parameter_dict)

        self.loss_func   = _get_callable_loss_func(self.input, self.label, graph.return_logits, loss_type)
        self.a_loss_func = None#_get_callable_loss_func(self.a_input, self.a_label, graph.return_logits, loss_type)

        # Get the prediction, loss, and accuracy
        y = graph.return_logits(x)

        with tf.name_scope('loss'):
            self.loss   = self.loss_func()()
            self.a_loss = self.a_loss_func()()
            if summaries == 'all':
                tf.summary.scalar('loss', loss)

        if accuracy_func is not None:
            with tf.name_scope('accuracy'):
                self.accuracy = accuracy_func(y_, y)
        else:
            self.accuracy = None

        ##### Meta Optimizer Setup #############################################
        # Get the previous meta optimizer's variables
        MO_options['load_from_file'] = load_prev_meta_opt

        # Get module wise variable sharing for the meta optimizer
        MO_options['shared_scopes'] = graph.scopes(scope_type=optimizer_sharing)

        # Meta optimization
        self.optimizer = MetaOptimizer(shared_scopes=shared_scopes, **MO_options)
        self.train_step, self.train_step_meta, self.meta_loss = optimizer.minimize(
                                    loss_func=self.loss_func,
                                    additional_loss_func=self.a_loss_func)

        ########## Other Setup #####################################################
        if (summaries == 'all') or (summaries == 'graph'):
            self.merged = tf.summary.merge_all()
            self.writer = tf.summary.FileWriter('./logs/simple', graph=sess.graph)
            # Command to run: tensorboard --logdir=l2l/logs/simple
        else:
            self.writer = None
            self.merged = None

        # Initialize Variables
        tf.global_variables_initializer().run()


    ### INITIALIZATION HELPERS #################################################
    def _get_placeholder_shapes():
        # Get 1 training example
        meta_build = self.meta_build
        meta_build['k_shot'] = 1
        self.metaDataManager.build_dataset(**meta_build)
        image, label = metaDataManager.get_train_batch('train')

        # Return the shapes
        image_shape = image.shape # Assuming (w,h,c)
        label_shape = label.shape # Assuming (l,)
        image_shape = [None] + image_shape
        label_shape = [None] + label_shape
        return image_shape, label_shape



    ### TRAINING ###############################################################
    def meta_train(self,
                   k_shot=None,
                   **options):
        # Optionally overwrite k_shot (but NOT num_classes)
        meta_build = self.meta_build
        if k_shot is not None:
            meta_build['k_shot'] = k_shot

        # Build new meta dataset
        self.metaDataManager.build_dataset(**meta_build)

        # Get batches
        x, y_ = self.metaDataManager.get_train_batch(self, meta_type='train')
        ##### TODO: FINISH THIS LINE !!!! #####
        ##### a_x, a_y_ = self.metaDataManager.get_train_batch(self, meta_type='train')

        for i in range(self.training_info['num_batches']):
            self.iteration_steps = i
            if i % 10 = 0:
                self._train_step(sess=self.sess,
                                 feed_dict={self.input: x, self.label: y_, self.a_input: a_x, self.a_label: a_y_},
                                 self.train_step,
                                 self.train_step_meta,
                                 self.loss,
                                 self.meta_loss,
                                 self.a_loss,
                                 self.merged,
                                 self.accuracy,
                                 self.y)
            else:
                self._train_step(sess=self.sess,
                                 feed_dict={self.input: x, self.label: y_, self.a_input: a_x, self.a_label: a_y_},
                                 train_step,
                                 train_step_meta)




    def meta_val(self, args):
        pass


    def meta_test(self, args):
        pass

    ### TRAINING HELPERS #######################################################
    def _train_step(self,
                    sess,
                    feed_dict,
                    train_step=None,
                    train_step_meta=None,
                    loss=None,
                    meta_loss=None,
                    a_loss=None,
                    merged=None,
                    accuracy=None,
                    y=None):
        '''Run a single iteration of sess. This can include training steps, 
        losses, etc. Return all variable in vals. Return printable variables in 
        printable_vals
        '''
        vals = run_sess(sess,
                        feed_dict,
                        train_step=train_step,
                        train_step_meta=train_step_meta,
                        loss=loss,
                        meta_loss=meta_loss,
                        a_loss=a_loss,
                        merged=merged,
                        accuracy=accuracy,
                        y=y)

        if vals['merged'] is not None:
            # Write out summary at current time_step
            self.writer.add_summary(summary, i)

        printable_vals = {
            'loss': vals['loss'],
            'meta_loss': vals['meta_loss'],
            'a_loss': vals['a_loss'],
            'y': vals['y'],
            'accuracy': vals['accuracy'],
        }
        self.loss_history.append(vals['loss'])
        self.total_steps += 1
        self.iteration_steps += 1

        return vals, printable_vals
