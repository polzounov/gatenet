import numpy as np
import tensorflow as tf
from graph.graph import Graph
from l2l.metaoptimizer import *
from dataset_loading.data_managers import SimpleProbTransferLearnDataManager

################################################################################


def _get_callable_loss_func(x, y_, fx, loss_type):
    '''Returns the callable loss function with the required variables set.'''
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
    '''Run the session and return all given vars in a dictionary with the same 
    name as the inputs. Skips all `None` inputs.

    Args:
        sess      : TensorFlow session.
        feed_dict : TensorFlow feed_dict.
        **kwargs  : TensorFlow graph elements to run (train_step, pred, loss,
                    etc). The keyword is the same for 'vals'.

        Return:
            vals: A dictionary returning the evaluated values from **kwargs. The
                  keywords are the same for **kwargs.
    '''
    names = []
    args = []
    for name, arg in kwargs.items():
        if arg is not None:
            names.append(name)
            args.append(arg)
    outputs = sess.run(args, feed_dict=feed_dict)
    vals = {}
    for (name, output) in zip(names, outputs):
        vals[name] = output
    return vals


################################################################################


class Trainer():
    '''Simlifies the training of the transfer learner. All saving & loading of
    networks is encompassed internally.
    '''
    def __init__(self,
                 sess,
                 parameter_dict=None,
                 MO_options=None,
                 training_info=None,
                 optimizer_sharing=None,
                 prev_meta_opt=None,
                 summaries=None,
                 loss_type=None,
                 accuracy_func=None,
                 ):
        '''
        Args: 
            sess             : TensorFlow session.
            parameter_dict   : Params for gatenet.
            MO_options       : Params for the meta optimizer
            training_info    : Training options (eg. batch size,)
            optimizer_sharing: The type of sharing to use for MO
            prev_meta_opt    : (Optionally) use previously initialized MO
            summaries        : Save tf summaries
            loss_type        : A callable taking in prediction & label returning tf (all tf nodes)
            accuracy_func    : A callable taking in prediction & label returning accuracy (all tf nodes)
        '''
        ##### Misc Vars ########################################################
        self.sess = sess
        self.training_info = training_info
        self.batch_size = training_info['batch_size']

        self.current_dataset = 0 # The current dataset being used (from 0 to num_datasets)
        self.current_task = 0  # The current task being trained (from 0 to tasks_per_dataset)
        self.loss_history = [] # Overall history of training, (dataset_iter, current_task, task_steps, loss, meta_loss)
        self.global_steps = 0  # Total number of update steps for Gatenet
        self.task_steps = 0    # Number of steps taken for current task

        # Initialize the data manager (transfer or meta learn data manager)
        self._data_manager_init()

        ##### Set up the graph #################################################
        # Input placeholders
        input_shape, label_shape = self._get_placeholder_shapes()
        self.input   = tf.placeholder(tf.float32, input_shape, name='x-input')
        self.label   = tf.placeholder(tf.float32, label_shape, name='y-input')
        self.a_input = tf.placeholder(tf.float32, input_shape, name='additional-x-input')
        self.a_label = tf.placeholder(tf.float32, label_shape, name='additional-y-input')

        # Initialize the graph
        self.graph = Graph(parameter_dict)

        self.loss_func   = _get_callable_loss_func(self.input,   self.label,   self.graph.return_logits, loss_type)
        self.a_loss_func = _get_callable_loss_func(self.a_input, self.a_label, self.graph.return_logits, loss_type)

        # Get the prediction, loss, and accuracy
        y = self.graph.return_logits(self.input)
        self.pred = y

        with tf.name_scope('loss'):
            self.loss   = self.loss_func()()
            self.a_loss = self.a_loss_func()()
            if summaries == 'all':
                tf.summary.scalar('loss', self.loss)

        if accuracy_func is not None:
            with tf.name_scope('accuracy'):
                self.accuracy = accuracy_func(y_=self.label, y=self.pred)
        else:
            self.accuracy = None

        ##### Meta Optimizer Setup #############################################
        # Get the previous meta optimizer's variables
        MO_options['load_from_file'] = prev_meta_opt

        # Get module wise variable sharing for the meta optimizer
        MO_options['shared_scopes'] = self.graph.scopes(scope_type=optimizer_sharing)

        # Meta optimization
        self.optimizer = MetaOptimizer(**MO_options)
        self.train_step, self.meta_loss_train = self.optimizer.train_step(
                                    loss_func=self.loss_func)
        self.train_step_meta, self.meta_loss_test = self.optimizer.train_step_meta(
                                    loss_func=self.loss_func,
                                    additional_loss_func=self.a_loss_func)

        '''#TASKADDI
        # TODO: Fix hacky solution (make a_loss work with no data?)
        self.train_step_meta_no_a_loss, self.meta_loss_test_no_a_loss = self.optimizer.train_step_meta(
                                    loss_func=self.loss_func,
                                    additional_loss_func=self.a_loss_func,
                                    additional_loss_scale=0.0)'''

        ##### Other Setup ######################################################
        if (summaries == 'all') or (summaries == 'graph'):
            self.merged = tf.summary.merge_all()
            self.writer = tf.summary.FileWriter('./logs', graph=sess.graph)
            # Command to run: tensorboard --logdir=logs
        else:
            self.writer = None
            self.merged = None

        # Initialize Variables
        tf.global_variables_initializer().run()

    def _data_manager_init(self):
        raise NotImplementedError()

    def _get_placeholder_shapes(self):
        raise NotImplementedError()

    def meta_train(self):
        raise NotImplementedError()

    def meta_val(self):
        raise NotImplementedError()

    def meta_test(self):
        raise NotImplementedError()

    def _train_step(self,
                    sess,
                    feed_dict=None,
                    train_step=None,
                    train_step_meta=None,
                    loss=None,
                    meta_loss=None,
                    a_loss=None,
                    merged=None,
                    accuracy=None,
                    y=None):
        '''Run a single iteration of sess. This can include training steps, 
        losses, etc. Return all variables in vals. Return printable variables in 
        printable_vals.
        '''
        vals = _run_sess(sess,
                         feed_dict,
                         train_step=train_step,
                         train_step_meta=train_step_meta,
                         loss=loss,
                         meta_loss=meta_loss,
                         a_loss=a_loss,
                         merged=merged,
                         accuracy=accuracy,
                         y=y)

        if vals.get('merged') is not None:
            # Write out summary at current time_step
            self.writer.add_summary(self.summary, global_step=self.global_steps)

        printable_vals = {
            'accuracy' : vals.get('accuracy'),
            'meta_loss': vals.get('meta_loss'),
            'a_loss'   : vals.get('a_loss'),
            'loss'     : vals.get('loss'),
            'y'        : vals.get('y'),
        }

        return vals, printable_vals

    def _print_progress(self, x=None, y_=None, **kwargs):
        '''Prints out info in a useful format.

        Args (required):
            x        : Input.
            y_       : The given label.

        Kwargs (optional):
            loss     : Loss for the optimizee.
            meta_loss: Current task loss for the optimizer.
            a_loss   : Additional loss (from test set of previous tasks in current
                       dataset - loss to prevent catastrophic forgetting).
            y        : Evaluated prediction.
        '''
        # Print if new task or dataset
        if self.task_steps == 0:
            if self.current_task == 0:
                print('\n\n###################################################',
                       '######################################################')
                print('\nNew Dataset {}'.format(self.current_dataset))
            print('\n\nNew task {}'.format(self.current_task))

        # Print how many steps taken
        print('Task Steps: {}, Total steps: {}'.format(
                self.task_steps, self.global_steps))

        # Print the losses
        loss_str = ''
        for loss_type in ('loss', 'meta_loss', 'a_loss'):
            if kwargs.get(loss_type) is not None:
                loss_str += loss_type + ': ' + str(kwargs.get(loss_type)) + ', '
        print(loss_str[:-2])

        if kwargs.get('accuracy') is not None:
            print('Train accuracy is: {}'.format(kwargs.get('accuracy')))

        # Print the results
        y = kwargs.get('y')
        if (y is not None) and (y_ is not None) and (x is not None):
            max_len = min(len(y_[0]), 3)
            print('Actual: {}, Pred: {}, Input: {}'.format(
                    y_[0][0:max_len],
                    y[0][0:max_len],
                    x[0][0:max_len]))
        elif (y is not None) and (y_ is not None):
            max_len = min(len(y_[0]), 3)
            print('Actual: {}, Pred: {}'.format(
                    y_[0][0:max_len],
                    y[0][0:max_len]))
        else:
            raise ValueError('Input values are invalid.'
                             + '\ndictionary is: {}'.format(kwargs))

    def _print_test(self, x=None, y_=None, **kwargs):
        '''Prints out info in a useful format.

        Args (required):
            x        : Input.
            y_       : The given label.

        Kwargs (optional):
            loss     : Loss for the optimizee.
            meta_loss: Current task loss for the optimizer.
            a_loss   : Additional loss (from test set of previous tasks in current
                       dataset - loss to prevent catastrophic forgetting).
            y        : Evaluated prediction.
        '''
        print('\nTest for task {}'.format(self.current_task))

        # Print how many steps taken
        print('Task Steps: {}, Total steps: {}'.format(
                self.task_steps, self.global_steps))

        # Print the losses
        loss_str = ''
        for loss_type in ('loss', 'meta_loss', 'a_loss'):
            if kwargs.get(loss_type) is not None:
                loss_str += loss_type + ': ' + str(kwargs.get(loss_type)) + ', '
        print(loss_str[:-2])

        if kwargs.get('accuracy') is not None:
            print('Test accuracy is: {}'.format(kwargs.get('accuracy')))

        # Print the results
        y = kwargs.get('y')
        if (y is not None) and (y_ is not None) and (x is not None):
            max_len = min(len(y_[0]), 3)
            print('Actual: {}, Pred: {}, Input: {}'.format(
                    y_[0][0:max_len],
                    y[0][0:max_len],
                    x[0][0:max_len]))
        elif (y is not None) and (y_ is not None):
            max_len = min(len(y_[0]), 3)
            print('Actual: {}, Pred: {}'.format(
                    y_[0][0:max_len],
                    y[0][0:max_len]))
        else:
            str1 = ''
            if y_ is None:
                str1 += 'Y_ is none. '
            if x is None:
                str1 += 'X is none. '
            raise ValueError(str1+'Input values are invalid.'
                             + '\ndictionary is: {}'.format(kwargs))


class TransferLearnTrainer(Trainer):
    def __init__(self,
                 sess,

                 # Options for initialization
                 parameter_dict=None,
                 MO_options=None,
                 training_info=None,
                 optimizer_sharing='m',
                 prev_meta_opt=None,

                 # For the transfer_learning problem
                 transfer_options=None, 

                 # Printing, graphs, etc.
                 summaries=None,

                 # Options for the specific problem type
                 loss_type=lambda y_, y: tf.reduce_mean(tf.abs(y_ - y)),
                 accuracy_func=lambda y_, y: tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1)), tf.float32)),
                 ):
        '''Initialize a Transfer Learn Trainer object

        Args: 
            sess             : TensorFlow session.
            parameter_dict   : Params for gatenet.
            MO_options       : Params for the meta optimizer
            training_info    : Training options (eg. batch size,)
            optimizer_sharing: The type of sharing to use for MO
            prev_meta_opt    : (Optionally) use previously initialized MO
            summaries        : Save tf summaries
            loss_type        : A callable taking in prediction & label returning tf (all tf nodes)
            accuracy_func    : A callable taking in prediction & label returning accuracy (all tf nodes)
        '''
        self.transfer_options = transfer_options
        super(TransferLearnTrainer, self).__init__(sess,
                                                   parameter_dict=parameter_dict,
                                                   MO_options=MO_options,
                                                   training_info=training_info,
                                                   optimizer_sharing=optimizer_sharing,
                                                   prev_meta_opt=prev_meta_opt,
                                                   summaries=summaries,
                                                   loss_type=loss_type,
                                                   accuracy_func=accuracy_func)

    ##### HELPERS  #############################################################
    def _data_manager_init(self):
        self._dm = SimpleProbTransferLearnDataManager(**self.transfer_options)

    def _get_placeholder_shapes(self): ## B TODO: remove hardcoding ##################
        # NOTE: Right now this is hardcoded for the simple problem!
        return (None, 10), (None, 10) # (batch_size, in/output dim)

    ##### Outer Loop Training ##################################################
    def meta_train(self):
        '''Train the meta optimizer for the transfer learning problem. This is
        the outermost loop of of meta-training.

        What it does each loop:
            - Clear/reset gatenet (the optimizee network)
            - Build new dataset with the desired sizes/shapes
            - Run transfer training loop
        '''
        print('\nmeta_train - outerloop') # REMOVE
        # num_datasets = Number of training iterations for the meta optimizer
        num_datasets = self.training_info['num_datasets']
        for ds_iter in range(num_datasets):
            self.current_dataset = ds_iter # Set ds iteration (for loss history)

            # Reset gatenet (optimizee network)
            self.graph.reset_graph(self.sess)

            # Build new transfer dataset
            self._dm.build_dataset(transfer_type='train')

            # Run transfer training (middle) loop
            self._transfer_train_loop()

    def meta_val(self):
        raise NotImplementedError()

    def meta_test(self):
        raise NotImplementedError()

    ##### Middle Loop Training #################################################
    @staticmethod
    def _create_data_getter(previous_test_inputs, previous_test_labels):
        '''Return a callable data getter for the prev test data. Make sure that
        there is more examples in prev_test than batch_size for the callable.
        '''
        def build(batch_size, x=previous_test_inputs, y=previous_test_labels):
            # Get batch_size random examples (with replacement)
            x = np.array(x)
            y = np.array(y)
            shuffled_indices = np.arange(x.shape[0])
            np.random.shuffle(shuffled_indices)
            shuffled_indices = shuffled_indices[:batch_size]
            return (x[shuffled_indices, :],
                    y[shuffled_indices, :])

        # Return None if there is no previous test input
        if len(previous_test_inputs) > 0:
            return build
        else:
            return None

    def _transfer_train_loop(self):
        '''Do transfer between tasks. This is a training iteration of the
        metaoptimizer.

        What it does each loop:
            - Get a new task (either single for simple prob or split )
            - Train the Gatenet (optimizee) for a specific task
            - Evaluate Gatenet performance on the task's test set
            - Sample test examples from previously trained task (in current
              dataset)
                * These sampled examples are to find catastrophic forgetting loss
                * A2 TODO: Should the additional loss be scaled on how many tasks 
                        were completed?
            - Add the test set for current task into the previous task examples
        '''
        print('\ttransfer_train_loop - middle loop')
        # Create a previous task (test) dataset for additional loss
        # A3 TODO (experiment): Possibility skip some tasks for the additional loss. This might increase generalization?
        previous_test_inputs = []
        previous_test_labels = []

        # Get the number of tasks to run.
        num_tasks = self.training_info['tasks_per_dataset']

        for task_iter in range(num_tasks):
            self.current_task = task_iter # Set the current task (for loss history)
            self._dm.build_task()

            # A callable data getter for current and additional losses
            train_dg = self._dm.get_test_batch
            # Train the next task
            self._train_optimizee_loop(data_getter=train_dg)

            # Callable data getters for current and previous test losses
            test_dg = self._dm.get_test_batch
            prev_test_dg = self._create_data_getter(previous_test_inputs, previous_test_labels)


            #Added here
            # Add the current task's test data to previous_test_inputs & labels
            prev_in, prev_label = self._dm.get_all_test_data()
            previous_test_inputs.append(prev_in)
            previous_test_labels.append(prev_label)
            #End added here


            # Update the metaoptimizer
            self._update_meta_optimizer(data_getter=test_dg, additional_data_getter=prev_test_dg)
            
            '''#TASKADDI
            # Add the current task's test data to previous_test_inputs & labels
            prev_in, prev_label = self._dm.get_all_test_data()
            previous_test_inputs.append(prev_in)
            previous_test_labels.append(prev_label)'''


    ##### Inner Loop Training ##################################################
    def _train_optimizee_loop(self, data_getter):
        '''Train Gatenet (optimizee). This trains Gatenet for a single task
    
        Args:
            data_getter: A callable that takes batch_size as an argument, and
                         returns x and y_ as numpy arrays, each shaped 
                         (batch_size, in/output dim).

        What it does each loop:
            - Train step of Gatenet using the meta optimizer
            - Print progress at certain iterations
        '''
        print('\t\t train - inner loop')
        print_every = self.training_info['print_every']
        num_batches = self.training_info['num_batches']
        batch_size  = self.training_info['batch_size']
        for i in range(num_batches):
            tr_input, tr_label = data_getter(batch_size)

            if self.task_steps % print_every == 0:
                vals, printable_vals = self._train_step(
                        self.sess,
                        feed_dict={self.input: tr_input, self.label: tr_label},
                        train_step=self.train_step,
                        loss=self.loss,
                        meta_loss=self.meta_loss_train,
                        merged=self.merged,
                        accuracy=self.accuracy,
                        y=self.pred)
                self._print_progress(x=tr_input, y_=tr_label, **printable_vals)
                
                # Update training history
                self.loss_history.append((
                        self.current_dataset,
                        self.current_task,
                        self.task_steps,
                        vals.get('loss'),
                        vals.get('meta_loss'),
                    ))
                self.global_steps += 1
                self.task_steps += 1

            else:
                self._train_step(
                        self.sess,
                        feed_dict={self.input: tr_input, self.label: tr_label},
                        train_step=self.train_step)
                self.global_steps += 1
                self.task_steps += 1

    def _update_meta_optimizer(self, data_getter, additional_data_getter=None):
        '''Test the performance of Gatenet for the current task, and do an
        update step for the meta optimizer.

        Args:
            data_getter: Callable data getter for the test data of current task.
            additional_data_getter: Callable data getter for the test data of 
            previously trained tasks in the same dataset

        What it does:
            - Train step of the meta optimizer
            - Print progess and test performance
        '''
        print('\t\t train - inner loop')
        ts_input, ts_label = data_getter(self.batch_size)

        if additional_data_getter is not None:
            train_step_meta = self.train_step_meta
            meta_loss = self.meta_loss_test
            a_ts_input, a_ts_label = additional_data_getter(self.batch_size)
            a_loss = self.a_loss
            feed_dict = {
                    self.input  : ts_input,
                    self.label  : ts_label,
                    self.a_input: a_ts_input,
                    self.a_label: a_ts_label,
            }
        else:
            a_loss = self.a_loss #TASKADDI None
            feed_dict = {
                    self.input  : ts_input,
                    self.label  : ts_label,
                    self.a_input: ts_input,
                    self.a_label: ts_label,
            }

        vals, printable_vals = self._train_step(
                self.sess,
                feed_dict=feed_dict,
                train_step_meta=self.train_step_meta,
                meta_loss=self.meta_loss_test,
                a_loss=a_loss,
                merged=self.merged,
                accuracy=self.accuracy,
                y=self.pred)
        self._print_test(x=ts_input, y_=ts_label, **printable_vals)


def main():
    # All of the default options
    shape = input_shape = output_shape = 10
    num_datasets = 4
    num_transfer_classes = 27
    examples_per_class = 50
    print_every = 30
    num_batches = 100
    batch_size = 20

    parameter_dict = {
            'C': output_shape,
            'sublayer_type': 'AdditionSublayerModule',
            'hidden_size': 20,
            'gamma': 2.,
            'M': 3,
            'L': 3,
            'module_type': 'Perceptron',
    }
    '''
    MO_options = {
            'optimizer_type':'CoordinateWiseLSTM',
            'second_derivatives':False,
            'params_as_state':False,
            'rnn_layers':(4,4),
            'len_unroll':3,
            'w_ts':[0.33 for _ in range(3)],
            'lr':0.001,
            'meta_lr':0.01,
            'save_summaries':{},
            'name':'MetaOptimizer'
    }
    '''
    MO_options = {
            'optimizer_type':'CoordinateWiseLSTM',
            'second_derivatives':False,
            'params_as_state':False,
            'rnn_layers':(4,4),
            'len_unroll':3,
            'w_ts':[0.33 for _ in range(3)],
            'lr':0.001,
            'meta_lr':0.01,
            'load_from_file':[
                    './save/meta_opt_identity_initialization', # 0
                    './save/meta_opt_identity_initialization', # 1
                    './save/meta_opt_identity_initialization', # 2
                    './save/meta_opt_identity_initialization', # 3
                    './save/meta_opt_identity_initialization', # 4
                    './save/meta_opt_identity_initialization', # 5
                    './save/meta_opt_identity_initialization', # 6
                    './save/meta_opt_identity_initialization', # 7
                    './save/meta_opt_identity_initialization', # 8
                    './save/meta_opt_identity_initialization', # 9
                    './save/meta_opt_identity_initialization', # 10
                    './save/meta_opt_identity_initialization', # 11
                    './save/meta_opt_identity_initialization', # 12
            ],
            'save_summaries':{},
            'name':'MetaOptimizer'
    }
    #'''
    training_info = {
            'batch_size':batch_size,
            'num_batches':num_batches,
            'print_every':print_every,
            'num_datasets':num_datasets, # Number of times to get a new dataset
            'classes_per_dataset':num_transfer_classes, # Classes per dataset
            'tasks_per_dataset':num_transfer_classes, # Tasks per dataset
    }
    '''meta = {
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
    }'''
    transfer_options = {
            'dataset':'simple-linear',
            'num_transfer_classes':num_transfer_classes,
            'task_classes':1,
            'data_split_meta':{'train':0.75, 'val': 0.0, 'test':0.25},
            'data_split_task':{'train': 0.7, 'test': 0.3},
            'examples_per_class':examples_per_class, ## B TODO: this number is a hyper param ##
            'num_datasets':num_datasets
    }

    with tf.Session() as sess:
        tlt = TransferLearnTrainer(sess,
                                   parameter_dict=parameter_dict,
                                   MO_options=MO_options,
                                   training_info=training_info,
                                   optimizer_sharing='m',
                                   transfer_options=transfer_options)
        tlt.meta_train()

if __name__ == '__main__':
    main()
