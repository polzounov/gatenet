import numpy as np
import tensorflow as tf
from graph.graph import Graph
from l2l.metaoptimizer import *
from dataset_loading.data_managers import SimpleProbTransferLearnDataManager


def _get_callable_loss_func(x, y_, fx, loss_type):
    def loss_func(x=x,
                  y_=y_,
                  fx=fx,
                  loss_type=loss_type,
                  mock_func=None,
                  var_dict=None):
        def build():
            y = fx(x)
            with tf.name_scope('loss'):
                return loss_type(y_, y)

        if (mock_func is not None) and (var_dict is not None):
            return mock_func(build, var_dict)

        return build

    return loss_func


def _run_sess(sess, feed_dict, **kwargs):
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


def _train_step(sess, feed_dict=None, train_step=None, train_step_meta=None, loss=None, meta_loss=None, a_loss=None, y=None):
    vals = _run_sess(sess,
                     feed_dict,
                     train_step=train_step,
                     train_step_meta=train_step_meta,
                     loss=loss,
                     meta_loss=meta_loss,
                     a_loss=a_loss,
                     y=y)
    printable_vals = {
        'meta_loss': vals.get('meta_loss'),
        'a_loss'   : vals.get('a_loss'),
        'loss'     : vals.get('loss'),
        'y'        : vals.get('y'),
    }
    return vals, printable_vals


def _create_data_getter(previous_test_inputs, previous_test_labels):
    def build(batch_size, x=previous_test_inputs, y=previous_test_labels):
        # Get batch_size random examples (with replacement)
        x = np.array(x)
        y = np.array(y)
        shuffled_indices = np.arange(x.shape[0])
        np.random.shuffle(shuffled_indices)
        shuffled_indices = shuffled_indices[:batch_size]
        
        input_dim = x.shape[-1]
        label_dim = y.shape[-1]
        return (x[shuffled_indices, :].reshape(-1, input_dim),
                y[shuffled_indices, :].reshape(-1, label_dim))

    # Return None if there is no previous test input
    if len(previous_test_inputs) > 0:
        return build
    else:
        return None


def _print_test(current_task=-1, task_steps=-1, global_steps=-1, x=None, y_=None, **kwargs):
    print('\nTest for task {}'.format(current_task))

    # Print how many steps taken
    print('Task Steps: {}, Total steps: {}'.format(
            task_steps, global_steps))

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
        raise ValueError(str1 + 'Input values are invalid.'
                         + '\ndictionary is: {}'.format(kwargs))


def _print_progress(current_task=-1, task_steps=-1, global_steps=-1, current_dataset=-1, x=None, y_=None, **kwargs):
    # Print if new task or dataset
    if task_steps == 0:
        if current_task == 0:
            print('\n\n###################################################',
                  '######################################################')
            print('\nNew Dataset {}'.format(current_dataset))
        print('\n\nNew task {}'.format(current_task))

    # Print how many steps taken
    print('Task Steps: {}, Total steps: {}'.format(
            task_steps, global_steps))

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


def train(sess,
          parameter_dict=None,
          MO_options=None,
          training_info=None,
          optimizer_sharing=None,
          transfer_options=None,
          prev_meta_opt=None,
          summaries=None,
          loss_type=lambda y_, y: tf.reduce_mean(tf.abs(y_ - y))):
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
    '''
    ##### Misc Vars ########################################################
    sess = sess
    num_datasets = training_info['num_datasets']
    print_every = training_info['print_every']
    num_batches = training_info['num_batches']
    batch_size = training_info['batch_size']

    current_dataset = 0  # The current dataset being used (from 0 to num_datasets)
    current_task = 0  # The current task being trained (from 0 to tasks_per_dataset)
    loss_history = []  # Overall history of training, (dataset_iter, current_task, task_steps, loss, meta_loss)
    global_steps = 0  # Total number of update steps for Gatenet
    task_steps = 0  # Number of steps taken for current task

    # Initialize the data manager (transfer or meta learn data manager)
    _dm = SimpleProbTransferLearnDataManager(**transfer_options)

    ##### Set up the graph #################################################
    # Input placeholders
    input_shape, label_shape = (None, 10), (None, 10)  # _get_placeholder_shapes()
    o_input = tf.placeholder(tf.float32, input_shape, name='x-input')
    label = tf.placeholder(tf.float32, label_shape, name='y-input')
    a_input = tf.placeholder(tf.float32, input_shape, name='additional-x-input')
    a_label = tf.placeholder(tf.float32, label_shape, name='additional-y-input')

    # Initialize the graph
    graph = Graph(parameter_dict)

    loss_func = _get_callable_loss_func(o_input, label, graph.return_logits, loss_type)
    a_loss_func = _get_callable_loss_func(a_input, a_label, graph.return_logits, loss_type)

    # Get the prediction, loss, and accuracy
    y = graph.return_logits(o_input)
    pred = y

    with tf.name_scope('loss'):
        loss = loss_func()()
        a_loss = a_loss_func()()

    ##### Meta Optimizer Setup #############################################
    # Get the previous meta optimizer's variables
    MO_options['load_from_file'] = prev_meta_opt

    # Get module wise variable sharing for the meta optimizer
    MO_options['shared_scopes'] = graph.scopes(scope_type='m')

    # Meta optimization
    optimizer = MetaOptimizer(**MO_options)
    train_step, meta_loss_train = optimizer.train_step(loss_func=loss_func)
    train_step_meta, meta_loss_test = optimizer.train_step_meta(
            loss_func=loss_func,
            additional_loss_func=a_loss_func)

    # Initialize Variables
    tf.global_variables_initializer().run()

    for ds_iter in range(num_datasets):
        current_dataset = ds_iter  # Set ds iteration (for loss history)
        # Reset gatenet (optimizee network)
        graph.reset_graph(sess)
        # Build new transfer dataset
        _dm.build_dataset(transfer_type='train')

        ########################################################################
        # Run transfer training (middle) loop
        previous_test_inputs = []
        previous_test_labels = []

        # Get the number of tasks to run.
        num_tasks = training_info['tasks_per_dataset']

        for task_iter in range(num_tasks):
            current_task = task_iter  # Set the current task (for loss history)
            _dm.build_task()

            # A callable data getter for current and additional losses
            train_dg = _dm.get_test_batch

            # Add the current task's test data to previous_test_inputs & labels
            prev_in, prev_label = _dm.get_all_test_data()
            previous_test_inputs.append(prev_in)
            previous_test_labels.append(prev_label)

            # Callable data getters for current and previous test losses
            test_dg = _dm.get_test_batch
            prev_test_dg = _create_data_getter(previous_test_inputs, previous_test_labels)

            ####################################################################
            print('\t\t train - inner loop')
            for i in range(num_batches):
                tr_input, tr_label = train_dg(batch_size)

                if task_steps % print_every == 0:
                    vals, printable_vals = _train_step(
                            sess,
                            feed_dict={o_input: tr_input, label: tr_label},
                            train_step=train_step,
                            loss=loss,
                            meta_loss=meta_loss_train,
                            y=pred)
                    _print_progress(x=tr_input, y_=tr_label, **printable_vals)

                    '''
                    # Update training history
                    loss_history.append((
                            current_dataset,
                            current_task,
                            task_steps,
                            vals.get('loss'),
                            vals.get('meta_loss'),
                        ))
                    global_steps += 1
                    task_steps += 1
                    '''

                else:
                    _train_step(
                            sess,
                            feed_dict={o_input: tr_input, label: tr_label},
                            train_step=train_step)
                    '''
                    global_steps += 1
                    task_steps += 1
                    '''

            ####################################################################
            print('\t\t test - inner loop')
            ts_input, ts_label = test_dg(batch_size)
            a_ts_input, a_ts_label = prev_test_dg(batch_size)
            feed_dict = {
                o_input: ts_input,
                label  : ts_label,
                a_input: a_ts_input,
                a_label: a_ts_label,
            }
            vals, printable_vals = _train_step(
                    sess,
                    feed_dict=feed_dict,
                    train_step_meta=train_step_meta,
                    meta_loss=meta_loss_test,
                    a_loss=a_loss,
                    y=pred)
            _print_test(x=ts_input, y_=ts_label, **printable_vals)


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
        'C'            : output_shape,
        'sublayer_type': 'AdditionSublayerModule',
        'hidden_size'  : 20,
        'gamma'        : 2.,
        'M'            : 3,
        'L'            : 3,
        'module_type'  : 'Perceptron',
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
        'optimizer_type'    : 'CoordinateWiseLSTM',
        'second_derivatives': False,
        'params_as_state'   : False,
        'rnn_layers'        : (4, 4),
        'len_unroll'        : 3,
        'w_ts'              : [0.33 for _ in range(3)],
        'lr'                : 0.001,
        'meta_lr'           : 0.01,
        'load_from_file'    : [
            './save/meta_opt_identity_initialization',  # 0
            './save/meta_opt_identity_initialization',  # 1
            './save/meta_opt_identity_initialization',  # 2
            './save/meta_opt_identity_initialization',  # 3
            './save/meta_opt_identity_initialization',  # 4
            './save/meta_opt_identity_initialization',  # 5
            './save/meta_opt_identity_initialization',  # 6
            './save/meta_opt_identity_initialization',  # 7
            './save/meta_opt_identity_initialization',  # 8
            './save/meta_opt_identity_initialization',  # 9
            './save/meta_opt_identity_initialization',  # 10
            './save/meta_opt_identity_initialization',  # 11
            './save/meta_opt_identity_initialization',  # 12
        ],
        'save_summaries'    : {},
        'name'              : 'MetaOptimizer'
    }
    # '''
    training_info = {
        'batch_size'         : batch_size,
        'num_batches'        : num_batches,
        'print_every'        : print_every,
        'num_datasets'       : num_datasets,  # Number of times to get a new dataset
        'classes_per_dataset': num_transfer_classes,  # Classes per dataset
        'tasks_per_dataset'  : num_transfer_classes,  # Tasks per dataset
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
        'dataset'             : 'simple-linear',
        'num_transfer_classes': num_transfer_classes,
        'task_classes'        : 1,
        'data_split_meta'     : {'train': 0.75, 'val': 0.0, 'test': 0.25},
        'data_split_task'     : {'train': 0.7, 'test': 0.3},
        'examples_per_class'  : examples_per_class,  ## B TODO: this number is a hyper param ##
        'num_datasets'        : num_datasets
    }

    with tf.Session() as sess:
        train(sess,
              parameter_dict=parameter_dict,
              MO_options=MO_options,
              training_info=training_info,
              optimizer_sharing='m',
              transfer_options=transfer_options)

if __name__ == '__main__':
    main()