import tensorflow as tf
import numpy as np
import os

from l2l import networks
import pickle


class RNN():
    def __init__(self,
                 rnn_type='cwlstm',
                 output_size=None,
                 layers=None,
                 lr=1.,
                 load_from_file=None):
        '''Creates an rnn'''
        # Optional load from file
        net_init = None
        if load_from_file is not None:
            with open(load_from_file, "rb") as f:
                net_init = pickle.load(f)
            
        Type = self._get_optimizer(rnn_type)
        self.rnn = Type(output_size=output_size,
                        layers=layers,
                        scale=lr,
                        initializer=net_init)
        self.prev_state = None
    
    def _get_optimizer(self, optimizer_name):
        '''Returns the optimizer class, works for strings too'''
        if isinstance(optimizer_name, str): # PYTHON 3 ONLY
            optimizer_name = optimizer_name.lower()
            optimizer_mapping = {
                'cw': networks.CoordinateWiseDeepLSTM,
                'cwlstm': networks.CoordinateWiseDeepLSTM,
                'coordinatewiselstm': networks.CoordinateWiseDeepLSTM,
                'lstm': networks.StandardDeepLSTM,
                'standardlstm': networks.StandardDeepLSTM,
                'kernel': networks.KernelDeepLSTM,
                'kernellstm': networks.KernelDeepLSTM,
                'sgd': networks.Sgd,
                'adam': networks.Adam,
            }
            return optimizer_mapping[optimizer_name]
        else:
            # If not string assume optimizer_name is a class
            return optimizer_name

    def save(self, sess, path='./save/lstm_identity'):
        '''Save meta-optimizer.'''
        filename = path + '.l2l'
        net_vars = networks.save(self.rnn, sess, filename=filename)
        return net_vars

    def run(self, inputs):
        if self.prev_state is None:
            self.prev_state = self.rnn.initial_state_for_inputs(inputs)
        outputs, self.prev_state = self.rnn(inputs, self.prev_state)            
        return outputs


def pr(*args, max_len=150): # Accept either list, tuple, or multiple arguments
    if len(args) == 1:
        nums = args[0]
    elif len(args) > 1:
        nums = [arg for arg in args]
    else:
        raise ValueError('Must have arguments')
    lens = [len(str(num)) for num in nums] + [max_len]
    l = np.amin(lens)
    return [str(num)[:l] for num in nums]


################################################################################
################                  MAIN                      ####################
################################################################################

def iden(batch_size, size):
    x = y = np.random.rand(batch_size, size)
    return x, y


def train(num_batches,
          batch_size,
          size,
          lr,
          layers,
          print_every,
          rnn_type):

    
    sess = tf.InteractiveSession()

    # Placeholders
    x  = tf.placeholder(tf.float32, [batch_size, size], name='x-input')
    y_ = tf.placeholder(tf.float32, [batch_size, size], name='y-input')

    # Init RNN
    rnn = RNN(rnn_type=rnn_type,
              output_size=size,
              layers=layers)
    y = rnn.run(x)

    l2_reg_func = tf.contrib.layers.l2_regularizer(0.1, scope=None)
    reg = tf.contrib.layers.apply_regularization(l2_reg_func, 
        weights_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES))

    loss = tf.reduce_mean(tf.reduce_sum((y_ - y) ** 2) + reg)
    tf.summary.scalar('loss', loss)

    diff = tf.reduce_mean(tf.abs(y_ - y))
    tf.summary.scalar('average_diff', diff)

    # Optimize RNN
    optimizer = tf.train.AdamOptimizer(lr)
    train_step = optimizer.minimize(loss)

    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter('./logs/identity', graph=sess.graph)
    # Command to run: tensorboard --logdir=experiments/logs/identity

    # Init Vars
    tf.initialize_all_variables().run()#tf.global_variables_initializer()

    ##### Training loop ######
    for i in range(num_batches):
            # Get data
            tr_data, tr_label = iden(batch_size, size)
    
            if i % print_every == 0:
                # Run RNN
                pred, loss_, diff_, summary, _ = sess.run([y, loss, diff, merged, train_step],
                                                    feed_dict={x: tr_data, y_:tr_label})    

                # Save progress
                writer.add_summary(summary, i)

                # Print progress
                print('\n\nIteration: {}, Loss: {}, Average Diff: {}'.format(i, loss_, diff_))
                for i in range(min(pred.shape[1], 10)):
                    pred1, input1, act1 = pr(pred[0][i], tr_data[0][i], tr_label[0][i])
                    #pred2, input2, act2 = pr(pred[1][i], tr_data[1][i], tr_label[1][i])
                    #print('Pred1: {}, Act1: {} --- Pred2: {}, Act2: {}'.format(pred1, act1, pred2, act2)
                    print('Pred: {}, Actual: {}, Input {}'.format(pred1, act1, input1))

            else:
                # Run RNN
                pred, loss_, diff_, _ = sess.run([y, loss, diff, train_step],
                                            feed_dict={x: tr_data, y_:tr_label})

    rnn.save(sess, path='./save/lstm_identity')


if __name__ == "__main__":

    train(num_batches=10000,
          batch_size=10,
          size=10,
          lr=1.0,
          layers=(4,4),
          print_every=10,
          rnn_type='cw')
