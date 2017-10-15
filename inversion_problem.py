import numpy as np
import tensorflow as tf
from graph.graph import Graph
from tensorflow_utils import *
import matplotlib.pyplot as plt

from dataset_loading.create_datasets import LinearProblemGenerator


def limit_len(a, limit=10):
    return str(a)[0:limit]


def smooth(history, average_len=10):
    history = history.reshape(-1, average_len, 2)
    history = np.mean(history, axis=1)
    history.reshape(-1, 2)
    return history


def train(parameter_dict, size, total_examples, print_every):
    lr = parameter_dict['learning_rate']
    hidden_size = output_size = input_size = size

    # Get train examples
    lpg = LinearProblemGenerator(dim=hidden_size, examples_per_class=total_examples, mat_shape=(3,1), num_datasets=1)
    train_inputs, train_labels = lpg.get_datasets(n=1)

    def get_batch(batch_size, x=train_inputs, y=train_labels, total_examples=total_examples):
        indices = np.random.choice(total_examples, size=batch_size)
        return x[0, indices, :], y[0, indices, :]

    # Probably switch to Session rather than InteractiveSession later on
    # Start session
    sess = tf.InteractiveSession()

    # Input placeholders
    x = tf.placeholder(tf.float32, [None, input_size], name='x-input')
    y_ = tf.placeholder(tf.float32, [None, output_size], name='y-input')
    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.train.exponential_decay(learning_rate=lr,
                                               global_step=global_step,
                                               decay_steps=10000,
                                               decay_rate=0.92,
                                               staircase=True,
                                               name=None)

    # Build computation graph
    graph = Graph(parameter_dict)
    y = graph.return_logits(x)

    # Cross Entropy
    with tf.name_scope('mse'):
        diff = tf.losses.mean_squared_error(labels=y_, predictions=y)
        loss = tf.reduce_mean(diff)

    # GradientDescent
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=global_step)

    # Initialize Variables
    tf.global_variables_initializer().run()

    writer = tf.summary.FileWriter('./logs', graph=tf.get_default_graph())
    # Command to run: tensorboard --logdir=logs

    history = np.zeros((parameter_dict['num_batches']/print_every, 2))

    for i in range(parameter_dict['num_batches']):
        tr_data, tr_label = get_batch(parameter_dict['batch_size'])

        if i % print_every == 0:
            mse_loss, pred, lr = sess.run([loss, y, learning_rate], feed_dict={x: tr_data, y_: tr_label})
            print('\nTraining {}, loss {}, lr {}'.format(i, mse_loss, lr))
            history[int(i/print_every-1),:] = [i, mse_loss]
            for j in range(min(tr_label.shape[1], 10)):
                print('\tPred {}, Actual {}, Input {}'.format(limit_len(pred[0,j]), limit_len(tr_label[0,j]), limit_len(tr_data[0,j])))

        sess.run(train_step, feed_dict={x: tr_data, y_: tr_label})

    return history


def main():
  size = 5
  total_examples = 3000
  print_every = 10
  hidden_size = output_size = size

  parameter_dict = {
    'C': output_size,
    'hidden_size': hidden_size*2,
    'gamma': 3.,
    'batch_size': 100,
    'num_batches': 30000,
    'learning_rate': 0.01,
    # 'learning_rate': 0.001,
    'output_file': 'test',
    'M': 1,
    'L': 3,
    'sublayer_type': 'AdditionSublayerModule',
    'module_type': 'PerceptronModule'
  }

  history = train(parameter_dict=parameter_dict,
                  size=size,
                  total_examples=total_examples,
                  print_every=print_every)[1000-1:-1]
  plt.plot(history[:, 0], history[:, 1])
  plt.show()

  history = smooth(history, average_len=100)
  plt.plot(history[:, 0], history[:, 1])
  plt.show()


if __name__ == "__main__":
    main()