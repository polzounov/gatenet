import numpy as np
import preprocessing
import tensorflow as tf
from parameters import Parameters
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets

def get_next_batch():
    mnist = read_data_sets(Parameters.data_dir,
                                      one_hot=True,
                                      fake_data=Parameters.fake_data)

    total_tr_data, total_tr_label = mnist.train.next_batch(mnist.train._num_examples);

    return total_tr_data, total_tr_label

    # Gathering a1 Data
    tr_data_a1 = total_tr_data[(total_tr_label[:, Parameters.a1] == 1.0)];
    tr_data_a2 = total_tr_data[(total_tr_label[:, Parameters.a2] == 1.0)];

    #preprocessing.preprocess_data(tr_data_a1)
    #preprocessing.preprocess_data(tr_data_a2)

    tr_data1 = np.append(tr_data_a1, tr_data_a2, axis=0);
    tr_label1 = np.zeros((len(tr_data1), 2), dtype=float);
    for i in range(len(tr_data1)):
        if (i < len(tr_data_a1)):
            tr_label1[i, 0] = 1.0;
        else:
            tr_label1[i, 1] = 1.0;


    return tr_data1, tr_label1