from binary_mnist_pathnet import *


def train_gate_network(parameter_dict, input_data, input_labels, test_data, test_labels, sess, use_gates=1):
    # Test Gate - Softmax Predictions
    tr_label = input_labels
    gates_train = input_data
    #gates_train *= 1000
    if use_gates == 1:
        input_size = parameter_dict['L'] * parameter_dict['M']
    else:
        input_size = 28 * 28

    # W = weight_variable([gates_train.shape[1], 10])
    h_layer_size = 100
    W1 = weight_variable([input_size, h_layer_size])
    b1 = bias_variable([h_layer_size])

    W2 = weight_variable([h_layer_size, 10])
    b2 = bias_variable([10])

    gate_x = tf.placeholder(tf.float32, [None, input_size])
    gate_y1 = tf.nn.relu(tf.matmul(gate_x, W1) + b1)
    gate_y = tf.matmul(gate_y1, W2) + b2
    gate_y_ = tf.placeholder(tf.float32, [None, 10])

    # Accuracy
    correct_prediction = tf.equal(tf.argmax(gate_y, 1), tf.argmax(gate_y_, 1))
    gate_accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    loss = tf.nn.softmax_cross_entropy_with_logits(labels=gate_y_, logits=gate_y)
    gate_train_step = tf.train.AdamOptimizer(parameter_dict['learning_rate']).minimize(loss)

    # Note that this resets the variables
    tf.global_variables_initializer().run()

    for i in range(parameter_dict['num_batches']):
        if use_gates == 1:
            data = gates_train
            label = tr_label
        else:
            data = gates_train
            label = tr_label

        idx = list(range(len(label)))
        np.random.shuffle(idx)
        data = data[idx]
        label = label[idx]

        data = data[:100]
        label = label[:100]

        if i % 100 == 0:
            acc = sess.run(gate_accuracy, feed_dict={gate_x: data, gate_y_: label})
            print('training %d, accuracy %g' % (i, acc))

        sess.run(gate_train_step, feed_dict={gate_x: data, gate_y_: label})

    test_accuracy = sess.run(gate_accuracy, feed_dict={gate_x: test_data, gate_y_: test_labels})

    return test_accuracy