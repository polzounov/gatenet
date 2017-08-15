
import tensorflow as tf
import sonnet as snt
import numpy as np

class DeepLSTM(snt.AbstractModule):
    """docstring for LSTM"""
    def __init__(self,
                 output_size,
                 batch_size=1,
                 layers=(5, 5), # Each el is the hidden size of the respective RNN cell (for deep RNNs)
                 scale=1.0,
                 name='LSTM'):
        super(DeepLSTM, self).__init__(name=name)
        self._batch_size = batch_size
        self._layers = layers
        self._scale = scale

        # Create cores (layers of a single RNN timestep)
        self._cores = []
        for i, size in enumerate(self._layers, start=1):
            name = "lstm_{}".format(i)

            ######################
            ## Test Code
            init = None
            ######################

            self._cores.append(snt.LSTM(size, name=name, initializers=init)) # TODO INIT

        # Create the Deep RNN
        self._rnn = snt.DeepRNN(self._cores, skip_connections=False, name="deep_rnn")

        ######################
        ## Test Code
        init = None
        ######################

        # Linear layer to get the output
        self._linear = snt.Linear(output_size, name="linear", initializers=init) # TODO INIT

        # Set the initial hidden state
        self.prev_state = self._rnn.initial_state(self._batch_size)

    def _build(self, inputs):
        ##inputs = self._preprocess(tf.expand_dims(inputs, -1))
        # Incorporates preprocessing into data dimension
        inputs = tf.reshape(inputs, [inputs.get_shape().as_list()[0], -1])
        output, next_state = self._rnn(inputs, self.prev_state)
        self.prev_state = next_state
        return self._linear(output) * self._scale



class CoordinateWiseDeepLSTM(DeepLSTM):
    """docstring for LSTM"""

    def __init__(self,
                 output_size,
                 batch_size=1,
                 layers=(5, 5),  # Each el is the hidden size of the respective RNN cell (for deep RNNs)
                 scale=1.0,
                 name='LSTM'):
        super(CoordinateWiseDeepLSTM, self).__init__(output_size,batch_size, layers, scale,name)


    def _build(self, inputs):
        ##inputs = self._preprocess(tf.expand_dims(inputs, -1))
        # Incorporates preprocessing into data dimension

        outputs = inputs
        for i in range(inputs.get_shape().as_list()[0]):
            input = inputs[inputs.get_shape().as_list()[0]][i]

            output, next_state = self._rnn(inputs, self.prev_state[i])
            self.prev_state[i] = next_state[i]
            outputs[i] = output
        return self._linear(outputs) * self._scale



deep = CoordinateWiseDeepLSTM((5,5))