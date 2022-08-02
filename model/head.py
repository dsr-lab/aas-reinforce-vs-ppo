import tensorflow as tf


class Head(tf.keras.layers.Layer):

    def __init__(self,
                 n_outputs,
                 hidden_sizes=None,
                 hidden_activation='tanh',
                 output_activation=None,
                 kernel_initializer='glorot_uniform'
                 ):

        super(Head, self).__init__()

        self.dense_layers = []

        if hidden_sizes is not None:
            self.dense_layers = \
                [tf.keras.layers.Dense(hidden_sizes[i], activation=hidden_activation) for i in range(len(hidden_sizes))]

        self.output_layer = \
            tf.keras.layers.Dense(n_outputs, activation=output_activation, kernel_initializer=kernel_initializer)

    def call(self, states):

        output = states

        for dense in self.dense_layers:
            output = dense(output)

        output = self.output_layer(output)

        return output
