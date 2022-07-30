import tensorflow as tf


class Actor(tf.keras.layers.Layer):

    def __init__(self,
                 hidden_sizes,
                 n_actions,
                 hidden_activation='tanh',
                 output_activation=None,  # 'softmax',
                 ):

        super(Actor, self).__init__()

        # self.dense_layers = \
        #    [tf.keras.layers.Dense(hidden_sizes[i], activation=hidden_activation) for i in range(n_actions)]

        self.output_layer = tf.keras.layers.Dense(n_actions, activation=output_activation)

    def call(self, states):

        # output = inputs

        # for dense in self.dense_layers:
        #    output = dense(output)

        output = self.output_layer(states)

        return output
