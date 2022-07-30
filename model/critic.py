import tensorflow as tf


class Critic(tf.keras.layers.Layer):
    def __init__(self,
                 hidden_sizes,
                 hidden_activation='tanh',
                 output_activation=None,
                 learning_rate=1e-3):

        super(Critic, self).__init__()

        #self.dense_layers = \
        #    [tf.keras.layers.Dense(hidden_sizes[i], activation=hidden_activation) for i in range(len(hidden_sizes))]

        self.output_layer = tf.keras.layers.Dense(1, activation=output_activation)

    def call(self, states):

        # output = inputs

        # for dense in self.dense_layers:
        #    output = dense(output)

        output = self.output_layer(states)
        output = tf.squeeze(output, axis=1)

        return output
