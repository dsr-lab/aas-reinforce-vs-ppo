import tensorflow as tf


class Actor(tf.keras.Model):

    def __init__(self,
                 hidden_sizes,
                 n_actions,
                 hidden_activation='tanh',
                 output_activation='softmax',
                 learning_rate='3e-4'):

        super(Actor, self).__init__()

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

        self.dense_layers = \
            [tf.keras.layers.Dense(hidden_sizes[i], activation=hidden_activation) for i in range(n_actions)]

        self.output_layer = tf.keras.layers.Dense(n_actions, activation=output_activation)

    def call(self, inputs):

        output = inputs

        for dense in self.dense_layers:
            output = dense(output)

        output = self.output_layer(output)

        return output

    @tf.function
    def train_step(self):
        pass






