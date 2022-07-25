import tensorflow as tf


class Critic(tf.keras.Model):

    def __init__(self,
                 hidden_sizes,
                 hidden_activation='tanh',
                 output_activation=None,
                 learning_rate=1e-3):

        super(Critic, self).__init__()

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

        self.dense_layers = \
            [tf.keras.layers.Dense(hidden_sizes[i], activation=hidden_activation) for i in range(len(hidden_sizes))]

        self.output_layer = tf.keras.layers.Dense(1, activation=output_activation)

    def call(self, inputs):

        output = inputs

        for dense in self.dense_layers:
            output = dense(output)

        output = self.output_layer(output)

        return output

    # @tf.function
    def train_step(self, states, returns):
        with tf.GradientTape() as tape:
            loss = tf.reduce_mean((returns - self(states)) ** 2)

        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))



