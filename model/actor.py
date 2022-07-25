import tensorflow as tf
import numpy as np

class Actor(tf.keras.Model):

    def __init__(self,
                 hidden_sizes,
                 n_actions,
                 hidden_activation='tanh',
                 output_activation='softmax',
                 learning_rate=3e-4,
                 clip_ratio=0.2):

        super(Actor, self).__init__()

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        self.clip_ratio = clip_ratio

        self.dense_layers = \
            [tf.keras.layers.Dense(hidden_sizes[i], activation=hidden_activation) for i in range(n_actions)]

        self.output_layer = tf.keras.layers.Dense(n_actions, activation=output_activation)

    def call(self, inputs):

        output = inputs

        for dense in self.dense_layers:
            output = dense(output)

        output = self.output_layer(output)

        return output

    # @tf.function
    def train_step(self, states, actions, action_probabilities, returns, advantages):
        with tf.GradientTape() as tape:
            policy_logits = self(states)
            new_action_probabilities = self._compute_probabilities(policy_logits, actions)

            ratio = action_probabilities / new_action_probabilities

            lower_bound_advantage = np.zeros(advantages.shape)
            lower_bound_advantage[advantages > 0] = advantages[advantages > 0] * (1 + self.clip_ratio)
            lower_bound_advantage[advantages <= 0] = advantages[advantages <= 0] * (1 - self.clip_ratio)
            loss = -tf.reduce_mean(
                tf.minimum(ratio * advantages, lower_bound_advantage)
            )

        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

        # TODO: add KL Trick?

    def get_action(self, state):
        policy_logits = self(state)
        action = tf.random.categorical(policy_logits, 1).numpy().flatten()[0]  # TODO: optimize
        action_probability = self._compute_probabilities(policy_logits, [action])[0]

        return action, action_probability

    @staticmethod
    def _compute_probabilities(logits, actions):
        actions = tf.expand_dims(actions, axis=-1)
        probabilities = tf.gather_nd(tf.nn.softmax(logits), actions, batch_dims=1)

        return probabilities






