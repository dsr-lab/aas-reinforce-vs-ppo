import tensorflow as tf
import numpy as np


from model.actor import Actor
from model.critic import Critic


class FullModel(tf.keras.Model):
    def __init__(self):
        super(FullModel, self).__init__()

        hidden_sizes = [64, 64]

        self.feature_extractor = \
            [tf.keras.layers.Dense(hidden_sizes[i], activation='tanh') for i in range(len(hidden_sizes))]

        self.critic = Critic(hidden_sizes=[64, 64])
        self.actor = Actor(hidden_sizes=[64, 64], n_actions=2)
        self.clip_ratio = 0.2

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=3e-4)

    def call(self, states):

        output = states

        for layer in self.feature_extractor:
            output = layer(output)

        actor_logits = self.actor(output)
        value = self.critic(output)

        return actor_logits, value

    def get_value(self, state):

        output = state

        for layer in self.feature_extractor:
            output = layer(output)

        output = self.critic(output)
        return output

    def get_action(self, state):

        output = state

        for layer in self.feature_extractor:
            output = layer(output)

        logits = self.actor(output)
        action = tf.squeeze(tf.random.categorical(logits, 1), axis=1)

        action_probability, action_distribution = self._compute_probabilities(logits, [action])

        action_probability = action_probability[0]

        return action, action_probability, action_distribution

    @staticmethod
    def entropy_loss(policy_logits, ent_discount_val):
        probs = tf.nn.softmax(policy_logits)
        entropy_loss = -tf.reduce_mean(tf.keras.losses.categorical_crossentropy(probs, probs))
        return entropy_loss * ent_discount_val

    def train_step(self, states, actions, action_probabilities, advantages, returns, action_distribution):
        with tf.GradientTape() as tape:
            actor_logits, value = self(states)

            ratio = self._compute_probabilities(actor_logits, actions)[0] / action_probabilities

            min_advantage = tf.where(
                advantages > 0,
                advantages * (1 + self.clip_ratio),
                advantages * (1 - self.clip_ratio),
            )

            actor_loss = -tf.reduce_mean(
                tf.minimum(ratio * advantages, min_advantage)
            )

            critic_loss = 0.5 * tf.reduce_mean((returns - value) ** 2)

            # TODO: add entropy?
            entropy_loss = self.entropy_loss(actor_logits, 0.01)

            loss = actor_loss + critic_loss + entropy_loss

        policy_grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(policy_grads, self.trainable_variables))

        actor_logits, _ = self(states)
        kl = tf.reduce_mean(
            action_probabilities / self._compute_probabilities(actor_logits, actions)[0]
        )
        kl = tf.reduce_sum(kl)
        return kl

    @staticmethod
    def _compute_probabilities(logits, actions):
        actions = tf.expand_dims(actions, axis=-1)
        action_distribution = tf.nn.softmax(logits)
        probabilities = tf.gather_nd(action_distribution, actions, batch_dims=1)

        return probabilities, action_distribution
