import tensorflow as tf
import numpy as np

from agent.agent import Agent


class PPOAgent(Agent):

    def __init__(self, n_actions, clip_ratio=0.2):
        super(PPOAgent, self).__init__(n_actions=n_actions)

        self.clip_ratio = clip_ratio

    def train_step(self, states, actions, action_probabilities, advantages, returns):
        with tf.GradientTape() as tape:
            actor_logits, value = self(states)

            actions = tf.expand_dims(actions, axis=-1)
            ratio = self._compute_probabilities(actor_logits, actions) / action_probabilities

            min_advantage = tf.where(
                advantages > 0,
                advantages * (1 + self.clip_ratio),
                advantages * (1 - self.clip_ratio),
            )

            actor_loss = -tf.reduce_mean(
                tf.minimum(ratio * advantages, min_advantage)
            )

            critic_loss = 0.5 * tf.reduce_mean((returns - value) ** 2)

            entropy_loss = self.compute_entropy(actor_logits) * 0.01

            loss = actor_loss + critic_loss + entropy_loss

        policy_grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(policy_grads, self.trainable_variables))

    @staticmethod
    def _compute_probabilities(logits, actions):
        action_distribution = tf.nn.softmax(logits)
        probabilities = tf.gather_nd(action_distribution, actions, batch_dims=actions.shape[1])

        return probabilities

    @staticmethod
    def compute_entropy(distribution):
        value, counts = np.unique(distribution, return_counts=True)
        norm_counts = counts / counts.sum()
        return -(norm_counts * np.log(norm_counts)).sum()
