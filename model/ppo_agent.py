import tensorflow as tf
import numpy as np

from model.agent import Agent


class PPOAgent(Agent):

    def __init__(self,
                 n_actions,
                 backbone_type,
                 clip_ratio=0.2,
                 clip_value_estimates=False):
        super(PPOAgent, self).__init__(n_actions=n_actions, backbone_type=backbone_type)

        self.clip_ratio = clip_ratio
        self.clip_value_estimates = clip_value_estimates

    def train_step(self, states, actions, action_probabilities, advantages, returns, old_values):
        with tf.GradientTape() as tape:
            actor_logits, values = self(states)

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

            if self.clip_value_estimates:
                values_clipped = old_values + tf.clip_by_value(values - old_values, - self.clip_ratio, self.clip_ratio)
                critic_loss_clipped = tf.maximum(tf.square(values - returns), tf.square(values_clipped - returns))
                critic_loss = 0.5 * tf.reduce_mean(critic_loss_clipped)
            else:
                critic_loss = 0.5 * tf.reduce_mean((returns - values) ** 2)

            entropy_loss = self.compute_entropy(actor_logits) * 0.01

            loss = actor_loss + critic_loss + entropy_loss

        policy_grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(policy_grads, self.trainable_variables))

        return loss

    @staticmethod
    def compute_entropy(distribution):
        value, counts = np.unique(distribution, return_counts=True)
        norm_counts = counts / counts.sum()
        return -(norm_counts * np.log(norm_counts)).sum()
