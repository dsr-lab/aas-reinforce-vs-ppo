import tensorflow as tf
import numpy as np

from model.agent import Agent


class PPOAgent(Agent):

    def __init__(self,
                 clip_ratio=0.2,
                 clip_value_estimates=False,
                 entropy_bonus_coefficient=0.01,
                 critic_loss_coefficient=0.5,
                 **base_agent_config):
        """
        Class used for initializing the PPOAgent, which is a trainable Tensorflow model

        Parameters
        ----------
        clip_ratio: float
            The clip ratio used for limiting policy updates
        clip_value_estimates: bool
            If True, then the new value estimates will be limited considering also the old estimates
        entropy_bonus_coefficient: float
            Coefficient used for scaling the entropy bonus (called c2 in the original PPO paper)
        critic_loss_coefficient: float
            Coefficient used for scaling the critic loss (called c1 in the original PPO paper)
        base_agent_config: dict
            Dictionary containing all the configuration required for initializing an instance of Agent class
        """

        super(PPOAgent, self).__init__(**base_agent_config)

        self.clip_ratio = clip_ratio
        self.clip_value_estimates = clip_value_estimates

        self.entropy_bonus_coefficient = entropy_bonus_coefficient
        self.critic_loss_coefficient = critic_loss_coefficient

    @tf.function
    def train_step(self, states, actions, action_probabilities, advantages, returns, old_values):
        with tf.GradientTape() as tape:
            actor_logits, values = self(states)

            actions = tf.expand_dims(actions, axis=-1)
            ratio = self.compute_probabilities(actor_logits, actions) / action_probabilities

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
                critic_loss = self.critic_loss_coefficient * tf.reduce_mean(critic_loss_clipped)
            else:
                critic_loss = self.critic_loss_coefficient * tf.reduce_mean(tf.square(values - returns))

            entropy_loss = \
                tf.numpy_function(self.compute_entropy, [actor_logits], tf.float32) * self.entropy_bonus_coefficient

            loss = actor_loss + critic_loss + entropy_loss

        # Compute the gradients
        policy_grads = tape.gradient(loss, self.trainable_variables)
        # Gradients clipping
        grads_clipped, _ = tf.clip_by_global_norm(policy_grads, 0.5)
        # Update model weights
        self.optimizer.apply_gradients(zip(grads_clipped, self.trainable_variables))

        return loss

    @staticmethod
    def compute_entropy(distribution):
        value, counts = np.unique(distribution, return_counts=True)
        norm_counts = np.divide(counts, counts.sum(), dtype=np.float32)
        return -(norm_counts * np.log(norm_counts)).sum()
