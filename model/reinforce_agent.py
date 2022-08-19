import tensorflow as tf
from model.agent import Agent


class ReinforceAgent(Agent):

    def __init__(self, with_baseline, critic_loss_coefficient=0.5, **base_agent_config):
        """

        Parameters
        ----------
        with_baseline: bool
            If True, then also the baseline is considered during the training
        critic_loss_coefficient: float
            Coefficient used for scaling the critic loss (called c1 in the original PPO paper)
        base_agent_config: dict
            Dictionary containing all the configuration required for initializing an instance of Agent class
        """

        super(ReinforceAgent, self).__init__(**base_agent_config)

        self.with_baseline = with_baseline
        self.critic_loss_coefficient = critic_loss_coefficient

        if with_baseline is False:
            self.critic.trainable = False

    @tf.function
    def train_step(self, states, returns, actions):

        with tf.GradientTape() as tape:
            actor_logits, values = self(states)

            # Compute the policy (actor) loss
            actions = tf.expand_dims(actions, axis=-1)
            probabilities = self.compute_probabilities(actor_logits, actions)
            log_action_probabilities = tf.math.log(probabilities)
            policy_loss = -tf.reduce_mean((returns - values) * log_action_probabilities)

            # Compute the value (critic) loss
            critic_loss = 0
            if self.with_baseline:
                critic_loss = self.critic_loss_coefficient * tf.reduce_mean(tf.square(values - returns))

            # Total loss
            loss = policy_loss + critic_loss

        # Compute the gradients
        policy_grads = tape.gradient(loss, self.trainable_variables)
        # Gradients clipping
        grads_clipped, _ = tf.clip_by_global_norm(policy_grads, 0.5)
        # Update model weights
        self.optimizer.apply_gradients(zip(grads_clipped, self.trainable_variables))

        return loss
