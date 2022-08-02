import tensorflow as tf
from agent.agent import Agent


class ReinforceAgent(Agent):

    def __init__(self, n_actions):
        super(ReinforceAgent, self).__init__(n_actions=n_actions)

    def train_step(self, states, returns):

        # 1) Get action_logits
        # 2) get proability distribution of actions with the softmax
        # 3) compute the logarithm of the previous probability
        # 4) multiply with the returns

        with tf.GradientTape() as tape:
            actor_logits, _ = self(states)
            action_distribution = tf.nn.softmax(actor_logits)
            log_action_distribution = tf.log(action_distribution)
            loss = -tf.reduce_mean(returns * log_action_distribution)

        policy_grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(policy_grads, self.trainable_variables))



