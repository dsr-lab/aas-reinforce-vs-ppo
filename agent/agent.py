import tensorflow as tf
from abc import abstractmethod

from model.head import Head
from model.impala_cnn import ImpalaNet
from model.nature_cnn import NatureNet


class Agent(tf.keras.Model):

    def __init__(self,
                 n_actions,
                 learning_rate=5e-4,
                 backbone_type='impala'):

        super(Agent, self).__init__()

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

        # Backbone
        if backbone_type == 'impala':
            self.feature_extractor = ImpalaNet()
        else:
            self.feature_extractor = NatureNet()

        # Model heads
        self.critic = Head(n_outputs=1, kernel_initializer=tf.keras.initializers.Orthogonal(gain=1))
        self.actor = Head(n_outputs=n_actions, kernel_initializer=tf.keras.initializers.Orthogonal(gain=0.01))

    @abstractmethod
    def train_step(self, **kwargs):
        pass

    def call(self, states):
        output = states

        output = self.feature_extractor(output)

        actor_logits = self.actor(output)

        value = self.critic(output)
        value = tf.squeeze(value, axis=-1)

        return actor_logits, value

    def get_value(self, state):
       _, value = self(state)
       return value

    def get_action(self, state):
        action_logits, _ = self(state)

        # Sample an action
        action = tf.random.categorical(action_logits, 1)

        action_probability = self._compute_probabilities(action_logits, action)

        action = tf.squeeze(action, axis=-1)

        return action, action_probability

    @staticmethod
    def _compute_probabilities(logits, actions):
        action_distribution = tf.nn.softmax(logits)
        probabilities = tf.gather_nd(action_distribution, actions, batch_dims=actions.shape[1])

        return probabilities
