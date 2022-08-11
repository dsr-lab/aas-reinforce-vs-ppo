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

        # Init Adam parameters as tf.Variable so as to avoid warnings when restoring model weights
        # https://github.com/tensorflow/tensorflow/issues/33150#issuecomment-659968267
        self.optimizer = tf.keras.optimizers.Adam(
            learning_rate=tf.Variable(learning_rate),
            beta_1=tf.Variable(0.9),
            beta_2=tf.Variable(0.999),
            epsilon=tf.Variable(1e-7),
        )
        self.optimizer.iterations  # Force the creation of the optimizer.iter variable
        self.optimizer.decay = tf.Variable(0.0)

        # Backbone
        if backbone_type == 'impala':
            self.feature_extractor = ImpalaNet()
        else:
            self.feature_extractor = NatureNet()

        # TODO: comments on initialization!
        # https://arxiv.org/abs/2005.12729
        # Model heads
        self.critic = Head(n_outputs=1, kernel_initializer=tf.keras.initializers.Orthogonal(gain=1))
        self.actor = Head(n_outputs=n_actions, kernel_initializer=tf.keras.initializers.Orthogonal(gain=0.01))

    @abstractmethod
    def train_step(self, **kwargs):
        pass

    @tf.function
    def call(self, states):
        output = states

        output = self.feature_extractor(output)

        actor_logits = self.actor(output)

        value = self.critic(output)
        value = tf.squeeze(value, axis=-1)

        return actor_logits, value

    @tf.function
    def get_value(self, state):
       _, value = self(state)
       return value

    @tf.function
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
