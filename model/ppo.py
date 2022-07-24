import gym
import config  # TODO: REMOVE
import tensorflow as tf

from model.actor import Actor
from model.critic import Critic


class PPO:

    def __init__(self, environment):

        model_input = tf.keras.Input(shape=config.OBSERVATON_SHAPE)

        self.critic = Critic(hidden_sizes=config.CRITIC_HIDDEN_SIZES)
        self.critic(model_input)
        self.critic.summary()

        self.actor = Actor(hidden_sizes=config.ACTOR_HIDDEN_SIZES, n_actions=config.N_ACTIONS)
        self.actor(model_input)
        self.actor.summary()

        self.environment = environment

    def train(self):
        state, episode_return, episode_length = self.environment.reset(), 0, 0

        for epoch in range(config.EPOCHS):

            for t in range(config.STEPS_PER_EPOCH):

                self.environment.render()

                state = tf.expand_dims(state, axis=0)

                actor_logits = self.actor(state)
                action = tf.squeeze(tf.random.categorical(actor_logits, 1), axis=1)

                next_state, reward, done, _ = self.environment.step(action[0].numpy())
                state = next_state

                if done:
                    state = self.environment.reset()

            # TODO: train ACTOR and CRITIC
            for i in range(config.TRAIN_POLICY_ITERATIONS):
                self.actor.train_step()

            for i in range(config.TRAIN_STATE_VALUE_ITERATIONS):
                self.critic.train_step()

            # TODO: print metrics
            # TODO: save weights
