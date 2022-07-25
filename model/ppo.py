import config  # TODO: REMOVE
import tensorflow as tf

from model.actor import Actor
from model.critic import Critic
from model.trajectory import Trajectory


class PPO:

    def __init__(self, environment):

        model_input = tf.keras.Input(shape=config.OBSERVATON_SHAPE)

        self.critic = Critic(hidden_sizes=config.CRITIC_HIDDEN_SIZES)
        self.critic(model_input)

        self.actor = Actor(hidden_sizes=config.ACTOR_HIDDEN_SIZES, n_actions=config.N_ACTIONS)
        self.actor(model_input)

        self.trajectory = Trajectory()

        # self.critic.summary()
        # self.actor.summary()

        self.environment = environment

    def train(self):
        state = self._environment_reset()
        episode_return = 0
        episode_length = 0

        for epoch in range(config.EPOCHS):

            # Initialize a new trajectory in each epoch
            self.trajectory.new_trajectory()

            for t in range(config.STEPS_PER_EPOCH):

                self.environment.render()

                value = self.critic(state)
                action, action_probability = self.actor.get_action(state)

                state, reward, episode_complete = self._environment_step(action)

                self.trajectory.add_episode_observation(state, action, reward, value, action_probability)

                if episode_complete:
                    self.trajectory.complete_episode(0)
                    state = self._environment_reset()

                elif t == config.STEPS_PER_EPOCH - 1:
                    value = self.critic(state)
                    self.trajectory.complete_episode(value)
                    state = self._environment_reset()

            states, actions, action_probabilities, returns, advantages = self.trajectory.get_trajectory()

            # TODO: train ACTOR and CRITIC
            for i in range(config.TRAIN_POLICY_ITERATIONS):
                self.actor.train_step(states, actions, action_probabilities, returns, advantages)

            for i in range(config.TRAIN_STATE_VALUE_ITERATIONS):
                self.critic.train_step(states, returns)

            # TODO: print metrics
            # TODO: save weights
            print(f'epoch {epoch} completed')

    def _environment_step(self, action):
        state, reward, episode_complete, _ = self.environment.step(action)
        state = tf.expand_dims(state, axis=0)

        return state, reward, episode_complete

    def _environment_reset(self):
        state = self.environment.reset()
        state = tf.expand_dims(state, axis=0)

        return state

