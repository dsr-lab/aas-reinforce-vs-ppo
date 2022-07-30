import config  # TODO: REMOVE
import tensorflow as tf

from model.actor import Actor
from model.critic import Critic
from model.full_model import FullModel
from model.trajectory import Trajectory

import numpy as np


class PPO:

    def __init__(self, environment):

        model_input = tf.keras.Input(shape=config.OBSERVATON_SHAPE)

        self.model = FullModel()
        self.model(model_input)

        self.trajectory = Trajectory()

        self.environment = environment

    def train(self):

        state = self._environment_reset()

        episode_return = 0
        episode_length = 0

        # TODO: move these in the configuration file
        N_ACTORS = 1
        # HORIZON = 4000
        ITERATIONS = 50

        AGENT_HORIZON = 4000
        TOTAL_TIME_STEPS = N_ACTORS * AGENT_HORIZON
        BATCH_SIZE = 1000
        N_BATCHES = TOTAL_TIME_STEPS // BATCH_SIZE
        MODEL_UPDATES_PER_ITERATION = 16
        INDS = np.arange(TOTAL_TIME_STEPS)
        RANDOMIZE=True

        agent_reward = np.zeros(N_ACTORS)
        agent_n_episodes = np.zeros(N_ACTORS)

        for iteration in range(ITERATIONS):

            # Initialize before every iteration
            self.trajectory.new_trajectory()

            sum_return = 0
            sum_length = 0
            num_episodes = 0

            for actor in range(N_ACTORS):

                for agent_t in range(AGENT_HORIZON):

                    if config.RENDER:
                        self.environment.render()

                    action, action_probability, action_distribution = self.model.get_action(state)

                    state_new, reward, done = self._environment_step(action[0].numpy())
                    episode_return += reward
                    episode_length += 1

                    value = self.model.get_value(state)

                    self.trajectory.add_episode_observation(state, action, reward, value, action_probability, action_distribution)

                    # Update the observation
                    state = state_new

                    # Finish trajectory if reached to a terminal state
                    terminal = done
                    if terminal or (agent_t == AGENT_HORIZON - 1):
                        # last_value = 0 if done else self.critic(state)
                        last_value = 0 if done else self.model.get_value(state)

                        self.trajectory.complete_episode(last_value)
                        sum_return += episode_return
                        sum_length += episode_length
                        num_episodes += 1
                        state, episode_return, episode_length = self._environment_reset(), 0, 0

            states, actions, action_probabilities, returns, advantages, action_distributions = self.trajectory.get_trajectory()

            for _ in range(40):

                if RANDOMIZE:
                    np.random.shuffle(INDS)

                for start in range(0, N_BATCHES, BATCH_SIZE):
                    end = start + BATCH_SIZE
                    mbinds = INDS[start:end]

                    # batch_advantages_mean, batch_advantages_std = (
                    #     np.mean(advantages[mbinds]),
                    #     np.std(advantages[mbinds]),
                    # )
                    # batch_advantages = (advantages[mbinds] - batch_advantages_mean) / (batch_advantages_std+1e-8)
                    batch_advantages = advantages[mbinds]

                    # batch_returns_mean, batch_returns_std = (
                    #     np.mean(returns[mbinds]),
                    #     np.std(returns[mbinds]),
                    # )
                    # batch_returns = (returns[mbinds] - batch_returns_mean) / (batch_returns_std+1e-8)
                    batch_returns = returns[mbinds]


                    # train_value_function(observation_buffer, return_buffer)
                    kl = self.model.train_step(states[mbinds], actions[mbinds], action_probabilities[mbinds],
                                          batch_advantages, batch_returns, action_distributions[mbinds])

                    #if kl > 1.5 * 0.01:
                        # Early Stopping
                    #    break

            # for i in range(80):
            #    self.model.train_step(states, actions, action_probabilities, advantages, returns, action_distributions)

            # TODO: divide in batches?
            '''
            for i in range(config.TRAIN_POLICY_ITERATIONS):
                for i, batch_size in enumerate(range(0, 4096, 256)):
                    print(batch_size*i, batch_size)
                self.actor.train_step(states, actions, action_probabilities, advantages)

            for i in range(config.TRAIN_STATE_VALUE_ITERATIONS):
                self.critic.train_step(states, returns)
            '''

            # TODO: save weights
            print(
                f" Epoch: {iteration + 1}. Mean Return: {sum_return / num_episodes}. Mean Length: {sum_length / num_episodes}. N. Episodes: {num_episodes}"
            )

    def _environment_step(self, action):
        state, reward, episode_complete, _ = self.environment.step(action)
        state = tf.expand_dims(state, axis=0)

        return state, reward, episode_complete

    def _environment_reset(self):
        state = self.environment.reset()
        state = tf.expand_dims(state, axis=0)

        return state

