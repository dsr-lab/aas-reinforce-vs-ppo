import config  # TODO: REMOVE
import tensorflow as tf

from buffer import Buffer
from model.actor import Actor
from model.critic import Critic
from model.trajectory import Trajectory

import numpy as np


class PPO:

    def __init__(self, environment):

        model_input = tf.keras.Input(shape=config.OBSERVATON_SHAPE)

        self.critic = Critic(hidden_sizes=config.CRITIC_HIDDEN_SIZES)
        self.critic(model_input)

        self.actor = Actor(hidden_sizes=config.ACTOR_HIDDEN_SIZES, n_actions=config.N_ACTIONS)
        self.actor(model_input)

        self.trajectory = Trajectory()
        # self.buffer = Buffer(config.OBSERVATON_SHAPE[0], config.STEPS_PER_EPOCH)

        self.critic.summary()
        self.actor.summary()

        self.environment = environment

    def train(self):

        state = self._environment_reset()
        # observation, episode_return, episode_length = self.environment.reset(), 0, 0

        episode_return = 0
        episode_length = 0


        N_ACTORS = 8
        HORIZON = 512
        ITERATIONS = 30

        for iteration in range(ITERATIONS):

            # Initialize before every iteration
            self.trajectory.new_trajectory()

            sum_return = 0
            sum_length = 0
            num_episodes = 0

            for actor in range(N_ACTORS):

                for horizon_t in range(HORIZON):

                    self.environment.render()

                    action, action_probability = self.actor.get_action(state)

                    state_new, reward, done = self._environment_step(action[0].numpy())
                    episode_return += reward
                    episode_length += 1

                    # Get the value and log-probability of the action
                    value = self.critic(state)
                    # logprobability_t = logprobabilities(logits, action)

                    # logprobability_t = actor._compute_probabilities(logits, action)

                    # Store obs, act, rew, v_t, logp_pi_t
                    # buffer.store(observation, action, reward, value_t, logprobability_t)

                    # self.buffer.store(state, action, reward, value, action_probability)
                    self.trajectory.add_episode_observation(state, action, reward, value, action_probability)

                    # Update the observation
                    state = state_new

                    # Finish trajectory if reached to a terminal state
                    terminal = done
                    if terminal or (horizon_t == HORIZON - 1):
                        last_value = 0 if done else self.critic(state)
                        # self.buffer.finish_trajectory(last_value)
                        self.trajectory.complete_episode(last_value)
                        sum_return += episode_return
                        sum_length += episode_length
                        num_episodes += 1
                        state, episode_return, episode_length = self._environment_reset(), 0, 0

            states, actions, action_probabilities, returns, advantages = self.trajectory.get_trajectory()
            advantages_mean, advantages_std = (
               np.mean(advantages),
               np.std(advantages),
            )
            advantages = (advantages - advantages_mean) / advantages_std

            # states2, actions2, advantages2, returns2, action_probabilities2 = self.buffer.get()

            for i in range(config.TRAIN_POLICY_ITERATIONS):
                for i, batch_size in enumerate(range(0, 4096, 256)):
                    print(batch_size*i, batch_size)
                self.actor.train_step(states, actions, action_probabilities, advantages)

            for i in range(config.TRAIN_STATE_VALUE_ITERATIONS):
                self.critic.train_step(states, returns)

            # TODO: print metrics
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

