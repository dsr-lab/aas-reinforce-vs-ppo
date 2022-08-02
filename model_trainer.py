

import config  # TODO: REMOVE
import tensorflow as tf

from agent.ppo_agent import PPOAgent

import numpy as np

from agent.reinforce_agent import ReinforceAgent
from environment.env_wrapper import EnvironmentWrapper
from environment.trajectory_buffer import TrajectoryBuffer


class ModelTrainer:

    def __init__(self, environment: EnvironmentWrapper, agent_type='ppo'):

        self.environment = environment

        # Init the model
        model_input = tf.keras.Input(shape=self.environment.get_state_shape())

        if agent_type == 'ppo':
            self.model = PPOAgent(n_actions=self.environment.get_n_actions())
        else:
            self.model = ReinforceAgent(n_actions=self.environment.get_n_actions())

        self.model(model_input)

        # Init required for processing the mini-batches
        n_actors = self.environment.environment.num
        total_time_steps = n_actors * config.AGENT_HORIZON
        self.n_batches = total_time_steps // config.BATCH_SIZE
        self.time_step_indices = np.arange(total_time_steps)

        # Init the buffer
        self.trajectory_buffer = TrajectoryBuffer(max_agent_steps=config.AGENT_HORIZON,
                                                  n_agents=n_actors,
                                                  obervation_shape=(64, 64, 3))

    def train(self):

        for iteration in range(config.ITERATIONS):

            # Init trajectory
            self.trajectory_buffer.reset()
            # Reset the environment
            state, first = self.environment.reset()

            for time_step in range(config.AGENT_HORIZON):

                # Select an action
                action, action_probability = self.model.get_action(state)

                # Perform the action and get feedback from environment
                reward, new_state, new_first = self.environment.step(action.numpy())

                # Value of current state
                value = self.model.get_value(state)

                # Gather the next value that is afterward used for GAE computation
                new_value = 0
                if time_step == config.AGENT_HORIZON - 1:
                    new_value = self.model.get_value(new_state)

                # Add element to the current trajectory
                self.trajectory_buffer.add_element(state, reward, first, action, action_probability, value, new_value, time_step)

                # Update current variables
                state = new_state
                first = new_first

            # Get the full trajectory from the set of all gathered episodes
            trajectory = self.trajectory_buffer.get()

            states = trajectory.states
            actions = trajectory.actions
            action_probabilities = trajectory.action_probabilities
            returns = trajectory.returns
            advantages = trajectory.advantages

            for _ in range(config.EPOCHS_MODEL_UPDATE):

                if config.RANDOMIZE_SAMPLES:
                    np.random.shuffle(self.time_step_indices)

                for start in range(0, self.n_batches, config.BATCH_SIZE):
                    end = start + config.BATCH_SIZE
                    mbinds = self.time_step_indices[start:end]

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

                    dict_args = {
                        "states": states[mbinds],
                        "actions": actions[mbinds],
                        "action_probabilities": action_probabilities[mbinds],
                        "advantages": batch_advantages,
                        "returns": batch_returns
                    }
                    self.model.train_step(**dict_args)

            # TODO: save weights

            assert(trajectory.n_episodes == trajectory.n_loss + trajectory.n_incomplete + trajectory.n_wins)
            win_ratio = trajectory.n_wins / trajectory.n_episodes

            print(
                f'Epoch: {iteration + 1} -  Win: {trajectory.n_wins} - Loss: {trajectory.n_loss} - incomplete: {trajectory.n_incomplete} - ratio: - winRatio: {win_ratio}'
            )

        self.model.save('weights/')

