import os

import config
import tensorflow as tf

# import absl.logging
# absl.logging.set_verbosity(absl.logging.ERROR)

from agent.ppo_agent import PPOAgent

import numpy as np

from agent.reinforce_agent import ReinforceAgent
from environment.env_wrapper import EnvironmentWrapper
from environment.trajectory_buffer import TrajectoryBuffer


class ModelTrainer:

    def __init__(self,
                 environment: EnvironmentWrapper):

        self.environment = environment

        # Init the model
        model_input = tf.keras.Input(shape=self.environment.get_state_shape())

        if config.AGENT_TYPE == 'ppo':
            self.model = PPOAgent(n_actions=self.environment.get_n_actions(),
                                  backbone_type=config.BACKBONE_TYPE)
        else:
            self.model = ReinforceAgent(n_actions=self.environment.get_n_actions(),
                                        backbone_type=config.BACKBONE_TYPE)

        # self.model = tf.keras.models.load_model(config.WEIGHTS_PATH)

        self.model(model_input)
        self.model_checkpoint = tf.train.Checkpoint(optimizer=self.model.optimizer, model=self.model)

        last_checkpoint = tf.train.latest_checkpoint(config.WEIGHTS_PATH)
        if last_checkpoint != None:
            a = self.model_checkpoint.restore(tf.train.latest_checkpoint(config.WEIGHTS_PATH))
            a.assert_consumed()

        #self.model.load_weights(config.WEIGHTS_PATH)

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

            for epoch in range(config.EPOCHS_MODEL_UPDATE):

                kl_loss = 0

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
                    # kl_loss += self.model.train_step(**dict_args)
                    self.model.train_step(**dict_args)

                # kl_loss /= self.n_batches
                # if kl_loss > 1.5 * 0.01:
                #     print(f'Early stopping after {epoch} epochs during iteration number {iteration}')
                #     break

            assert(trajectory.n_episodes == trajectory.n_loss + trajectory.n_incomplete + trajectory.n_wins)
            win_ratio = trajectory.n_wins / trajectory.n_episodes

            print(
                f'Epoch: {iteration + 1} -  Win: {trajectory.n_wins} - Loss: {trajectory.n_loss} - incomplete: {trajectory.n_incomplete} - ratio: - winRatio: {win_ratio}'
            )

            if config.SAVE_WEIGHTS and iteration % 10 == 0:
                self.model_checkpoint.save(file_prefix=os.path.join(config.WEIGHTS_PATH, "ckpt"))
