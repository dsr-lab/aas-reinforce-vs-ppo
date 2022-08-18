import os
import logging
import warnings

import numpy as np
import tensorflow as tf

from datetime import datetime
from abc import abstractmethod
from environment.env_wrapper import EnvironmentWrapper
from environment.trajectory_buffer import TrajectoryBuffer


class Trainer:
    """
    Convenience class responsible of:
     - creating the model
     - creating the trajectory
     - trainining the model
    """
    def __init__(self,
                 environment: EnvironmentWrapper,
                 n_agents=32,
                 n_iterations=256,
                 agent_horizon=1024,
                 batch_size=256,
                 epochs_model_update=3,
                 randomize_samples=True,
                 save_logs=True,
                 save_weights=True,
                 logs_path='logs',
                 weights_path='weights',
                 use_negative_rewards_for_losses=True
                 ):

        # Init trainer configuration variables
        self.n_iterations = n_iterations
        self.agent_horizon = agent_horizon
        self.batch_size = batch_size
        self.epochs_model_update = epochs_model_update
        self.randomize_samples = randomize_samples
        self.save_logs = save_logs
        self.save_weights = save_weights
        self.weights_path = weights_path
        self.environment = environment

        # Init variables required for processing the mini-batches
        total_time_steps = n_agents * self.agent_horizon
        self.n_batches = total_time_steps // batch_size
        self.time_step_indices = np.arange(total_time_steps)

        # Define the maximum number of steps that an agent can compute before considering an episode as incomplete
        max_game_steps = \
            environment.get_max_game_steps() if agent_horizon > environment.get_max_game_steps() else agent_horizon

        # Init the Replay Buffer
        self.trajectory_buffer = TrajectoryBuffer(max_agent_steps=agent_horizon,
                                                  max_game_steps=max_game_steps,
                                                  n_agents=n_agents,
                                                  obervation_shape=environment.get_state_shape(),
                                                  set_negative_rewards_for_losses=use_negative_rewards_for_losses)

        # Init the model
        model_input = tf.keras.Input(shape=self.environment.get_state_shape())
        if self.model is None:
            raise Exception("Model not initialized!")
        self.model(model_input)

        # Set model checkpoints
        self.model_checkpoint = tf.train.Checkpoint(optimizer=self.model.optimizer, model=self.model)

        # Possibly load model weights
        last_checkpoint = tf.train.latest_checkpoint(self.weights_path)
        if last_checkpoint is not None:
            checkpoint_restored = self.model_checkpoint.restore(last_checkpoint)
            checkpoint_restored.assert_consumed()
        else:
            warnings.warn('Warning: no weights found. You can safely ignore this warning if you are training '
                          'a new model.')

        # Init logger
        if self.save_logs:
            filename = os.path.join(logs_path, f'log_{datetime.now().strftime("%Y%m%d%H%M%S")}.txt')
            print(f'Saving logs in {filename}')

            for handler in logging.root.handlers[:]:
                logging.root.removeHandler(handler)

            logging.basicConfig(filename=filename,
                                filemode='a',
                                format='%(message)s',
                                level=logging.INFO)

    @abstractmethod
    def update_model_weights(self, trajectory):
        pass

    def train(self, training=True):
        """
        The actual train loop

        Parameters
        ----------
        training; bool
            If True, then the model weights will be updated.
            It should be set to false for model evaluation purposes.
        """
        for iteration in range(self.n_iterations):

            trajectory = self.create_trajectory()

            iteration_loss = 0
            if training:
                iteration_loss = self.update_model_weights(trajectory)

                if iteration % 10 == 0 or iteration == self.n_iterations - 1:
                    self.save_model_weights()

            self.log_iteration_results(iteration, iteration_loss, trajectory)

    def create_trajectory(self):
        # Init trajectory
        self.trajectory_buffer.reset()
        # Reset the environment
        state, first = self.environment.reset()

        for time_step in range(self.agent_horizon):

            # Select an action
            action, action_probability = self.model.get_action(state)

            # Perform the action and get feedback from environment
            reward, new_state, new_first = self.environment.step(action.numpy())

            # Value of current state
            value = self.model.get_value(state)

            # Gather the next value that is afterward used for GAE computation
            new_value = 0
            if time_step == self.agent_horizon - 1:
                new_value = self.model.get_value(new_state)

            # Add element to the current trajectory
            self.trajectory_buffer.add_element(state,
                                               reward,
                                               first,
                                               action,
                                               action_probability,
                                               value,
                                               new_value,
                                               time_step)

            # Update current variables
            state = new_state
            first = new_first

        # Get the full trajectory from the set of all gathered episodes
        return self.trajectory_buffer.get_trajectory()

    def evaluate(self):
        self.train(update_model=False)

    def log_iteration_results(self, iteration, iteration_loss, trajectory):
        win_ratio = trajectory.n_wins / (trajectory.n_loss + trajectory.n_incomplete + trajectory.n_wins)
        print(
            f'Epoch: {iteration + 1} -  '
            f'Win: {trajectory.n_wins} - '
            f'Loss: {trajectory.n_loss} - '
            f'Incomplete: {trajectory.n_incomplete} - '
            f'Ratio: - winRatio: {win_ratio}'
        )

        if self.save_logs:
            dict_log = {
                f'ITERATION_{iteration}': {
                    "actions": trajectory.actions.tolist(),
                    "advantages": trajectory.advantages.tolist(),
                    "returns": trajectory.returns.tolist(),
                    "loss": iteration_loss,
                    "win_ratio": win_ratio,
                    "n_win": trajectory.n_wins,
                    "n_loss": trajectory.n_loss,
                    "n_incomplete": trajectory.n_incomplete
                }
            }

            logging.info(f'{dict_log}')

    def save_model_weights(self):
        if self.save_weights:
            self.model_checkpoint.save(file_prefix=os.path.join(self.weights_path, "ckpt"))

    @staticmethod
    def normalize(x):
        return (x - np.mean(x)) / (np.std(x) + 1e-8)
