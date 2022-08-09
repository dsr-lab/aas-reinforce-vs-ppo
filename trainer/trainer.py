import os
import config
import logging
import numpy as np
import tensorflow as tf

from datetime import datetime
from model.agent import Agent
from abc import abstractmethod
from environment.env_wrapper import EnvironmentWrapper
from environment.flattened_trajectory import FlattenedTrajectory
from environment.trajectory_buffer import TrajectoryBuffer


class Trainer:

    def __init__(self,
                 environment: EnvironmentWrapper):
        self.environment = environment
        self.trajectory_buffer = self.init_trajectory_buffer()

        # Init the model
        model_input = tf.keras.Input(shape=self.environment.get_state_shape())
        self.model = self.init_agent(config.BACKBONE_TYPE)
        self.model(model_input)

        # Set model checkpoints
        self.model_checkpoint = tf.train.Checkpoint(optimizer=self.model.optimizer, model=self.model)

        # Possibly load model weights
        last_checkpoint = tf.train.latest_checkpoint(config.WEIGHTS_PATH)
        if last_checkpoint is not None:
            checkpoint_restored = self.model_checkpoint.restore(last_checkpoint)
            checkpoint_restored.assert_consumed()

        # Init logger
        if config.SAVE_LOGS:
            logging.basicConfig(filename=f'logs/log_{datetime.now().strftime("%Y%m%d%H%M%S")}.txt',
                                filemode='a',
                                format='%(message)s',
                                level=logging.INFO)

    @abstractmethod
    def init_agent(self, backbone_type) -> Agent:
        pass

    @abstractmethod
    def init_trajectory_buffer(self) -> TrajectoryBuffer:
        pass

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def can_save_weights(self, **kwargs):
        pass

    @staticmethod
    def normalize(x):
        return (x - np.mean(x)) / (np.std(x) + 1e-8)

    def compute_post_iteration_operations(self, iteration, iteration_loss, trajectory: FlattenedTrajectory):

        self._log_iteration_results(iteration, iteration_loss, trajectory)

        self._save_model_weights(iteration)

    @staticmethod
    def _log_iteration_results(iteration, iteration_loss, trajectory):
        win_ratio = trajectory.n_wins / (trajectory.n_loss + trajectory.n_incomplete + trajectory.n_wins)
        print(
            f'Epoch: {iteration + 1} -  '
            f'Win: {trajectory.n_wins} - '
            f'Loss: {trajectory.n_loss} - '
            f'Incomplete: {trajectory.n_incomplete} - '
            f'Ratio: - winRatio: {win_ratio}'
        )

        if config.SAVE_LOGS:
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

    def _save_model_weights(self, iteration):
        if self.can_save_weights(iteration):
            self.model_checkpoint.save(file_prefix=os.path.join(config.WEIGHTS_PATH, "ckpt"))
