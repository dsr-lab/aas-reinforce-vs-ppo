import os
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
                 environment: EnvironmentWrapper,
                 use_negative_rewards_for_losses=True,
                 backbone_type='impala',
                 weights_path='',
                 save_logs=True,
                 logs_path='',
                 save_weights=True,
                 ):
        self.environment = environment
        self.save_logs = save_logs
        self.save_weights = save_weights
        self.weights_path = weights_path
        self.trajectory_buffer = self.init_trajectory_buffer(use_negative_rewards_for_losses)

        # Init the model
        model_input = tf.keras.Input(shape=self.environment.get_state_shape())
        self.model = self.init_agent(backbone_type)
        self.model(model_input)

        # Set model checkpoints
        self.model_checkpoint = tf.train.Checkpoint(optimizer=self.model.optimizer, model=self.model)

        # Possibly load model weights
        last_checkpoint = tf.train.latest_checkpoint(self.weights_path)
        if last_checkpoint is not None:
            checkpoint_restored = self.model_checkpoint.restore(last_checkpoint)
            checkpoint_restored.assert_consumed()

        # Init logger
        if self.save_logs:
            logging.basicConfig(filename=f'{logs_path}log_{datetime.now().strftime("%Y%m%d%H%M%S")}.txt',
                                filemode='a',
                                format='%(message)s',
                                level=logging.INFO)

    @abstractmethod
    def init_agent(self, backbone_type) -> Agent:
        pass

    @abstractmethod
    def init_trajectory_buffer(self, set_negative_rewards_for_losses) -> TrajectoryBuffer:
        pass

    @abstractmethod
    def train(self):
        pass

    @staticmethod
    def normalize(x):
        return (x - np.mean(x)) / (np.std(x) + 1e-8)

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
