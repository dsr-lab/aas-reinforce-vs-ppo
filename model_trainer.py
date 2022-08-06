import os
import config
import logging
import numpy as np
import tensorflow as tf

from datetime import datetime
from agent.ppo_agent import PPOAgent
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
            self.model = PPOAgent(n_actions=self.environment.n_actions,
                                  backbone_type=config.BACKBONE_TYPE)
        else:
            self.model = ReinforceAgent(n_actions=self.environment.n_actions,
                                        backbone_type=config.BACKBONE_TYPE)

        self.model(model_input)

        # Set model checkpoints
        self.model_checkpoint = tf.train.Checkpoint(optimizer=self.model.optimizer, model=self.model)

        # Possibly load model weights
        last_checkpoint = tf.train.latest_checkpoint(config.WEIGHTS_PATH)
        if last_checkpoint is not None:
            checkpoint_restored = self.model_checkpoint.restore(tf.train.latest_checkpoint(config.WEIGHTS_PATH))
            checkpoint_restored.assert_consumed()

        # Init variables required for processing the mini-batches
        n_actors = self.environment.environment.num
        total_time_steps = n_actors * config.AGENT_HORIZON
        self.n_batches = total_time_steps // config.BATCH_SIZE
        self.time_step_indices = np.arange(total_time_steps)

        # Init the buffer
        self.trajectory_buffer = TrajectoryBuffer(max_agent_steps=config.AGENT_HORIZON,
                                                  max_game_steps=self.environment.get_max_game_steps(),
                                                  n_agents=n_actors,
                                                  obervation_shape=(64, 64, 3))

        if config.SAVE_LOGS:
            logging.basicConfig(filename=f'logs/log_{datetime.now().strftime("%Y%m%d%H%M%S")}.txt',
                                filemode='a',
                                format='%(message)s',
                                level=logging.INFO)

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
                self.trajectory_buffer.add_element(state, reward, first, action, action_probability, value, new_value,
                                                   time_step)

                # Update current variables
                state = new_state
                first = new_first

            # Get the full trajectory from the set of all gathered episodes
            trajectory = self.trajectory_buffer.get_trajectory()

            states = trajectory.states
            actions = trajectory.actions
            action_probabilities = trajectory.action_probabilities
            returns = trajectory.returns
            advantages = trajectory.advantages

            iteration_loss = 0

            for epoch in range(config.EPOCHS_MODEL_UPDATE):

                if config.RANDOMIZE_SAMPLES:
                    np.random.shuffle(self.time_step_indices)

                for start in range(0, self.n_batches, config.BATCH_SIZE):
                    end = start + config.BATCH_SIZE
                    interval = self.time_step_indices[start:end]

                    dict_args = {
                        "states": states[interval],
                        "actions": actions[interval],
                        "action_probabilities": action_probabilities[interval],
                        "advantages": advantages[interval],
                        "returns": returns[interval]
                    }
                    iteration_loss += self.model.train_step(**dict_args)

            iteration_loss /= (self.n_batches * config.EPOCHS_MODEL_UPDATE)

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
                        "actions": actions.tolist(),
                        "advantages": advantages.tolist(),
                        "returns": returns.tolist(),
                        "loss": iteration_loss.numpy(),
                        "win_ratio": win_ratio,
                        "n_win": trajectory.n_wins,
                        "n_loss": trajectory.n_loss,
                        "n_incomplete": trajectory.n_incomplete
                    }
                }
                logging.info(f'{dict_log}')

            if config.SAVE_WEIGHTS and (iteration % 10 == 0 or iteration == config.ITERATIONS - 1):
                self.model_checkpoint.save(file_prefix=os.path.join(config.WEIGHTS_PATH, "ckpt"))
