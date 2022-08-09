import config
import numpy as np

from model.agent import Agent
from model.ppo_agent import PPOAgent
from environment.env_wrapper import EnvironmentWrapper
from environment.trajectory_buffer import TrajectoryBuffer
from trainer.trainer import Trainer


class PPOTrainer(Trainer):

    def __init__(self, environment: EnvironmentWrapper):
        super(PPOTrainer, self).__init__(environment=environment)

        # Init variables required for processing the mini-batches
        total_time_steps = config.PPO_N_AGENTS * config.PPO_AGENT_HORIZON
        self.n_batches = total_time_steps // config.PPO_BATCH_SIZE
        self.time_step_indices = np.arange(total_time_steps)

    def init_agent(self, backbone_type) -> Agent:
        return PPOAgent(n_actions=self.environment.n_actions,
                        backbone_type=backbone_type)

    def init_trajectory_buffer(self, set_negative_rewards_for_losses) -> TrajectoryBuffer:
        # Init the buffer
        return TrajectoryBuffer(max_agent_steps=config.PPO_AGENT_HORIZON,
                                max_game_steps=self.environment.get_max_game_steps(),
                                n_agents=config.PPO_N_AGENTS,
                                obervation_shape=(64, 64, 3),
                                set_negative_rewards_for_losses=set_negative_rewards_for_losses)

    def can_save_weights(self, iteration):
        return config.SAVE_WEIGHTS and (iteration % 10 == 0 or iteration == config.PPO_ITERATIONS - 1)

    def train(self):

        for iteration in range(config.PPO_ITERATIONS):

            trajectory = self._create_trajectory()

            iteration_loss = self._update_model_weights(trajectory)

            self.compute_post_iteration_operations(iteration, iteration_loss, trajectory)

    def _create_trajectory(self):
        # Init trajectory
        self.trajectory_buffer.reset()
        # Reset the environment
        state, first = self.environment.reset()
        for time_step in range(config.PPO_AGENT_HORIZON):

            # Select an action
            action, action_probability = self.model.get_action(state)

            # Perform the action and get feedback from environment
            reward, new_state, new_first = self.environment.step(action.numpy())

            # Value of current state
            value = self.model.get_value(state)

            # Gather the next value that is afterward used for GAE computation
            new_value = 0
            if time_step == config.PPO_AGENT_HORIZON - 1:
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
        return self.trajectory_buffer.get_trajectory(flattened=True)

    def _update_model_weights(self, trajectory):
        iteration_loss = 0
        for epoch in range(config.PPO_EPOCHS_MODEL_UPDATE):

            if config.PPO_RANDOMIZE_SAMPLES:
                np.random.shuffle(self.time_step_indices)

            for start in range(0, self.n_batches, config.PPO_BATCH_SIZE):
                end = start + config.PPO_BATCH_SIZE
                interval = self.time_step_indices[start:end]

                dict_args = {
                    "states": trajectory.states[interval],
                    "actions": trajectory.actions[interval],
                    "action_probabilities": trajectory.action_probabilities[interval],
                    "advantages": self.normalize(trajectory.advantages[interval]),
                    "returns": trajectory.returns[interval],
                    "old_values": trajectory.values[interval]
                }

                iteration_loss += self.model.train_step(**dict_args)

        iteration_loss /= (self.n_batches * config.PPO_EPOCHS_MODEL_UPDATE)

        return iteration_loss
