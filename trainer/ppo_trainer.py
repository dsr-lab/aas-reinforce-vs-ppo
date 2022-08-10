import numpy as np

from model.agent import Agent
from model.ppo_agent import PPOAgent
from environment.env_wrapper import EnvironmentWrapper
from environment.trajectory_buffer import TrajectoryBuffer
from trainer.trainer import Trainer


class PPOTrainer(Trainer):

    def __init__(self,
                 environment: EnvironmentWrapper,
                 n_agents=32,
                 n_iterations=256,
                 agent_horizon=1024,
                 batch_size=256,
                 epochs_model_update=3,
                 randomize_samples=True,
                 clip_ratio=0.2,
                 clip_value_estimates=False,
                 normalize_advantages=False,
                 **trainer_args):

        self.n_agents = n_agents
        self.clip_ratio = clip_ratio
        self.agent_horizon = agent_horizon
        self.epochs_model_update = epochs_model_update
        self.randomize_samples = randomize_samples
        self.batch_size = batch_size
        self.n_iterations = n_iterations
        self.normalize_advantages = normalize_advantages
        self.clip_value_estimates = clip_value_estimates

        super(PPOTrainer, self).__init__(environment=environment, **trainer_args)

        # Init variables required for processing the mini-batches
        total_time_steps = n_agents * agent_horizon
        self.n_batches = total_time_steps // batch_size
        self.time_step_indices = np.arange(total_time_steps)

    def init_agent(self, backbone_type) -> Agent:
        return PPOAgent(n_actions=self.environment.n_actions,
                        backbone_type=backbone_type,
                        clip_ratio=self.clip_ratio,
                        clip_value_estimates=self.clip_value_estimates)

    def init_trajectory_buffer(self, set_negative_rewards_for_losses) -> TrajectoryBuffer:
        # Init the buffer
        return TrajectoryBuffer(max_agent_steps=self.agent_horizon,
                                max_game_steps=self.environment.get_max_game_steps(),
                                n_agents=self.n_agents,
                                obervation_shape=(64, 64, 3),  # TODO: parametrize
                                set_negative_rewards_for_losses=set_negative_rewards_for_losses)

    def train(self):

        for iteration in range(self.n_iterations):

            trajectory = self._create_trajectory()

            iteration_loss = self._update_model_weights(trajectory)

            self.log_iteration_results(iteration, iteration_loss, trajectory)

            if iteration % 10 == 0 or iteration == self.n_iterations - 1:
                self.save_model_weights()

    def _create_trajectory(self):
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
        return self.trajectory_buffer.get_trajectory(flattened=True)

    def _update_model_weights(self, trajectory):
        iteration_loss = 0
        for epoch in range(self.epochs_model_update):

            if self.randomize_samples:
                np.random.shuffle(self.time_step_indices)

            for start in range(0, self.n_batches, self.batch_size):
                end = start + self.batch_size
                interval = self.time_step_indices[start:end]

                dict_args = {
                    "states": trajectory.states[interval],
                    "actions": trajectory.actions[interval],
                    "action_probabilities":
                        self.normalize(trajectory.advantages[interval]) if self.normalize_advantages
                        else trajectory.action_probabilities[interval],
                    "advantages": trajectory.advantages[interval],
                    "returns": trajectory.returns[interval],
                    "old_values": trajectory.values[interval]
                }

                iteration_loss += self.model.train_step(**dict_args)

        iteration_loss /= (self.n_batches * self.epochs_model_update)

        return iteration_loss
