from model.agent import Agent
from model.reinforce_agent import ReinforceAgent
from environment.env_wrapper import EnvironmentWrapper
from environment.trajectory_buffer import TrajectoryBuffer
from trainer.trainer import Trainer


class ReinforceTrainer(Trainer):

    def __init__(self,
                 environment: EnvironmentWrapper,
                 n_iterations,
                 n_checkpoints,
                 n_logs,
                 **trainer_args):
        super(ReinforceTrainer, self).__init__(environment=environment,
                                               **trainer_args)

        self.n_iterations = n_iterations
        self.n_checkpoints = n_checkpoints
        self.n_logs = n_logs

    def init_agent(self, backbone_type) -> Agent:
        return ReinforceAgent(n_actions=self.environment.n_actions,
                              backbone_type=backbone_type)

    def init_trajectory_buffer(self, set_negative_rewards_for_losses) -> TrajectoryBuffer:
        # Init the buffer
        return TrajectoryBuffer(max_agent_steps=self.environment.get_max_game_steps(),
                                max_game_steps=self.environment.get_max_game_steps(),
                                n_agents=1,
                                obervation_shape=self.environment.get_state_shape(),
                                set_negative_rewards_for_losses=set_negative_rewards_for_losses)

    def train(self, update_model=True):
        total_steps = 0
        n_episodes = 0

        # Variables for controlling when model's weights are saved
        n_checkpoint_saved = 1
        checkpoint_step = self.n_iterations / self.n_checkpoints

        # Variables for controlling when logs are showns and saved
        n_log_saved = 1
        log_step = self.n_iterations / self.n_logs

        # Run the trainer
        while total_steps < self.n_iterations:
            episode = self._create_episode()

            if update_model:
                iteration_loss = self._update_model_weights(episode)

                # Check whether weights can be saved
                if self._can_process_iteration(total_steps, checkpoint_step, n_checkpoint_saved, episode.n_steps()):
                    self.save_model_weights()
                    n_checkpoint_saved += 1

            # Check whether logs can be shown and saved
            if self._can_process_iteration(total_steps, log_step, n_log_saved, episode.n_steps()):
                self.log_iteration_results(n_episodes, iteration_loss, episode)
                n_log_saved += 1

            total_steps += episode.n_steps()
            n_episodes += 1

        if update_model:
            self.save_model_weights()

    def _can_process_iteration(self, total_steps, step, n_saved, episode_length):
        return int(total_steps / (step * n_saved)) == 1 \
               or self._is_last_iteration(total_steps, episode_length)

    @staticmethod
    def _is_last_iteration(total_steps, n_steps):
        return total_steps + n_steps > total_steps

    def _create_episode(self):
        # Init trajectory
        self.trajectory_buffer.reset()
        # Reset the environment
        state, first = self.environment.reset()

        iteration = 0
        while True:
            # Select an action
            action, action_probability = self.model.get_action(state)

            # Perform the action and get feedback from environment
            reward, new_state, new_first = self.environment.step(action.numpy())

            if new_first:
                break

            # Value of current state
            # value = self.model.get_value(state)

            # Add element to the current trajectory
            self.trajectory_buffer.add_element(state,
                                               reward,
                                               first,
                                               action,
                                               action_probability,
                                               None,
                                               None,
                                               iteration)

            # Update current variables
            state = new_state
            first = new_first
            iteration += 1

        # Get the single episode
        return self.trajectory_buffer.get_trajectory(flattened=True)

    def _update_model_weights(self, trajectory):
        dict_args = {
            "states": trajectory.states,
            "action_probabilities": trajectory.action_probabilities,
            "returns": trajectory.returns,
            "actions": trajectory.actions
        }

        iteration_loss = self.model.train_step(**dict_args)

        return iteration_loss
