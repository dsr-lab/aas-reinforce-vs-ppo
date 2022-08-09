import config
from model.agent import Agent
from model.reinforce_agent import ReinforceAgent
from environment.env_wrapper import EnvironmentWrapper
from environment.trajectory_buffer import TrajectoryBuffer
from trainer.trainer import Trainer


class ReinforceTrainer(Trainer):

    def __init__(self, environment: EnvironmentWrapper):
        super(ReinforceTrainer, self).__init__(environment=environment)

    def init_agent(self, backbone_type) -> Agent:
        return ReinforceAgent(n_actions=self.environment.n_actions,
                              backbone_type=backbone_type)

    def init_trajectory_buffer(self, set_negative_rewards_for_losses) -> TrajectoryBuffer:
        # Init the buffer
        return TrajectoryBuffer(max_agent_steps=self.environment.get_max_game_steps(),
                                max_game_steps=self.environment.get_max_game_steps(),
                                n_agents=1,
                                obervation_shape=(64, 64, 3),
                                set_negative_rewards_for_losses=set_negative_rewards_for_losses)

    def can_save_weights(self, iteration):
        return config.SAVE_WEIGHTS and (iteration % 1000 == 0)

    def train(self):
        total_steps = 0
        n_episodes = 0
        while total_steps < config.REINFORCE_ITERATIONS:
            episode = self._create_episode()

            iteration_loss = self._update_model_weights(episode)

            self.compute_post_iteration_operations(n_episodes, iteration_loss, episode)

            total_steps += episode.n_steps()
            n_episodes += 1

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
            value = self.model.get_value(state)

            # Add element to the current trajectory
            self.trajectory_buffer.add_element(state,
                                               reward,
                                               first,
                                               action,
                                               action_probability,
                                               value,
                                               None,
                                               iteration)

            # Update current variables
            state = new_state
            first = new_first
            iteration += 1

        # Get the single episode
        return self.trajectory_buffer.get_trajectory(flattened=False)[0]

    def _update_model_weights(self, trajectory):
        dict_args = {
            "states": trajectory.states,
            "action_probabilities": trajectory.action_probabilities,
            "returns": trajectory.returns,
        }

        iteration_loss = self.model.train_step(**dict_args)

        return iteration_loss
