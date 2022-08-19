import numpy as np

from model.ppo_agent import PPOAgent
from environment.env_wrapper import EnvironmentWrapper
from trainer.trainer import Trainer


class PPOTrainer(Trainer):

    def __init__(self,
                 environment: EnvironmentWrapper,
                 agent_config: dict,
                 trainer_config: dict):
        """
        Class used for training a model based on a PPO Agent.

        Parameters
        ----------
        environment: EnvironmentWrapper
            The environment type. Valid values are: NinjaEnvironment, LeaperEnvironment, CoinrunEnvironment
        agent_config: dict
            Dictionary containing all configurations required for the Agent
        trainer_config: dict
            Dictionary containing common configurations used for initializing the parent object (Trainer)
        """

        self.model = PPOAgent(**agent_config)

        super(PPOTrainer, self).__init__(environment=environment, **trainer_config)

    def update_model_weights(self, trajectory):
        """
        Update the model weights according to the PPO paper (i.e., use the same trajectory for update the model
        multiple times).

        Parameters
        ----------
        trajectory: FlattenedTrajectory
            The trajectory with all episodes flattened in a single array.

        Returns
        ----------
        iteration_loss: the recorded loss recorded while updating the weights
        """
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
                    "action_probabilities": trajectory.action_probabilities[interval],
                    "advantages": self.normalize(trajectory.advantages[interval]) if self.normalize_advantages else
                    trajectory.advantages[interval],
                    "returns": trajectory.returns[interval],
                    "old_values": trajectory.values[interval]
                }

                iteration_loss += self.model.train_step(**dict_args)

        iteration_loss /= (self.n_batches * self.epochs_model_update)

        return iteration_loss
