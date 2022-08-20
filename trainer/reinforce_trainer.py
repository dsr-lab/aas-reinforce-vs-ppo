from model.reinforce_agent import ReinforceAgent
from environment.env_wrapper import EnvironmentWrapper
from trainer.trainer import Trainer
import numpy as np


class ReinforceTrainer(Trainer):

    def __init__(self,
                 environment: EnvironmentWrapper,
                 agent_config: dict,
                 trainer_config: dict):
        """
        Class used for training a model based on a REINFORCE Agent.

        Parameters
        ----------
        environment: EnvironmentWrapper
            The environment type. Valid values are: NinjaEnvironment, LeaperEnvironment, CoinrunEnvironment
        agent_config: dict
            Dictionary containing all configurations required for the ReiforceAgent
        trainer_config: dict
            Dictionary containing common configurations used for initializing the parent object (Trainer)
        """

        self.model = ReinforceAgent(**agent_config)

        super(ReinforceTrainer, self).__init__(environment=environment, **trainer_config)

    def update_model_weights(self, trajectory):
        """
        Update the model weights. Differently from the PPOTrainer, the trajectory is used only once for updating the
        model weights.

        Parameters
        ----------
        trajectory: FlattenedTrajectory
            The trajectory with all episodes flattened in a single array.

        Returns
        ----------
        iteration_loss: the recorded loss recorded while updating the weights
        """
        iteration_loss = 0

        if self.randomize_samples:
            np.random.shuffle(self.time_step_indices)

        for start in range(0, self.n_batches, self.batch_size):
            end = start + self.batch_size
            interval = self.time_step_indices[start:end]

            dict_args = {
                "states": trajectory.states[interval],
                "returns": trajectory.returns[interval],
                "actions": trajectory.actions[interval],
            }

            iteration_loss += self.model.train_step(**dict_args)

        iteration_loss /= self.n_batches

        return iteration_loss
