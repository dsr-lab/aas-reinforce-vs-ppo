from model.reinforce_agent import ReinforceAgent
from environment.env_wrapper import EnvironmentWrapper
from trainer.trainer import Trainer
import numpy as np


class ReinforceTrainer(Trainer):

    def __init__(self,
                 environment: EnvironmentWrapper,
                 backbone_type='impala',
                 learning_rate=5e-4,
                 with_baseline=True,
                 **trainer_args):
        """
        Class used for training a model based on a REINFORCE Agent.

        Parameters
        ----------
        environment: EnvironmentWrapper
            The environment type. Valid values are: NinjaEnvironment, LeaperEnvironment, CoinrunEnvironment
        backbone_type: str
            The feature extractor/backbone type to use. Valid values are: 'impala', 'nature'
        learning_rate: float
            The learning rate used for initializing the Optimizer
        with_baseline: bool
            If True, then also the baseline is considered during the training
        trainer_args: dict
            Dictionary containing common configurations used for initializing the parent object (Trainer)

        """
        self.model = ReinforceAgent(n_actions=environment.n_actions,
                                    backbone_type=backbone_type,
                                    learning_rate=learning_rate,
                                    with_baseline=with_baseline)

        super(ReinforceTrainer, self).__init__(environment=environment,
                                               **trainer_args)

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
