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
        self.model = ReinforceAgent(n_actions=environment.n_actions,
                                    backbone_type=backbone_type,
                                    learning_rate=learning_rate,
                                    with_baseline=with_baseline)

        super(ReinforceTrainer, self).__init__(environment=environment,
                                               **trainer_args)

    def update_model_weights(self, trajectory):
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
