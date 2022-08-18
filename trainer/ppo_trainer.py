import numpy as np

from model.ppo_agent import PPOAgent
from environment.env_wrapper import EnvironmentWrapper
from trainer.trainer import Trainer


class PPOTrainer(Trainer):

    def __init__(self,
                 environment: EnvironmentWrapper,
                 clip_ratio=0.2,
                 backbone_type='impala',

                 learning_rate=5e-4,
                 entropy_bonus_coefficient=0.01,
                 critic_loss_coefficient=0.5,
                 clip_value_estimates=False,
                 normalize_advantages=False,
                 **trainer_args):

        self.normalize_advantages = normalize_advantages
        self.model = PPOAgent(n_actions=environment.n_actions,
                              backbone_type=backbone_type,
                              clip_ratio=clip_ratio,
                              clip_value_estimates=clip_value_estimates,
                              learning_rate=learning_rate,
                              entropy_bonus_coefficient=entropy_bonus_coefficient,
                              critic_loss_coefficient=critic_loss_coefficient)

        super(PPOTrainer, self).__init__(environment=environment, **trainer_args)

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
