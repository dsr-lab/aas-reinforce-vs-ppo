import config
from pathlib import Path

from trainer.ppo_trainer import PPOTrainer
from trainer.reinforce_trainer import ReinforceTrainer


def create_required_directories():
    Path(config.LOGS_PATH).mkdir(parents=True, exist_ok=True)
    Path(config.WEIGHTS_PATH).mkdir(parents=True, exist_ok=True)


def main():
    create_required_directories()

    if config.AGENT_TYPE == 'ppo':
        env = config.ENVIRONMENT_TYPE(**config.ppo_environment_config)

        trainer = PPOTrainer(environment=env,
                             **config.ppo_trainer_config,
                             **config.trainer_common_config)
    else:
        env = config.ENVIRONMENT_TYPE(**config.reinforce_environment_config)

        trainer = ReinforceTrainer(environment=env,
                                   **config.reinforce_trainer_config,
                                   **config.trainer_common_config)

    trainer.train()


if __name__ == '__main__':
    main()
