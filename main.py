import json
import config
from pathlib import Path

from trainer.ppo_trainer import PPOTrainer
from trainer.reinforce_trainer import ReinforceTrainer


def create_required_directories():
    Path(config.LOGS_PATH).mkdir(parents=True, exist_ok=True)
    Path(config.WEIGHTS_PATH).mkdir(parents=True, exist_ok=True)


def get_trainer_configurations():

    if config.AGENT_TYPE == 'ppo':
        trainer_type = PPOTrainer
        training_config = config.ppo_train_config
    elif config.AGENT_TYPE == 'reinforce':
        trainer_type = ReinforceTrainer
        training_config = config.reinfoce_train_config
    else:
        raise NotImplementedError('Agent type not supported. You should choose either ppo or reinforce.')

    # Evaluation config is the same regardless the agent type
    evaluation_config = config.evaluation_config

    return trainer_type, training_config, evaluation_config


def main():
    create_required_directories()

    env = config.ENVIRONMENT_TYPE(**config.environment_config)

    print(f'Policy: {config.AGENT_TYPE}')
    print(f'Environment: {config.ENVIRONMENT_TYPE.__name__}')

    trainer_type, training_config, evaluation_config = get_trainer_configurations()

    if config.TRAIN:
        print(f'Train configuration:')
        print(json.dumps(training_config, indent=4))
        trainer = trainer_type(environment=env, **training_config)
    else:
        print(f'Eval configuration:')
        print(json.dumps(evaluation_config, indent=4))
        trainer = trainer_type(environment=env, **evaluation_config)

    trainer.train(training=config.TRAIN)


if __name__ == '__main__':
    main()
