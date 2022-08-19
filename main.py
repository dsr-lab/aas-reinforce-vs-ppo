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
        agent_config = config.ppo_agent_config
    elif config.AGENT_TYPE == 'reinforce':
        trainer_type = ReinforceTrainer
        agent_config = config.reinfoce_agent_config
    else:
        raise NotImplementedError('Agent type not supported. You should choose either ppo or reinforce.')

    return trainer_type, agent_config


def main():
    create_required_directories()

    env = config.ENVIRONMENT_TYPE(**config.environment_config)

    print(f'Policy: {config.AGENT_TYPE}')
    print(f'Environment: {config.ENVIRONMENT_TYPE.__name__}')

    trainer_type, agent_config = get_trainer_configurations()

    agent_config['n_actions'] = env.n_actions

    if config.TRAIN:
        print(f'Train configuration:')
        print(json.dumps(config.train_config, indent=4))
        trainer = trainer_type(environment=env, agent_config=agent_config, trainer_config=config.train_config)
    else:
        print(f'Eval configuration:')
        print(json.dumps(config.evaluation_config, indent=4))
        trainer = trainer_type(environment=env, agent_config=agent_config, trainer_config=config.evaluation_config)

    trainer.train(training=config.TRAIN)


if __name__ == '__main__':
    main()
