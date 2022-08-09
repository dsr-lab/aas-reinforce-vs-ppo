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
        env = config.ENVIRONMENT_TYPE(num=config.PPO_N_AGENTS,
                                      render_mode=config.RENDER_MODE,
                                      save_video=config.SAVE_VIDEO)

        trainer = PPOTrainer(environment=env)
    else:
        env = config.ENVIRONMENT_TYPE(num=1,
                                      render_mode=config.RENDER_MODE,
                                      save_video=config.SAVE_VIDEO)

        trainer = ReinforceTrainer(environment=env)

    trainer.train()


if __name__ == '__main__':
    main()
