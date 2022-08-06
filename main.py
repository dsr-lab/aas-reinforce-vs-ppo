import config
from model_trainer import ModelTrainer
from pathlib import Path


def create_required_directories():
    Path(config.LOGS_PATH).mkdir(parents=True, exist_ok=True)
    Path(config.WEIGHTS_PATH).mkdir(parents=True, exist_ok=True)


def main():
    create_required_directories()

    env = config.ENVIRONMENT_TYPE(num=config.N_AGENTS,
                                  render_mode=config.RENDER_MODE,
                                  save_video=config.SAVE_VIDEO)

    model_trainer = ModelTrainer(environment=env)
    model_trainer.train()


if __name__ == '__main__':
    main()
