import config
from environment.env_wrapper import EnvironmentWrapper, RenderMode
from model_trainer import ModelTrainer


def main():
    game_name = 'ninja'
    print(game_name)
    env = EnvironmentWrapper(num=config.N_AGENTS, env_name=config.GAME_NAME, render_mode=RenderMode.off)
    model_trainer = ModelTrainer(environment=env)
    model_trainer.train()


if __name__ == '__main__':

    main()




