import config
from environment.env_wrapper import EnvironmentWrapper, RenderMode
from model_trainer import ModelTrainer

import random
import numpy as np
import tensorflow as tf


def set_seeds():
    random.seed(3105)
    np.random.seed(3105)
    tf.random.set_seed(3105)


def main():
    set_seeds()
    env = EnvironmentWrapper(num=config.N_AGENTS, env_name=config.GAME_NAME, render_mode=RenderMode.off)
    model_trainer = ModelTrainer(environment=env)
    model_trainer.train()


if __name__ == '__main__':
    main()




