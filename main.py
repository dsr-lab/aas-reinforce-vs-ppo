import tensorflow as tf
import gym

from model.ppo import PPO


def main():
    env = gym.make("CartPole-v0")
    ppo = PPO(environment=env)
    ppo.train()

import numpy as np
if __name__ == '__main__':
    a = np.array([[1], [2]])
    main()