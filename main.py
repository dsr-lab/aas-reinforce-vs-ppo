import tensorflow as tf
import gym3
import gym
from procgen import ProcgenGym3Env

from model.ppo import PPO


def main():
    env = gym.make("CartPole-v0")
    ppo = PPO(environment=env)
    ppo.train()

import numpy as np
if __name__ == '__main__':

    # env = ProcgenGym3Env(num=4, env_name="leaper", render_mode="rgb_array")
    # env = gym3.ViewerWrapper(env, info_key="rgb")
    # a = env
    # step = 0
    # rewards=[]
    # while True:
    #     env.act(gym3.types_np.sample(env.ac_space, bshape=(env.num,)))
    #     rew, obs, first = env.observe()
    #     if first.any():
    #         print()
    #     print(f"step {step} reward {rew} first {first}")
    #     step += 1
    #     rewards.append(rew)

    # r is set to f1().
    # Operations in f2 (e.g., tf.add) are not executed


    main()