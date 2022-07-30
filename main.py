
import gym
from model.ppo import PPO

# from procgen import ProcgenGym3Env
# import tensorflow as tf
# import gym3
# import random
# import matplotlib.pyplot as plt
# import numpy as np

def main():
    env = gym.make("CartPole-v0")
    ppo = PPO(environment=env)
    ppo.train()


if __name__ == '__main__':

    # env = ProcgenGym3Env(num=1, env_name="leaper", render_mode="rgb_array")
    # # state = env.callmethod("get_state")
    # # env.callmethod("set_state", state)
    # # env = gym3.ViewerWrapper(env, info_key="rgb")
    #
    #
    # ACTION_LEFT = 1
    # ACTION_RIGHT = 7
    # ACTION_DOWN = 3
    # ACTION_UP = 5
    # ACTION_NONE = 4
    #
    # actions = [ACTION_LEFT, ACTION_RIGHT, ACTION_DOWN, ACTION_UP, ACTION_NONE]
    #
    #
    # # obs2 = env.callmethod('get_state')
    #
    # rew, obs, first = env.observe()
    # step = 0
    # rewards=[]
    #
    # while True:
    #
    #     env.act(gym3.types_np.sample(env.ac_space, bshape=(env.num,)))
    #     rew, obs, first = env.observe()
    #
    #     rewards.append(rew)


    main()