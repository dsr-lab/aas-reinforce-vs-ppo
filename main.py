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
    x = tf.constant([-1, -2, 3, 4])

    a = tf.where(x>0)
    b = tf.where(x<=0)

    BATCH_SIZE = 256

    for i, batch_size in enumerate(range(0, 4096, 256)):
        print(BATCH_SIZE * i, batch_size+BATCH_SIZE)

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
    # Operations in f2 (e.g., tf.add) are not executed.
    nenvs = 1
    nsteps = 4000
    nminibatches = 4
    nbatch = nenvs * nsteps
    nbatch_train = nbatch // nminibatches
    noptepochs = 2
    inds = np.arange(nbatch)

    for _ in range(noptepochs):
        # Randomize the indexes
        np.random.shuffle(inds)
        # 0 to batch_size with batch_train_size step
        for start in range(0, nbatch, nbatch_train):
            end = start + nbatch_train
            mbinds = inds[start:end]
            pass


    main()