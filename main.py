import tensorflow as tf
import gym

from model.ppo import PPO


def compute_discounted_cumulative_sum(x, discount_rate):

    n_elements = len(x)
    result = tf.zeros(n_elements)
    last_value = 0

    for i, r in enumerate(input[::-1]):
        result[n_elements - i - 1] = r + last_value * discount_rate
        last_value = result[n_elements - i - 1]


def main():
    env = gym.make("CartPole-v0")
    ppo = PPO(environment=env)
    ppo.train()


if __name__ == '__main__':
    main()