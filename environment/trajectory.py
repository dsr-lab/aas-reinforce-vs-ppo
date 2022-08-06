from environment.episode import Episode
import numpy as np


class Trajectory(Episode):
    """
    The trajectory object contains all the episodes gathered from the environmnent
    """
    def __init__(self, episodes: [Episode], max_game_steps=1000):
        super().__init__()

        self.n_episodes = 0
        self.n_wins = 0
        self.n_loss = 0
        self.n_incomplete = 0
        self.max_game_steps = max_game_steps

        self._flat_episodes(episodes)

    def _flat_episodes(self, episodes: [Episode]):
        rewards = []
        states = []
        actions = []
        action_probabilities = []
        returns = []
        advantages = []

        n_win = 0
        n_loss = 0
        n_incomplete = 0

        for e in episodes:
            rewards.append(e.rewards)
            states.append(e.states)
            actions.append(e.actions)
            action_probabilities.append(e.action_probabilities)
            returns.append(e.returns)
            advantages.append(e.advantages)
            n_win += np.count_nonzero(e.rewards > 0)
            n_loss += np.count_nonzero(e.rewards < 0)

            # Not considering those episodes that are truncated due to the max_episode_steps limit
            n_incomplete += 1 if len(e.rewards) == self.max_game_steps else 0

            # This would considers also episodes trucated due to max_episode_steps limit
            # n_incomplete += 0 if e.rewards[-1] != 0 else 1

        assert \
            len(rewards) == len(states) == len(actions) == len(returns) == len(advantages) == len(action_probabilities)

        self.rewards = np.concatenate(rewards)
        self.states = np.concatenate(states)
        self.actions = np.concatenate(actions)
        self.returns = np.concatenate(returns)
        self.advantages = np.concatenate(advantages)
        self.action_probabilities = np.concatenate(action_probabilities)

        self.n_episodes = len(episodes)
        self.n_wins = n_win
        self.n_loss = n_loss
        self.n_incomplete = n_incomplete
