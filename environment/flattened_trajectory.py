from environment.episode import Episode
import numpy as np


class FlattenedTrajectory(Episode):
    """
    The FlattenedTrajectory object contains all the episodes gathered from the environmnent
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
        true_rewards = []
        states = []
        actions = []
        action_probabilities = []
        returns = []
        advantages = []
        values = []

        n_win = 0
        n_loss = 0
        n_incomplete = 0

        for e in episodes:
            rewards.append(e.rewards)
            true_rewards.append(e.true_rewards)
            states.append(e.states)
            actions.append(e.actions)
            action_probabilities.append(e.action_probabilities)
            returns.append(e.returns)
            advantages.append(e.advantages)
            values.append(e.values)

            n_win += np.count_nonzero(e.rewards > 0)
            n_loss += np.count_nonzero(e.rewards < 0)

            # Not considering those episodes that are truncated due to the max_episode_steps limit
            n_incomplete += 1 if len(e.rewards) == self.max_game_steps else 0

            # This would considers also episodes trucated due to max_episode_steps limit
            # n_incomplete += 0 if e.rewards[-1] != 0 else 1

        # Sanity check
        assert \
            len(rewards) == len(states) == len(actions) == len(returns) == len(advantages) == len(action_probabilities)

        self.rewards = np.concatenate(rewards)
        self.true_rewards = np.concatenate(true_rewards)
        self.states = np.concatenate(states)
        self.actions = np.concatenate(actions)
        self.returns = np.concatenate(returns)
        self.advantages = np.concatenate(advantages)
        self.action_probabilities = np.concatenate(action_probabilities)
        self.values = np.concatenate(values)

        self.n_episodes = len(episodes)
        self.n_wins = n_win
        self.n_loss = n_loss
        self.n_incomplete = n_incomplete
