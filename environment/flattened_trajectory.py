from environment.episode import Episode
import numpy as np


class FlattenedTrajectory(Episode):
    """
    The FlattenedTrajectory object contains all the episodes gathered from the environmnent.
    Basically, it is a very big Episode (i.e., the parent class), with some additional variables used for statistical
    information.
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

        # Loop all episodes
        for e in episodes:
            rewards.append(e.rewards)
            true_rewards.append(e.true_rewards)
            states.append(e.states)
            actions.append(e.actions)
            action_probabilities.append(e.action_probabilities)
            returns.append(e.returns)
            advantages.append(e.advantages)
            values.append(e.values)

            # Set metrics
            n_win += np.count_nonzero(e.rewards > 0)
            n_loss += np.count_nonzero(e.rewards < 0)
            n_incomplete += 1 if len(e.rewards) == self.max_game_steps else 0

        # Sanity check
        assert \
            len(rewards) == len(states) == len(actions) == len(returns) == len(advantages) == len(action_probabilities)

        # Flat everything
        self.rewards = np.concatenate(rewards)
        self.true_rewards = np.concatenate(true_rewards)
        self.states = np.concatenate(states)
        self.actions = np.concatenate(actions)
        self.returns = np.concatenate(returns)
        self.advantages = np.concatenate(advantages)
        self.action_probabilities = np.concatenate(action_probabilities)
        self.values = np.concatenate(values)

        # Set metrics
        self.n_episodes = len(episodes)
        self.n_wins = n_win
        self.n_loss = n_loss
        self.n_incomplete = n_incomplete
