import numpy as np

from environment.episode import Episode
from environment.trajectory import Trajectory


class TrajectoryBuffer:

    def __init__(self,
                 max_agent_steps=1000,
                 max_game_steps=1000,
                 n_agents=2,
                 obervation_shape=(64, 64, 3)):

        self.n_agents = n_agents
        self.obervation_shape = obervation_shape
        self.max_agent_steps = max_agent_steps
        self.max_game_steps = max_game_steps

        self.states = None
        self.rewards = None
        self.firsts = None
        self.actions = None
        self.action_probabilities = None
        self.values = None
        self.new_values = None
        self.episodes = None

        self.reset()

    def add_element(self, state, reward, first, action, action_probability, value, new_value, step):

        self.states[step] = state
        self.rewards[step] = reward
        self.firsts[step] = first
        self.actions[step] = action
        self.values[step] = value
        self.new_values[step] = new_value
        self.action_probabilities[step] = action_probability

    def get_trajectory(self):

        self._penalize_game_fails()

        self._force_at_least_one_episode_per_agent()

        self._split_in_episodes()

        return Trajectory(self.episodes, max_game_steps=self.max_game_steps)

    def reset(self):
        self.states = np.zeros((self.max_agent_steps, self.n_agents,) + self.obervation_shape, dtype=np.float32)
        self.rewards = np.zeros((self.max_agent_steps, self.n_agents,), dtype=np.float32)
        self.firsts = np.zeros((self.max_agent_steps, self.n_agents,), dtype=np.bool_)
        self.actions = np.zeros((self.max_agent_steps, self.n_agents,), dtype=np.int32)
        self.action_probabilities = np.zeros((self.max_agent_steps, self.n_agents,), dtype=np.float32)
        self.values = np.zeros((self.max_agent_steps, self.n_agents,), dtype=np.float32)
        self.new_values = np.zeros((self.max_agent_steps, self.n_agents,), dtype=np.float32)

        self.episodes = []

    def _penalize_game_fails(self):
        """
        This method is necessary to penalize the losses, because the environment always returns a 0-reward also
        in case of a loss.

        What we have is the firsts array, which contains the position of new episodes. Example:

            | time_step  |  new_episode  |
            -----------------------------
            |    0       |     True      |
            |    1       |     False     |
            |    2       |     False     |
            |    3       |     False     |
            |    4       |     True      |
            |    5       |     False     |
            |    6       |     False     |

        At time_step=4 we have the beginning of a new episode, therefore it is necessary to check if the reward
        at the previous time step:
          - if > 0 ==> Game won, do not penalize
          - if = 0 ==> Game lost, penalize

        Additionally:
          - The first row is always True, because at the beginning of a trajectory the environment is reset,
            and new episodes are started. Thus, that line is removed from the array, also because it would not
            be possible to penalize previous time steps.
          - In order to keep the dimensions a new row is added at the end, composed by only False values. This happens
            because we never need to possibly penalize the last row of a trajectory, because we cannot know wheter the
            next time step would cause a win or a loss.


        """
        shifted_first = self.firsts[1:, :]
        shifted_first = np.append(shifted_first, [[False]*self.n_agents], axis=0)
        
        self.rewards[shifted_first & (self.rewards <= 0)] = -1

    def _force_at_least_one_episode_per_agent(self):
        """
        This should not be necessary, because at the beginning of a new trajectory the environment is reset, and
        new episodes are automatically started. Nevertheless, this function has been implemented also for testing
        purposes (see test package).
        """
        self.firsts[0, :] = True

    def _split_in_episodes(self):

        # Get indices of new episodes
        agents, time_steps = np.where(self.firsts.T)

        time_step_start = 0

        # Cycle each timestep for each agent
        for i, (time_step, agent) in enumerate(zip(time_steps, agents)):

            # Complete agent episode time_steps in edge cases
            if self._is_next_agent_different(i, agent, agents) or self._is_last_time_step(i, time_steps):
                self._complete_agent_timesteps(time_step_start, agent)
                time_step_start = 0
                continue

            # Set indices
            time_step_end = time_steps[i+1]

            # Create the episode
            self.episodes.append(self._create_episode(time_step_start, time_step_end, agent))

            # Update indices
            time_step_start = time_step_end

    @staticmethod
    def _is_last_time_step(i, time_steps):
        return i == len(time_steps) - 1

    @staticmethod
    def _is_next_agent_different(i, agent, agents):
        return i != len(agents)-1 and agent != agents[i+1]

    def _complete_agent_timesteps(self, row_start, col):
        if row_start < self.max_agent_steps:
            self.episodes.append(self._create_episode(row_start, self.max_agent_steps, col))

    def _create_episode(self, row_start, row_end, col):
        episode = Episode()

        episode.rewards = self.rewards[row_start:row_end, col]
        episode.states = self.states[row_start:row_end, col]
        episode.actions = self.actions[row_start:row_end, col]
        episode.action_probabilities = self.action_probabilities[row_start:row_end, col]
        episode.values = self.values[row_start:row_end, col]

        v_t_next = 0
        if episode.rewards[-1] == 0:
            v_t_next = self.new_values[row_start:row_end, col][-1]

        episode.compute_advantages(v_t_next=v_t_next)
        episode.compute_returns()

        return episode








