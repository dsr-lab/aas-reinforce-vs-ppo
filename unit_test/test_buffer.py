from unittest import TestCase
import numpy as np

from environment.trajectory_buffer import TrajectoryBuffer


class TestBuffer(TestCase):

    @classmethod
    def setUpClass(cls):

        """
        Init variables used from all the tests
        """

        cls.buffer_size = 10

        # Arrange
        cls.rewards = np.asarray(
            [
                [0, 0],
                [0, 0],
                [0, 10],
                [0, 0],  # Episode end at [3, 1] (Success)
                [0, 0],   # Episode end at [4, 0] (Fail)
                [0, 0],
                [0, 0],   # Episode end at [6, 1] (Fail)
                [10, 0],
                [0, 10],
                [0, 0]   # Episode end at [9, 0] (Success)
            ])

        cls.first = np.asarray(
            [
                [True, True],
                [False, False],
                [False, False],
                [False, True],
                [True, False],
                [False, False],
                [False, True],
                [False, False],
                [True, False],
                [False, True]
            ])

        n_agents = 2
        state_shape = (3, 64, 64)

        cls.states = np.random.rand(*(cls.buffer_size, n_agents,) + state_shape)

    def test_get_buffer_expected_number_of_episodes(self):

        # Arrange
        buffer = TrajectoryBuffer(max_agent_steps=self.buffer_size)
        buffer.states = self.states
        buffer.firsts = self.first
        buffer.rewards = self.rewards

        # Act
        _ = buffer.get_trajectory()
        episodes = buffer.episodes

        # Assert
        self.assertEqual(7, len(episodes))

    def test_get_buffer_state_equalities(self):

        # Arrange
        buffer = TrajectoryBuffer(max_agent_steps=self.buffer_size)
        buffer.states = self.states
        buffer.firsts = self.first
        buffer.rewards = self.rewards

        # Act
        _ = buffer.get_trajectory()
        episodes = buffer.episodes

        # Assert
        np.testing.assert_array_equal(self.states[0:4, 0], episodes[0].states)
        np.testing.assert_array_equal(self.states[4:8, 0], episodes[1].states)
        np.testing.assert_array_equal(self.states[8:10, 0], episodes[2].states)
        np.testing.assert_array_equal(self.states[0:3, 1], episodes[3].states)
        np.testing.assert_array_equal(self.states[3:6, 1], episodes[4].states)
        np.testing.assert_array_equal(self.states[6:9, 1], episodes[5].states)
        np.testing.assert_array_equal(self.states[9:10, 1], episodes[6].states)

    def test_get_buffer_all_first_false_expected_number_of_episodes(self):

        # Arrange
        rewards = np.asarray(
            [
                [0, 0],
                [0, 0],
                [0, 0],
                [0, 0],  # Episode end at [3, 1] (Success)
                [0, 0],  # Episode end at [4, 0] (Fail)
                [0, 0],
                [0, 0],  # Episode end at [6, 1] (Fail)
                [0, 0],
                [0, 0],
                [0, 0]  # Episode end at [9, 0] (Success)
            ])

        first = np.asarray(
            [
                [False, False],
                [False, False],
                [False, False],
                [False, False],
                [False, False],
                [False, False],
                [False, False],
                [False, False],
                [False, False],
                [False, False]
            ])

        buffer = TrajectoryBuffer(max_agent_steps=self.buffer_size)
        buffer.state = self.states
        buffer.first = first
        buffer.reward = rewards

        # Act
        _ = buffer.get_trajectory()
        episodes = buffer.episodes

        # Assert
        self.assertEqual(2, len(episodes))

    def test_get_buffer_two_first_true_expected_number_of_episodes(self):

        # Arrange
        rewards = np.asarray(
            [
                [0, 0],
                [0, 0],
                [0, 0],
                [0, 0],  # Episode end at [3, 1] (Success)
                [0, 0],  # Episode end at [4, 0] (Fail)
                [0, 0],
                [0, 0],  # Episode end at [6, 1] (Fail)
                [0, 0],
                [0, 10],
                [0, 0]  # Episode end at [9, 0] (Success)
            ])

        first = np.asarray(
            [
                [False, False],
                [True, False],
                [False, False],
                [False, False],
                [False, False],
                [False, False],
                [False, False],
                [False, False],
                [False, False],
                [False, True]
            ])

        buffer = TrajectoryBuffer(max_agent_steps=self.buffer_size)
        buffer.states = self.states
        buffer.firsts = first
        buffer.rewards = rewards

        # Act
        _ = buffer.get_trajectory()
        episodes = buffer.episodes

        # Assert
        self.assertEqual(4, len(episodes))
