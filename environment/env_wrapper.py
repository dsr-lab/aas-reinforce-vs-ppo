from procgen import ProcgenGym3Env
import numpy as np
import gym3
import enum


class RenderMode(enum.Enum):
    off = None
    on = 'rgb_array'


class EnvironmentWrapper:
    def __init__(self,
                 num: int,
                 env_name: str,
                 render_mode: RenderMode):
        self.environment = ProcgenGym3Env(num,
                                          env_name=env_name,
                                          render_mode=render_mode.value,
                                          distribution_mode="easy")

        if render_mode == RenderMode.on:
            self.environment = gym3.ViewerWrapper(self.environment, info_key='rgb')

        self.initial_state = self._get_state()

        # Define the subset of actions that are available for the chosen game
        # self.action_mapping = {
        #     0: 1,  # LEFT
        #     1: 7,  # RIGHT
        #     2: 5,  # UP
        #     # 3: 3,   # DOWN
        #     3: 4,  # NONE
        # }
        self.action_mapping = {
            0: 0,
            1: 1,
            2: 2,
            3: 3,
            4: 4,
            5: 5,
            6: 6,
            7: 7,
            8: 8,
            9: 9,
            10: 10,
            11: 11,
            12: 12,
            13: 13,
            14: 14}
        #     0: 1,  # LEFT
        #     1: 7,  # RIGHT
        #     2: 5,  # UP
        #     # 3: 3,   # DOWN
        #     3: 4,  # NONE
        # }

        """
        PLUNDER
        self.action_mapping = {
            0: 1,  # LEFT
            1: 7,  # RIGHT
            # 2: 5,# UP
            2: 9,  # FIRE
            # 3: 3,   # DOWN
            3: 4,  # NONE
        }
        """
        # 9 = SPARA NINJA RIGHT
        # 10 = SPARA IN ALTO RIGHT
        """
            return [
                ("LEFT", "DOWN"),
                ("LEFT",),
                ("LEFT", "UP"),
                ("DOWN",),
                (),
                ("UP",),
                ("RIGHT", "DOWN"),
                ("RIGHT",),
                ("RIGHT", "UP"),
                ("D",), 9
                ("A",), 10
                ("W",), 11
                ("S",), 12
                ("Q",), 13
                ("E",), 14
            ]
        """

    def step(self, actions):
        actions = np.array([*map(self.action_mapping.get, actions)])
        # actions = np.array([11] * 32)

        self.environment.act(actions)
        return self._observe()

    def reset(self):
        self._set_state(self.initial_state)
        _, obs, first = self._observe()
        return obs, first

    def get_n_actions(self):
        return len(self.action_mapping)

    def get_state_shape(self):
        return self.environment.ob_space['rgb'].shape

    def _get_state(self):
        return self.environment.callmethod("get_state")

    def _set_state(self, state):
        return self.environment.callmethod("set_state", state)

    def _observe(self):
        rew, obs, first = self.environment.observe()

        obs = self._normalize_observation(obs)

        return rew, obs, first

    @staticmethod
    def _normalize_observation(obs):
        obs = obs['rgb']
        obs = obs / 255.0

        return obs

# class LeaperEnvironment(Environment):
#     def __init__(self):
#         super(LeaperEnvironment, self).__init__(
#             num=1,
#             env_name='leaper',
#             render_mode=RenderModes.off
#         )
