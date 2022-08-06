from abc import abstractmethod

from procgen import ProcgenGym3Env
import numpy as np
import gym3
import enum
import warnings


class RenderMode(enum.Enum):
    off = None
    on = 'rgb_array'


class EnvironmentWrapper:
    def __init__(self,
                 num: int,
                 env_name: str,
                 render_mode: RenderMode = None,
                 save_video: bool = False,
                 model_output_to_actions: dict = None):

        self.environment = ProcgenGym3Env(num,
                                          env_name=env_name,
                                          render_mode=render_mode.value,
                                          distribution_mode="easy")

        self.n_actions, self.model_output_to_actions = self._configure_actions(model_output_to_actions)

        self._set_wrapper(render_mode, save_video)

        self.initial_state = self._get_state()

        # Logs
        print(f'GAME: {env_name}')

        if model_output_to_actions is None:
            print(f'Using all the {self.n_actions} available actions.')
        else:
            print(f'Using {self.n_actions}, which is a subset of the available actions.')



    @abstractmethod
    def get_max_game_steps(self):
        pass

    def step(self, actions):
        actions = np.array([*map(self.model_output_to_actions.get, actions)])

        # Testing
        # actions = gym3.types_np.sample(self.environment.ac_space, bshape=(self.environment.num,))
        # actions = np.array([4] * self.environment.num)

        self.environment.act(actions)
        return self._observe()

    def reset(self):
        self._set_state(self.initial_state)
        _, obs, first = self._observe()
        return obs, first

    def get_state_shape(self):
        return self.environment.ob_space['rgb'].shape

    def _configure_actions(self, model_output_to_actions):

        if model_output_to_actions is None:
            n_actions = len(self.environment.get_combos())
            key_value = np.arange(0, len(self.environment.get_combos()))
            model_output_to_actions = {k: v for k, v in zip(key_value, key_value)}
        else:
            n_actions = len(model_output_to_actions)

        return n_actions, model_output_to_actions

    def _set_wrapper(self, render_mode, save_video):
        if render_mode == RenderMode.on:
            if save_video:
                self.environment = gym3.VideoRecorderWrapper(self.environment,
                                                             'recordings',
                                                             info_key='rgb')
            else:
                self.environment = gym3.ViewerWrapper(self.environment, info_key='rgb')

        # Warn in case of bad configuration
        if render_mode == RenderMode.off and save_video:
            warnings.warn("Warning: save_video flag is True, and render_mode is OFF. If you need to record the video,"
                          "then you should change the render_mode to ON")

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


class NinjaEnvironment(EnvironmentWrapper):

    def __init__(self, num, render_mode: RenderMode, save_video):
        # model_output_to_actions = {
        #     0: 1,  # LEFT
        #     1: 4,  # NONE
        #     2: 5,  # JUMP (UP)
        #     3: 7,  # RIGHT
        #     4: 9,  # FIRE RIGHT
        #     5: 10,  # FIRE RIGHT-UP
        #     6: 11,  # FIRE UP
        #     7: 12,  # FIRE DOWN
        # }

        super(NinjaEnvironment, self).__init__(num=num,
                                               env_name='ninja',
                                               render_mode=render_mode,
                                               save_video=save_video,
                                               model_output_to_actions=None)

    def get_max_game_steps(self):
        return 1000


class LeaperEnvironment(EnvironmentWrapper):
    def __init__(self, num, render_mode: RenderMode, save_video):
        # model_output_to_actions = {
        #     0: 1,  # LEFT
        #     1: 4,  # NONE
        #     2: 5,  # UP
        #     3: 7   # RIGHT
        # }

        super(LeaperEnvironment, self).__init__(num=num,
                                                env_name='leaper',
                                                render_mode=render_mode,
                                                save_video=save_video,
                                                model_output_to_actions=None)

    @abstractmethod
    def get_max_game_steps(self):
        return 500


class CoinrunEnvironment(EnvironmentWrapper):
    def __init__(self, num, render_mode: RenderMode, save_video):
        # model_output_to_actions = {
        #     0: 1,  # LEFT
        #     1: 4,  # NONE
        #     2: 5,  # UP
        #     3: 7   # RIGHT
        # }

        super(CoinrunEnvironment, self).__init__(num=num,
                                                 env_name='coinrun',
                                                 render_mode=render_mode,
                                                 save_video=save_video,
                                                 model_output_to_actions=None)

    @abstractmethod
    def get_max_game_steps(self):
        return 1000
