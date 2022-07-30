import numpy as np
import tensorflow as tf
import scipy.signal

class Trajectory:

    def __init__(self, gamma=0.99, lamda=0.97):
        self.gamma = gamma
        self.lamda = lamda

        self.new_trajectory()

    def new_trajectory(self):
        self._init_trajectory_variables()
        self._init_episode_variables()

    def add_episode_observation(self, state, action, reward, value, action_probability, action_distribution):

        state = tf.squeeze(state, axis=0)
        action = tf.squeeze(action, axis=0)
        action_probability = tf.squeeze(action_probability, axis=0)

        self.episode_states.append(state)
        self.episode_actions.append(action)
        self.episode_rewards.append(reward)
        self.episode_values.append(value)
        self.episode_action_probabilities.append(action_probability)
        self.episode_actions_distribution.append(action_distribution)

    def complete_episode(self, value):
        # self.add_episode_observation(None, None, value, value, None, end_state=True)

        rewards = np.append(self.episode_rewards, value)
        values = np.append(self.episode_values, value)

        """
        Generalized Advantage Estimate (GAE): https://arxiv.org/abs/1506.02438
        
        Rather than using directly the expected return for computing the policy step, it is typically better to use
        an advantage function, which goal is to reduce the variance.
        
            Aₜ(s,a) = Eπ[Gₜ | Sₜ=s, Aₜ=a] - V(s) = δₜ
        
            where: 
                
                Eπ[Gₜ | Sₜ=s, Aₜ=a] = r₀ + γ * v(s') 
            
        Typically, more time steps are considered, especially for reducing the bias. However, adding time steps
        could increase the variance.
        In order to avoid this, the GAE paper demonstrated that doing a discounted sum of the advantages A(s,a) provides
        the best results, with a lower variance.
        
        Thus:
            Aₜ(s,a) = Ʃ(γλ)ᵗ δₜ₊₁(s,a)
        """
        adv = np.zeros((len(rewards)-1), dtype=np.float64)
        adv_t_prev = 0
        # for t in range(len(rewards)-1):
        for t in range(len(adv), 0, -1):
            r_t = rewards[t-1]
            v_t_next = values[t]
            v_t = values[t-1]

            delta = r_t + self.gamma * v_t_next - v_t
            adv_t = delta + self.gamma * self.lamda * adv_t_prev

            adv_t_prev = adv_t

            adv[len(adv) - t] = adv_t

        adv = adv[::-1]
        delta = rewards[:-1] + self.gamma * values[1:] - values[:-1]

        advantages = self._compute_discounted_cumulative_sum(delta, self.gamma * self.lamda)
        returns = self._compute_discounted_cumulative_sum(rewards, self.gamma)[:-1]

        # advantages2 = self.gae(rewards, values, self.gamma, self.lamda)
        # returns2 = self.discount_returns(rewards,self.gamma)

        returns_mean, returns_std = (
            np.mean(returns),
            np.std(returns),
        )
        if returns_std != 0:
            returns = (returns - returns_mean) / returns_std

        advantages_mean, advantages_std = (
            np.mean(advantages),
            np.std(advantages),
        )
        if advantages_std != 0:
            returns = (advantages - advantages_mean) / advantages_std

        self.trajectory_states.append(self.episode_states)
        self.trajectory_actions.append(self.episode_actions)
        self.trajectory_action_probabilities.append(self.episode_action_probabilities)
        self.trajectory_returns.append(returns)
        self.trajectory_advantages.append(advantages)
        self.trajectory_actions_distribution.append(self.episode_actions_distribution)

        self._init_episode_variables()

    def get_trajectory(self):

        return self._flat_list(self.trajectory_states, np.float32), \
               self._flat_list(self.trajectory_actions, np.int32), \
               self._flat_list(self.trajectory_action_probabilities, np.float32), \
               self._flat_list(self.trajectory_returns, np.float32), \
               self._flat_list(self.trajectory_advantages, np.float32), \
               self._flat_list(self.trajectory_actions_distribution, np.float32)

    def _init_episode_variables(self):
        self.episode_states = []
        self.episode_actions = []
        self.episode_action_probabilities = []
        self.episode_rewards = []
        self.episode_values = []
        self.episode_actions_distribution = []

    def _init_trajectory_variables(self):
        self.trajectory_states = []
        self.trajectory_actions = []
        self.trajectory_action_probabilities = []
        self.trajectory_returns = []
        self.trajectory_advantages = []
        self.trajectory_actions_distribution = []

    @staticmethod
    def _compute_discounted_cumulative_sum(x, discount_rate):
        return scipy.signal.lfilter([1], [1, float(-discount_rate)], x[::-1], axis=0)[::-1]
        '''
        n_elements = len(x)
        result = np.zeros(n_elements)
        last_value = 0

        for i, r in enumerate(x[::-1]):
            result[n_elements - i - 1] = r + last_value * discount_rate
            last_value = result[n_elements - i - 1]

        return result
        '''

    @staticmethod
    def _flat_list(unflattened_list, dtype):
        flatten_list = [x for xs in unflattened_list for x in xs]
        numpy_list = np.array(flatten_list, dtype=dtype)
        return numpy_list

    @staticmethod
    def discount_returns(rewards, gamma):
        """Reward discounting: Rₜ = rₜ + γRₜ₊₁"""
        discounted = []
        #for er in rewards:
        er_flip = np.flip(rewards, (0,))  # er.flip((0,))
        dr = [er_flip[0]]
        for r in er_flip[1:]:
            dr.append(r + gamma * dr[-1])
        # dr = torch.tensor(dr).flip((0,))
        dr = np.flip(dr, (0,))
        discounted.append(dr)
        return discounted

    @staticmethod
    def gae(rewards, values, gamma, _lambda):
        """Generalized Advantage Estimate"""
        advantages = []
        #for rewards, values in zip(r,v):
        r_flip = np.flip(rewards, (0,)) # episode.rewards.flip((0,))
        v_flip = np.flip(values,(0,))

        ep_adv = []
        v_prev = 0
        a_prev = 0

        for r, v in zip(r_flip, v_flip):
            delta = r - v + gamma * v_prev  # δₜ = rₜ + γV(sₜ₊₁) - V(s)
            a = delta + gamma * _lambda * a_prev  # Aₜ = δₜ + λγAₜ₊₁
            ep_adv.append(a)
            v_prev = v
            a_prev = a
        # ep_adv = torch.tensor(ep_adv).flip((0,))
        ep_adv = np.flip(ep_adv, (0,))
        advantages.append(ep_adv)
        return advantages