import numpy as np


class Episode:
    """
    Class for managing single episodes that are collected while creating the trajectory.
    """
    def __init__(self,
                 gam=0.99,
                 lam=0.95):

        self.states = None
        self.rewards = None
        self.true_rewards = None
        self.actions = None

        self.values = None
        self.action_probabilities = None

        self.returns = None
        self.advantages = None

        self.gam = gam
        self.lam = lam

    def n_steps(self):
        """
        Number of time step of the episode

        Returns
        ----------
        counter: int
            The number of time steps
        """
        return len(self.actions) if self.actions is not None else 0

    def compute_returns(self, normalize=False):
        self.returns = self._compute_discounted_cumulative_sum(self.true_rewards, self.gam)

        if normalize:
            returns_mean, returns_std = (
                np.mean(self.returns),
                np.std(self.returns),
            )

            self.returns = (self.returns - returns_mean) / (returns_std + 1e-8)

    def compute_advantages(self, normalize=False, v_t_next=0):
        """
           Generalized Advantage Estimate (GAE): https://arxiv.org/abs/1506.02438

           Rather than using directly the expected return for computing the policy step, it is typically better to use
           an advantage function, which goal is to reduce the variance.

               Aₜ(s,a) = Eπ[Gₜ | Sₜ=s, Aₜ=a] - V(s) = δₜ

               where:

                   Eπ[Gₜ | Sₜ=s, Aₜ=a] = r₀ + γ * v(s')

           Typically, more time steps are considered, especially for reducing the bias. However, adding time steps
           could increase the variance.
           In order to avoid this, the GAE paper demonstrated that doing a discounted sum of the advantages A(s,a)
           provides the best results, with a lower variance.

           Thus:
               Aₜ(s,a) = Ʃ(γλ)ᵗ δₜ₊₁(s,a)
           """
        self.advantages = np.zeros((len(self.true_rewards)), dtype=np.float32)
        adv_t_prev = 0

        for t in range(len(self.advantages)-1, -1, -1):

            r_t = self.true_rewards[t]
            v_t = self.values[t]

            delta = r_t + self.gam * v_t_next - v_t
            adv_t = delta + self.gam * self.lam * adv_t_prev

            adv_t_prev = adv_t
            v_t_next = v_t

            self.advantages[len(self.advantages) - t - 1] = adv_t

        self.advantages = np.flip(self.advantages, axis=0)

        if normalize:
            advantages_mean, advantages_std = (
                np.mean(self.advantages),
                np.std(self.advantages),
            )

            self.advantages = (self.advantages - advantages_mean) / (advantages_std + 1e-8)

    @staticmethod
    def _compute_discounted_cumulative_sum(x, discount_rate):

        n_elements = len(x)
        result = np.zeros(n_elements, dtype=np.float32)
        last_value = 0

        for i, r in enumerate(x[::-1]):
            result[n_elements - i - 1] = r + last_value * discount_rate
            last_value = result[n_elements - i - 1]

        return result




