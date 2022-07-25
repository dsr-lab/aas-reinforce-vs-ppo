"""

List of hyper-parameters.
The values have been taken from the openAI documentation and the original PPO paper:
 - https://spinningup.openai.com/en/latest/algorithms/ppo.html
 - https://arxiv.org/abs/1707.06347

"""

# Actor-Critc
ACTOR_HIDDEN_SIZES = [64, 64]
CRITIC_HIDDEN_SIZES = [64, 64]
CRITIC_N_HIDDEN_LAYERS = 2

# Training
EPOCHS = 10
STEPS_PER_EPOCH = 500  # 4000
TRAIN_POLICY_ITERATIONS = 80
TRAIN_STATE_VALUE_ITERATIONS = 80

POLICY_LR = 3e-4
VALUE_FUNCTION_LR = 1e-3


OBSERVATON_SHAPE = (4,)
N_ACTIONS = 2
