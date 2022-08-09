from environment.env_wrapper import RenderMode, NinjaEnvironment, LeaperEnvironment, CoinrunEnvironment

"""
AGENT SETTINGS
"""
PPO_N_AGENTS = 32
PPO_ITERATIONS = 256
PPO_AGENT_HORIZON = 1024
PPO_BATCH_SIZE = 256
PPO_EPOCHS_MODEL_UPDATE = 12
PPO_RANDOMIZE_SAMPLES = True

REINFORCE_ITERATIONS = PPO_N_AGENTS * PPO_ITERATIONS * PPO_AGENT_HORIZON  # Do the same number of iterations of PPO

AGENT_TYPE = 'ppo'  # valid values ['ppo', 'reinforce']
BACKBONE_TYPE = 'impala'  # valid values ['impala', 'nature']

SAVE_WEIGHTS = True
WEIGHTS_PATH = 'weights/'

"""
ENVIRONMENT SETTINGS
"""
ENVIRONMENT_TYPE = NinjaEnvironment  # valid values [NinjaEnvironment, LeaperEnvironment, CoinrunEnvironment]
NEGATIVE_REWARDS_FOR_LOSSES = True
RENDER_MODE = RenderMode.off
SAVE_VIDEO = False

"""
LOGS
"""
LOGS_PATH = 'logs'
SAVE_LOGS = True

print(f'N_AGENTS: {PPO_N_AGENTS}')
print(f'ITERATIONS: {PPO_ITERATIONS}')
print(f'AGENT_HORIZON: {PPO_AGENT_HORIZON}')
print(f'BATCH_SIZE: {PPO_BATCH_SIZE}')
print(f'EPOCHS_MODEL_UPDATE: {PPO_EPOCHS_MODEL_UPDATE}')
print(f'RANDOMIZE_SAMPLES: {PPO_RANDOMIZE_SAMPLES}')
print(f'AGENT_TYPE: {AGENT_TYPE}')
print(f'BACKBONE_TYPE: {BACKBONE_TYPE}')
print(f'SAVE_WEIGHTS: {SAVE_WEIGHTS}')
