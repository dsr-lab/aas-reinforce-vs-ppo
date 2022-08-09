from environment.env_wrapper import RenderMode, NinjaEnvironment, LeaperEnvironment, CoinrunEnvironment

"""
AGENT SETTINGS
"""
PPO_N_AGENTS = 1
PPO_ITERATIONS = 1
PPO_AGENT_HORIZON = 1024
PPO_BATCH_SIZE = 256
PPO_EPOCHS_MODEL_UPDATE = 12
PPO_RANDOMIZE_SAMPLES = True

REINFORCE_ITERATIONS = PPO_N_AGENTS * PPO_ITERATIONS * PPO_AGENT_HORIZON  # Do the same number of iterations of PPO

AGENT_TYPE = 'ppo'  # valid values ['ppo', 'reinforce']
BACKBONE_TYPE = 'impala'  # valid values ['impala', 'nature']

SAVE_WEIGHTS = False
WEIGHTS_PATH = 'weights/'

"""
ENVIRONMENT SETTINGS
"""
ENVIRONMENT_TYPE = LeaperEnvironment  # valid values [NinjaEnvironment, LeaperEnvironment, CoinrunEnvironment]
RENDER_MODE = RenderMode.on
SAVE_VIDEO = True

"""
LOGS
"""
LOGS_PATH = 'logs'
SAVE_LOGS = True

# print(f'N_AGENTS: {N_AGENTS}')
# print(f'ITERATIONS: {ITERATIONS}')
# print(f'AGENT_HORIZON: {AGENT_HORIZON}')
# print(f'BATCH_SIZE: {BATCH_SIZE}')
# print(f'EPOCHS_MODEL_UPDATE: {EPOCHS_MODEL_UPDATE}')
# print(f'RANDOMIZE_SAMPLES: {RANDOMIZE_SAMPLES}')
# print(f'AGENT_TYPE: {AGENT_TYPE}')
# print(f'BACKBONE_TYPE: {BACKBONE_TYPE}')
# print(f'SAVE_WEIGHTS: {SAVE_WEIGHTS}')
