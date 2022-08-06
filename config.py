from environment.env_wrapper import RenderMode, NinjaEnvironment, LeaperEnvironment, CoinrunEnvironment

"""
AGENT SETTINGS
"""
N_AGENTS = 32
ITERATIONS = 256
AGENT_HORIZON = 1024
BATCH_SIZE = 256
EPOCHS_MODEL_UPDATE = 12
RANDOMIZE_SAMPLES = True
AGENT_TYPE = 'ppo'  # valid values ['ppo', 'reinforce']
BACKBONE_TYPE = 'impala'  # valid values ['impala', 'nature']

SAVE_WEIGHTS = True
WEIGHTS_PATH = 'weights/'

"""
ENVIRONMENT SETTINGS
"""
ENVIRONMENT_TYPE = LeaperEnvironment  # valid values [NinjaEnvironment, LeaperEnvironment, CoinrunEnvironment]
RENDER_MODE = RenderMode.off
SAVE_VIDEO = False

"""
LOGS
"""
LOGS_PATH = 'logs'
SAVE_LOGS = True

print(f'N_AGENTS: {N_AGENTS}')
print(f'ITERATIONS: {ITERATIONS}')
print(f'AGENT_HORIZON: {AGENT_HORIZON}')
print(f'BATCH_SIZE: {BATCH_SIZE}')
print(f'EPOCHS_MODEL_UPDATE: {EPOCHS_MODEL_UPDATE}')
print(f'RANDOMIZE_SAMPLES: {RANDOMIZE_SAMPLES}')
print(f'AGENT_TYPE: {AGENT_TYPE}')
print(f'BACKBONE_TYPE: {BACKBONE_TYPE}')
print(f'SAVE_WEIGHTS: {SAVE_WEIGHTS}')
