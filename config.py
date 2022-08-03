GAME_NAME = 'ninja'

# N_AGENTS = 32
# ITERATIONS = 250
# AGENT_HORIZON = 1024
# BATCH_SIZE = 64
# EPOCHS_MODEL_UPDATE = 15
# RANDOMIZE_SAMPLES = True
# AGENT_TYPE = 'ppo'  # valid values ['ppo', 'reinforce']
# BACKBONE_TYPE = 'nature'  # valid values ['impala', 'nature']
# SAVE_WEIGHTS = True
# WEIGHTS_PATH = 'weights/'

N_AGENTS = 16
ITERATIONS = 2
AGENT_HORIZON = 1000
BATCH_SIZE = 64
EPOCHS_MODEL_UPDATE = 3
RANDOMIZE_SAMPLES = True
AGENT_TYPE = 'ppo'  # valid values ['ppo', 'reinforce']
BACKBONE_TYPE = 'nature'  # valid values ['impala', 'nature']
SAVE_WEIGHTS = True
WEIGHTS_PATH = 'weights/'

print(f'GAME_NAME: {GAME_NAME}')
print(f'N_AGENTS: {N_AGENTS}')
print(f'ITERATIONS: {ITERATIONS}')
print(f'AGENT_HORIZON: {AGENT_HORIZON}')
print(f'BATCH_SIZE: {BATCH_SIZE}')
print(f'EPOCHS_MODEL_UPDATE: {EPOCHS_MODEL_UPDATE}')
print(f'RANDOMIZE_SAMPLES: {RANDOMIZE_SAMPLES}')
print(f'AGENT_TYPE: {AGENT_TYPE}')
print(f'BACKBONE_TYPE: {BACKBONE_TYPE}')
print(f'SAVE_WEIGHTS: {SAVE_WEIGHTS}')

