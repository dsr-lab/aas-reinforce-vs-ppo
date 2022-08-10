from environment.env_wrapper import RenderMode, NinjaEnvironment, LeaperEnvironment, CoinrunEnvironment

AGENT_TYPE = 'ppo'  # valid values ['ppo', 'reinforce']
ENVIRONMENT_TYPE = NinjaEnvironment  # valid values [NinjaEnvironment, LeaperEnvironment, CoinrunEnvironment]


"""""""""""""""""""""""""""""""""""""""""
COMMON SETTINGS
"""""""""""""""""""""""""""""""""""""""""
WEIGHTS_PATH = 'weights/'
LOGS_PATH = 'logs/'
trainer_common_config = {
    'use_negative_rewards_for_losses': False,
    'backbone_type': 'impala',  # valid values ['impala', 'nature']
    'save_weights': True,
    'weights_path': WEIGHTS_PATH,   # TODO: fix logs with os.append
    'save_logs': True,
    'logs_path': LOGS_PATH  # TODO: fix logs with os.append
}

environment_common_config = {
    'render_mode': RenderMode.off,  # valid values [RenderMode.off, RenderMode.on]
    'save_video': False
}

"""""""""""""""""""""""""""""""""""""""""
PPO
"""""""""""""""""""""""""""""""""""""""""
PPO_N_AGENTS = 32

ppo_trainer_config = {
    'n_agents': PPO_N_AGENTS,
    'n_iterations': 256,
    'agent_horizon': 1024,
    'batch_size': 256,
    'epochs_model_update': 12,
    'randomize_samples': True
}

ppo_environment_config = {
    'num': PPO_N_AGENTS,
    **environment_common_config
}

"""""""""""""""""""""""""""""""""""""""""
REINFORCE
"""""""""""""""""""""""""""""""""""""""""
reinforce_trainer_config = {
    'n_iterations': 8000000,  # TODO...
}

reinforce_environment_config = {
    'num': 1,
    **environment_common_config
}


# print(f'N_AGENTS: {PPO_N_AGENTS}')
# print(f'ITERATIONS: {PPO_ITERATIONS}')
# print(f'AGENT_HORIZON: {PPO_AGENT_HORIZON}')
# print(f'BATCH_SIZE: {PPO_BATCH_SIZE}')
# print(f'EPOCHS_MODEL_UPDATE: {PPO_EPOCHS_MODEL_UPDATE}')
# print(f'RANDOMIZE_SAMPLES: {PPO_RANDOMIZE_SAMPLES}')
# print(f'AGENT_TYPE: {AGENT_TYPE}')
# print(f'BACKBONE_TYPE: {BACKBONE_TYPE}')
# print(f'SAVE_WEIGHTS: {SAVE_WEIGHTS}')
