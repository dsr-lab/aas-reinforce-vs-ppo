from environment.env_wrapper import RenderMode, NinjaEnvironment, LeaperEnvironment, CoinrunEnvironment

AGENT_TYPE = 'reinforce'  # valid values ['ppo', 'reinforce']
ENVIRONMENT_TYPE = NinjaEnvironment  # valid values [NinjaEnvironment, LeaperEnvironment, CoinrunEnvironment]


"""""""""""""""""""""""""""""""""""""""""
COMMON SETTINGS
"""""""""""""""""""""""""""""""""""""""""
WEIGHTS_PATH = 'weights'    # path where model weights are saved
LOGS_PATH = 'logs/'         # path where execution logs are saved

trainer_common_config = {
    'use_negative_rewards_for_losses': True,   # assign -1 for every game loss
    'backbone_type': 'impala',                  # Valid values ['impala', 'nature']
    'save_weights': True,
    'weights_path': WEIGHTS_PATH,
    'save_logs': True,
    'logs_path': LOGS_PATH
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
    'randomize_samples': True,
    'clip_ratio': 0.2,
    'normalize_advantages': False,
    'clip_value_estimates': False
}

ppo_environment_config = {
    'num': PPO_N_AGENTS,
    **environment_common_config
}

"""""""""""""""""""""""""""""""""""""""""
REINFORCE
"""""""""""""""""""""""""""""""""""""""""
reinforce_trainer_config = {
    'n_iterations': 8388608,
    'n_checkpoints': 10,
    'n_logs': 256
}

reinforce_environment_config = {
    'num': 1,
    **environment_common_config
}
