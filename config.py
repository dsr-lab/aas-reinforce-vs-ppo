from environment.env_wrapper import RenderMode, NinjaEnvironment, LeaperEnvironment, CoinrunEnvironment

# ----------------------------------------
# COMMON SETTINGS
# ----------------------------------------
TRAIN = False
N_AGENTS = 32
N_EVAL_AGENTS = 1

AGENT_TYPE = 'ppo'  # valid values ['ppo', 'reinforce']
ENVIRONMENT_TYPE = LeaperEnvironment  # valid values [NinjaEnvironment, LeaperEnvironment, CoinrunEnvironment]

WEIGHTS_PATH = 'weights/ppo_leaper'  # path where model weights are saved
LOGS_PATH = 'logs'  # path where execution logs are saved

train_config = {
    'backbone_type': 'impala',  # Valid values ['impala', 'nature']
    'n_agents': N_AGENTS,
    'n_iterations': 768,
    'agent_horizon': 1024,
    'batch_size': 256,
    'epochs_model_update': 12,
    'randomize_samples': True,
    'save_logs': True,
    'save_weights': True,
    'logs_path': LOGS_PATH,
    'weights_path': WEIGHTS_PATH,
    'use_negative_rewards_for_losses':  # assign -1 for every game loss
        False if ENVIRONMENT_TYPE is LeaperEnvironment else True,
    'learning_rate': 5e-4
}

evaluation_config = {
    'backbone_type': 'impala',  # Valid values ['impala', 'nature']
    'n_agents': N_EVAL_AGENTS,
    'n_iterations': 1,
    'agent_horizon': 2048,
    'save_logs': True,
    'logs_path': LOGS_PATH,
    'weights_path': WEIGHTS_PATH,
}

# ----------------------------------------
# PPO SPECIFIC SETTINGS
# ----------------------------------------
ppo_train_config = {
    **train_config,
    'clip_ratio': 0.2,
    'entropy_bonus_coefficient': 0.01,
    'critic_loss_coefficient': 0.5,
    'clip_value_estimates': True,
    'normalize_advantages': True
}

# ----------------------------------------
# REINFORCE SPECIFIC SETTINGS
# ----------------------------------------
reinfoce_train_config = {
    **train_config,
    'with_baseline': False
}

# ----------------------------------------
# ENVIRONMENT SETTINGS
# ----------------------------------------
environment_config = {
    'num': N_AGENTS if TRAIN else N_EVAL_AGENTS,
    'render_mode': RenderMode.on,  # valid values [RenderMode.off, RenderMode.on]
    'save_video': True
}
