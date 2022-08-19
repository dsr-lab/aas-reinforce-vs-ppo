from environment.env_wrapper import RenderMode, NinjaEnvironment, LeaperEnvironment, CoinrunEnvironment

# ----------------------------------------
# COMMON SETTINGS
# ----------------------------------------
TRAIN = False  # If True, then the model is trained (i.e., weights are updated)
N_AGENTS = 32  # Number of parallel agents. Used for creating a Trajectory when training the model.
N_EVAL_AGENTS = 1  # Number of parallel agents. Used for creating a Trajectory when evaluating the model.

AGENT_TYPE = 'ppo'  # The agent to use. Valid values ['ppo', 'reinforce']
BACKBONE_TYPE = 'impala'  # The backbone/feature extractor tupe. Valid values ['impala', 'nature']
ENVIRONMENT_TYPE = LeaperEnvironment  # The environment type. Valid values ['NinjaEnvironment', 'LeaperEnvironment',
                                      # 'CoinrunEnvironment']

WEIGHTS_PATH = 'weights/ppo_leaper'  # Path where model weights are saved
LOGS_PATH = 'logs'  # Path where execution logs are saved

# ----------------------------------------
# TRAINER SETTINGS
# ----------------------------------------
# Dictionary containing all the configurations used during the training of the model.
train_config = {
    'n_agents': N_AGENTS,
    'n_iterations': 768,
    'agent_horizon': 1024,
    'batch_size': 256,
    'epochs_model_update': 12,
    'randomize_samples': True,
    'normalize_advantages': True,
    'save_logs': True,
    'save_weights': True,
    'logs_path': LOGS_PATH,
    'weights_path': WEIGHTS_PATH,
    'use_negative_rewards_for_losses':  False if ENVIRONMENT_TYPE is LeaperEnvironment else True
}

# Dictionary containing all the configurations used during the evaluation of the model
evaluation_config = {
    'n_agents': N_EVAL_AGENTS,
    'n_iterations': 1,
    'agent_horizon': 2048,
    'save_logs': True,
    'logs_path': LOGS_PATH,
    'weights_path': WEIGHTS_PATH,
}

# ----------------------------------------
# AGENT CONFIGURATION
# ----------------------------------------
# Common agent configurations
base_agent_config = {
    'backbone_type': BACKBONE_TYPE,
    'learning_rate': 5e-4,
    'n_actions': 15
}

# PPO agent specific settings
ppo_agent_config = {
    **base_agent_config,
    'clip_ratio': 0.2,
    'entropy_bonus_coefficient': 0.01,
    'critic_loss_coefficient': 0.5,
    'clip_value_estimates': True
}

# REINFORCE agent specific settings
reinfoce_agent_config = {
    **base_agent_config,
    'with_baseline': False
}

# ----------------------------------------
# ENVIRONMENT SETTINGS
# ----------------------------------------
# Dictionary containing specific Environments configurations.
environment_config = {
    'num': N_AGENTS if TRAIN else N_EVAL_AGENTS,  # Number of parallel agents
    'render_mode': RenderMode.on,  # Flag for showing/hiding the video while the agent is playing a game.
                                   # Valid values [RenderMode.off, RenderMode.on].

    'save_video': True             # If render_mode is on and save_video is True, then the video will be saved in
                                   # the recordings directory rather than seeing it on the screen.
}
