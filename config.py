from environment.env_wrapper import RenderMode, NinjaEnvironment, LeaperEnvironment, CoinrunEnvironment

# ----------------------------------------
# COMMON SETTINGS
# ----------------------------------------
TRAIN = False  # If True, then the model is trained (i.e., weights are updated)
N_AGENTS = 32  # Number of parallel agents. Used for creating a Trajectory when training.
N_EVAL_AGENTS = 1  # Number of parallel agents. Used for creating a Trajectory when evaluating.

AGENT_TYPE = 'ppo'  # The agent to use. Valid values ['ppo', 'reinforce']

ENVIRONMENT_TYPE = LeaperEnvironment  # The environment type.
                                      # Valid values [NinjaEnvironment, LeaperEnvironment, CoinrunEnvironment]

WEIGHTS_PATH = 'weights/ppo_leaper'  # Path where model weights are saved
LOGS_PATH = 'logs'  # Path where execution logs are saved

# Dictionary containing all the configurations used during the training of the model.
train_config = {
    'backbone_type': 'impala',  # Feature extractor/backbone type. Valid values ['impala', 'nature']
    'n_agents': N_AGENTS,  # Number of parallel agents
    'n_iterations': 768,  # Number of iterations (i.e., the number of epochs)
    'agent_horizon': 1024,  # Maximum steps that a single agent can compute
    'batch_size': 256,  # The batch size used for updating the model weights
    'epochs_model_update': 12,  # The number of times a single trajectory is reused for training the policy.
    'randomize_samples': True,  # If True, then the samples are shuffled before training.
    'save_logs': True,  # If True, the logs will be saved in the logs_path
    'save_weights': True,  # If True, the weights will be save in the weights_path
    'logs_path': LOGS_PATH,
    'weights_path': WEIGHTS_PATH,
    'use_negative_rewards_for_losses':  # assign -1 for every game loss or incomplete.
                                        # See report section 'Credit assignment problem'
        False if ENVIRONMENT_TYPE is LeaperEnvironment else True,
    'learning_rate': 5e-4   # Learning rate used during the training
}

# Dictionary containing all the configurations used during the evaluation of the model (see train_config for a complete
# description of each key/value meaning)
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
# Dictionary containing specific PPO configurations.
ppo_train_config = {
    **train_config,
    'clip_ratio': 0.2,  # The clip ratio used for limiting policy updates.
    'entropy_bonus_coefficient': 0.01,  # called c2 in the original PPO paper.
    'critic_loss_coefficient': 0.5,  # called c1 in the original PPO paper.
    'clip_value_estimates': True,   # Clip the value estimates between one different iterations.
    'normalize_advantages': True    # Normalize advantages at batch level.
}

# ----------------------------------------
# REINFORCE SPECIFIC SETTINGS
# ----------------------------------------
# Dictionary containing specific REINFORCE configurations.
reinfoce_train_config = {
    **train_config,
    'with_baseline': False  # If True, then the baseline will be considered for computing the loss.
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
