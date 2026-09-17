import gymnasium as gym

from . import agents


gym.register(
    id="LeggedLab-Isaac-AMP-Lens110-LegPitchRoll-v0",
    entry_point="legged_lab.envs:ManagerBasedAmpEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.lens110_leg_pitch_roll_env_cfg:Lens110LegPitchRollAmpEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Lens110LegPitchRollRslRlOnPolicyRunnerAmpCfg",
    },
)

gym.register(
    id="LeggedLab-Isaac-AMP-Lens110-LegPitchRoll-Play-v0",
    entry_point="legged_lab.envs:ManagerBasedAmpEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.lens110_leg_pitch_roll_env_cfg:Lens110LegPitchRollAmpEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Lens110LegPitchRollRslRlOnPolicyRunnerAmpCfg",
    },
)
