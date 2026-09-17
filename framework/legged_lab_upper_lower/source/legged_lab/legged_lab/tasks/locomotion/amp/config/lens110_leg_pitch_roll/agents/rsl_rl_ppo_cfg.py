from isaaclab.utils import configclass

from legged_lab.tasks.locomotion.amp.config.lens110.agents.rsl_rl_ppo_cfg import (
    Lens110RslRlOnPolicyRunnerAmpCfg,
)


@configclass
class Lens110LegPitchRollRslRlOnPolicyRunnerAmpCfg(Lens110RslRlOnPolicyRunnerAmpCfg):
    experiment_name = "lens110_amp_leg_pitch_roll"

    def __post_init__(self):
        self.algorithm.symmetry_cfg = None
