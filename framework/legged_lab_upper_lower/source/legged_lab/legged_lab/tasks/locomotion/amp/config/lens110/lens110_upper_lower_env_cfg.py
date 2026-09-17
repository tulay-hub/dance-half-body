from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import legged_lab.tasks.locomotion.amp.mdp as mdp
from legged_lab.data.Robots.model_humanoid_lens110.lens110 import LENS110_CFG

from .lens110_amp_env_cfg import ANIMATION_TERM_NAME, Lens110AmpEnvCfg, Lens110AmpEnvCfg_PLAY

UPPER_LOWER_ANKLE_MODEL_PATH = mdp.DEFAULT_ANKLE_MODEL_PATH


@configclass
class Lens110UpperLowerAmpEnvCfg(Lens110AmpEnvCfg):
    """Lens110 AMP training config with upper/lower ankle policy IO.

    The simulated URDF still owns ankle_pitch/ankle_roll joints. Policy observations expose those
    slots as ankle_upper/ankle_lower through the fitted polynomial model, and policy ankle actions
    are converted back to pitch/roll targets before being applied to the simulated robot.
    """

    def __post_init__(self):
        super().__post_init__()
        self._use_upper_lower_ankle_io()
        if self.__class__.__name__ == "Lens110UpperLowerAmpEnvCfg":
            self.disable_zero_weight_rewards()

    def _use_upper_lower_ankle_io(self):
        # Keep the training plant on the URDF pitch/roll ankle robot. Only the
        # policy-facing ankle IO is converted to upper/lower coordinates.
        self.scene.robot = LENS110_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        self.actions.joint_pos = mdp.Lens110UpperLowerJointPositionActionCfg(
            asset_name="robot",
            joint_names=[".*"],
            scale={
                "torso_yaw_joint": 0.125,
                ".*_shoulder_.*_joint": 0.18,
                ".*_elbow_joint": 0.18,
                "^(?!torso_yaw_joint|.*_shoulder_.*_joint|.*_elbow_joint).*": 0.25,
            },
            use_default_offset=True,
            clip={
                "left_ankle_upper_joint": (-0.70, 1.60),
                "left_ankle_lower_joint": (-0.55, 1.50),
                "right_ankle_upper_joint": (-0.75, 1.60),
                "right_ankle_lower_joint": (-0.60, 1.50),
            },
            model_path=UPPER_LOWER_ANKLE_MODEL_PATH,
        )

        self.observations.policy.joint_pos = ObsTerm(
            func=mdp.joint_pos_upper_lower_rel,
            noise=Unoise(n_min=-0.03, n_max=0.03),
            params={"model_path": UPPER_LOWER_ANKLE_MODEL_PATH},
        )
        self.observations.policy.joint_vel = ObsTerm(
            func=mdp.joint_vel_upper_lower_rel,
            noise=Unoise(n_min=-1.75, n_max=1.75),
            params={"model_path": UPPER_LOWER_ANKLE_MODEL_PATH},
        )

        self.observations.critic.joint_pos = ObsTerm(
            func=mdp.joint_pos_upper_lower,
            params={"model_path": UPPER_LOWER_ANKLE_MODEL_PATH},
        )
        self.observations.critic.joint_vel = ObsTerm(
            func=mdp.joint_vel_upper_lower,
            params={"model_path": UPPER_LOWER_ANKLE_MODEL_PATH},
        )
        self.observations.disc.joint_pos = ObsTerm(
            func=mdp.joint_pos_upper_lower,
            params={"model_path": UPPER_LOWER_ANKLE_MODEL_PATH},
        )
        self.observations.disc.joint_vel = ObsTerm(
            func=mdp.joint_vel_upper_lower,
            params={"model_path": UPPER_LOWER_ANKLE_MODEL_PATH},
        )
        self.observations.disc_demo.ref_joint_pos = ObsTerm(
            func=mdp.ref_joint_pos_upper_lower,
            params={
                "animation": ANIMATION_TERM_NAME,
                "flatten_steps_dim": False,
                "model_path": UPPER_LOWER_ANKLE_MODEL_PATH,
            },
        )
        self.observations.disc_demo.ref_joint_vel = ObsTerm(
            func=mdp.ref_joint_vel_upper_lower,
            params={
                "animation": ANIMATION_TERM_NAME,
                "flatten_steps_dim": False,
                "model_path": UPPER_LOWER_ANKLE_MODEL_PATH,
            },
        )


@configclass
class Lens110UpperLowerAmpEnvCfg_PLAY(Lens110AmpEnvCfg_PLAY, Lens110UpperLowerAmpEnvCfg):
    def __post_init__(self):
        super().__post_init__()
