from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import legged_lab.tasks.locomotion.amp.mdp as mdp
from legged_lab.data.Robots.model_humanoid_lens110.lens110 import LENS110_CFG

from .lens110_amp_env_cfg import ANIMATION_TERM_NAME, Lens110AmpEnvCfg, Lens110AmpEnvCfg_PLAY


WEIGHTED_MLP_ANKLE_MODEL_DIR = mdp.DEFAULT_WEIGHTED_MLP_MODEL_DIR


@configclass
class Lens110MlpUpperLowerAmpEnvCfg(Lens110AmpEnvCfg):
    """Lens110 AMP config with upper/lower policy IO and weighted-MLP ankle conversion.

    The simulated URDF still has ankle_pitch/ankle_roll joints. Policy observations expose
    those slots as ankle_upper/ankle_lower through the weighted MLP state model. Policy
    ankle actions are interpreted as upper/lower targets. The action term computes
    deployment-style upper/lower PD torque, then converts that torque to the real
    URDF ankle_pitch/ankle_roll effort command.
    """

    def __post_init__(self):
        super().__post_init__()
        self._use_mlp_upper_lower_ankle_io()
        if self.__class__.__name__ == "Lens110MlpUpperLowerAmpEnvCfg":
            self.disable_zero_weight_rewards()

    def _use_mlp_upper_lower_ankle_io(self):
        self.scene.robot = LENS110_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.robot.actuators["feet"].stiffness = 0.0
        self.scene.robot.actuators["feet"].damping = 0.0

        self.actions.joint_pos = mdp.Lens110MlpUpperLowerJointTorquePdActionCfg(
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
            model_dir=WEIGHTED_MLP_ANKLE_MODEL_DIR,
            ankle_stiffness=5.0,
            ankle_damping=5.0,
            ankle_effort_limit=36.0,
            ankle_pd_space="upper_lower",
        )

        self.observations.policy.joint_pos = ObsTerm(
            func=mdp.joint_pos_upper_lower_mlp_rel,
            noise=Unoise(n_min=-0.03, n_max=0.03),
            params={"model_dir": WEIGHTED_MLP_ANKLE_MODEL_DIR},
        )
        self.observations.policy.joint_vel = ObsTerm(
            func=mdp.joint_vel_upper_lower_mlp_rel,
            noise=Unoise(n_min=-1.75, n_max=1.75),
            params={"model_dir": WEIGHTED_MLP_ANKLE_MODEL_DIR},
        )

        self.observations.critic.joint_pos = ObsTerm(
            func=mdp.joint_pos_upper_lower_mlp,
            params={"model_dir": WEIGHTED_MLP_ANKLE_MODEL_DIR},
        )
        self.observations.critic.joint_vel = ObsTerm(
            func=mdp.joint_vel_upper_lower_mlp,
            params={"model_dir": WEIGHTED_MLP_ANKLE_MODEL_DIR},
        )
        self.observations.disc.joint_pos = ObsTerm(
            func=mdp.joint_pos_upper_lower_mlp,
            params={"model_dir": WEIGHTED_MLP_ANKLE_MODEL_DIR},
        )
        self.observations.disc.joint_vel = ObsTerm(
            func=mdp.joint_vel_upper_lower_mlp,
            params={"model_dir": WEIGHTED_MLP_ANKLE_MODEL_DIR},
        )
        self.observations.disc_demo.ref_joint_pos = ObsTerm(
            func=mdp.ref_joint_pos_upper_lower_mlp,
            params={
                "animation": ANIMATION_TERM_NAME,
                "flatten_steps_dim": False,
                "model_dir": WEIGHTED_MLP_ANKLE_MODEL_DIR,
            },
        )
        self.observations.disc_demo.ref_joint_vel = ObsTerm(
            func=mdp.ref_joint_vel_upper_lower_mlp,
            params={
                "animation": ANIMATION_TERM_NAME,
                "flatten_steps_dim": False,
                "model_dir": WEIGHTED_MLP_ANKLE_MODEL_DIR,
            },
        )


@configclass
class Lens110MlpUpperLowerAmpEnvCfg_PLAY(Lens110AmpEnvCfg_PLAY, Lens110MlpUpperLowerAmpEnvCfg):
    def __post_init__(self):
        super().__post_init__()
