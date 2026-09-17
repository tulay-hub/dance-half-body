from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.terrains import MeshPlaneTerrainCfg, TerrainGeneratorCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import legged_lab.tasks.locomotion.amp.mdp as mdp
from legged_lab.data.Robots.model_humanoid_lens110.lens110_mjcf import (
    LENS110_MJCF_ACTIVE_JOINT_NAMES,
    LENS110_MJCF_CFG,
)

from .lens110_amp_env_cfg import ANIMATION_TERM_NAME, Lens110AmpEnvCfg, Lens110AmpEnvCfg_PLAY


@configclass
class Lens110MjcfUpperLowerAmpEnvCfg(Lens110AmpEnvCfg):
    """Lens110 AMP config that trains directly on the MJCF upper/lower ankle plant."""

    def __post_init__(self):
        super().__post_init__()
        self._use_local_flat_terrain()
        self.scene.robot = LENS110_MJCF_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.contact_forces.prim_path = "{ENV_REGEX_NS}/Robot/pelvis/.*"
        self.commands.base_velocity.debug_vis = False
        self._use_direct_upper_lower_io()
        self._use_mjcf_reward_terms()
        self._use_consistent_mjcf_resets()
        if self.__class__.__name__ == "Lens110MjcfUpperLowerAmpEnvCfg":
            self.disable_zero_weight_rewards()

    def _use_local_flat_terrain(self):
        self.scene.terrain.terrain_type = "generator"
        self.scene.terrain.terrain_generator = TerrainGeneratorCfg(
            size=(8.0, 8.0),
            num_rows=1,
            num_cols=1,
            use_cache=False,
            sub_terrains={"flat": MeshPlaneTerrainCfg(proportion=1.0)},
        )
        self.scene.terrain.max_init_terrain_level = 0

    def _use_direct_upper_lower_io(self):
        active_joints = list(LENS110_MJCF_ACTIVE_JOINT_NAMES)
        active_asset = SceneEntityCfg("robot", joint_names=active_joints, preserve_order=True)

        self.actions.joint_pos = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=active_joints,
            preserve_order=True,
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
        )

        self.observations.policy.joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": active_asset},
            noise=Unoise(n_min=-0.03, n_max=0.03),
        )
        self.observations.policy.joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": active_asset},
            noise=Unoise(n_min=-1.75, n_max=1.75),
        )

        self.observations.critic.joint_pos = ObsTerm(func=mdp.joint_pos, params={"asset_cfg": active_asset})
        self.observations.critic.joint_vel = ObsTerm(func=mdp.joint_vel, params={"asset_cfg": active_asset})
        self.observations.disc.joint_pos = ObsTerm(func=mdp.joint_pos, params={"asset_cfg": active_asset})
        self.observations.disc.joint_vel = ObsTerm(func=mdp.joint_vel, params={"asset_cfg": active_asset})
        self.observations.disc_demo.ref_joint_pos = ObsTerm(
            func=mdp.ref_joint_pos_upper_lower,
            params={
                "animation": ANIMATION_TERM_NAME,
                "flatten_steps_dim": False,
                "source_joint_names": mdp.LENS110_MOTION_JOINT_NAMES,
                "output_joint_names": active_joints,
                "output_joint_signs": {"right_elbow_joint": -1.0},
            },
        )
        self.observations.disc_demo.ref_joint_vel = ObsTerm(
            func=mdp.ref_joint_vel_upper_lower,
            params={
                "animation": ANIMATION_TERM_NAME,
                "flatten_steps_dim": False,
                "source_joint_names": mdp.LENS110_MOTION_JOINT_NAMES,
                "output_joint_names": active_joints,
                "output_joint_signs": {"right_elbow_joint": -1.0},
            },
        )

    def _use_mjcf_reward_terms(self):
        arm_asset = SceneEntityCfg(
            "robot",
            joint_names=[
                "left_shoulder_pitch_joint",
                "left_shoulder_roll_joint",
                "left_shoulder_yaw_joint",
                "left_elbow_joint",
                "right_shoulder_pitch_joint",
                "right_shoulder_roll_joint",
                "right_shoulder_yaw_joint",
                "right_elbow_joint",
            ],
            preserve_order=True,
        )
        hip_yaw_asset = SceneEntityCfg(
            "robot",
            joint_names=["left_hip_yaw_joint", "right_hip_yaw_joint"],
            preserve_order=True,
        )
        self.rewards.arm_symmetry.func = mdp.lens110_mjcf_arm_symmetry_l2
        self.rewards.arm_symmetry.params = {"asset_cfg": arm_asset}
        self.rewards.arm_vel_symmetry.func = mdp.lens110_mjcf_arm_vel_symmetry_l2
        self.rewards.arm_vel_symmetry.params = {"asset_cfg": arm_asset}
        self.rewards.hip_yaw_symmetry.func = mdp.lens110_mjcf_hip_yaw_symmetry_l2
        self.rewards.hip_yaw_symmetry.params = {"asset_cfg": hip_yaw_asset}

    def _use_consistent_mjcf_resets(self):
        self.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
        self.events.reset_robot_joints.params["velocity_range"] = (0.0, 0.0)
        self.terminations.base_contact.params["sensor_cfg"].body_names = [
            "pelvis",
            "torso_yaw_link",
            ".*_hip_.*_link",
            ".*_knee_link",
            ".*Link4.*",
            ".*_shoulder_.*_link",
            ".*_elbow_link",
        ]


@configclass
class Lens110MjcfUpperLowerAmpEnvCfg_PLAY(Lens110AmpEnvCfg_PLAY, Lens110MjcfUpperLowerAmpEnvCfg):
    def __post_init__(self):
        super().__post_init__()
