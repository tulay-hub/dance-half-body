import math
import os

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import legged_lab.tasks.locomotion.amp.mdp as mdp
from legged_lab import LEGGED_LAB_ROOT_DIR
from legged_lab.data.Robots.model_humanoid_lens110.lens110 import LENS110_CFG
from legged_lab.tasks.locomotion.amp.amp_env_cfg import LocomotionAmpEnvCfg


ANIMATION_TERM_NAME = "animation"
AMP_NUM_STEPS = 3
LENS110_BASE_HEIGHT = 0.66
LENS110_TOTAL_MASS_KG = 22.92451
LENS110_BASE_MASS_RANDOMIZATION_KG = round(LENS110_TOTAL_MASS_KG * 0.033, 2)
LENS110_MOTION_DATA_DIR = os.path.join(LEGGED_LAB_ROOT_DIR, "data", "MotionData", "lens110_gmr_lab")
LENS110_KEY_BODY_NAMES = [
    "left_ankle_roll_link",
    "right_ankle_roll_link",
    "left_elbow_link",
    "right_elbow_link",
    "left_shoulder_roll_link",
    "right_shoulder_roll_link",
]
LENS110_MOTION_NAMES = [
    "02_02_stageii_walk",
    "127_03_stageii_run",
    "127_04_stageii_walk2run",
    "127_06_stageii_run",
    "143_02_stageii_run2stop",
    "143_03_stageii_start2run",
    "16_34_stageii_walk2stand",
    "B10_-__Walk_turn_left_45_stageii",
    "B13_-__Walk_turn_right_90_stageii",
    "B14_-__Walk_turn_right_45_t2_stageii",
    "B15_-__Walk_turn_around_stageii",
    "B9_-__Walk_turn_left_90_stageii",
    "C12_-_run_turn_left_45_stageii",
    "C17_-_run_change_direction_stageii",
    "move_back",
    "move_l",
    "move_r",
    "turn_l",
    "turn_r",
]


@configclass
class Lens110AmpRewards:
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_exp,
        weight=0.0,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_exp,
        weight=0.0,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )

    alive = RewTerm(func=mdp.is_alive, weight=0.0)

    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=0.0)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=0.0)
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=0.0)
    base_height = RewTerm(func=mdp.base_height, weight=0.0, params={"target_height": LENS110_BASE_HEIGHT})

    joint_vel_l2 = RewTerm(func=mdp.joint_vel_l2, weight=0.0)
    joint_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=0.0)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=0.0)
    smoothness_1 = RewTerm(func=mdp.smoothness_1, weight=0.0)
    joint_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=0.0)
    joint_energy = RewTerm(func=mdp.joint_energy, weight=0.0)
    joint_regularization = RewTerm(func=mdp.joint_deviation_l1, weight=0.0)
    joint_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=0.0)

    joint_deviation_hip = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=0.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_yaw_joint", ".*_hip_roll_joint"])},
    )
    joint_deviation_arms = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=0.0,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    ".*_shoulder_.*_joint",
                    ".*_elbow_joint",
                ],
            )
        },
    )
    joint_deviation_torso = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=0.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names="torso_yaw_joint")},
    )
    arm_symmetry = RewTerm(func=mdp.lens110_arm_symmetry_l2, weight=0.0)
    arm_vel_symmetry = RewTerm(func=mdp.lens110_arm_vel_symmetry_l2, weight=0.0)
    hip_yaw_symmetry = RewTerm(func=mdp.lens110_hip_yaw_symmetry_l2, weight=0.0)

    feet_air_time = RewTerm(
        func=mdp.feet_air_time_positive_biped,
        weight=0.0,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
            "threshold": 0.0,
        },
    )
    feet_distance = RewTerm(
        func=mdp.feet_distance_y,
        weight=0.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=[".*ankle_roll.*"]), "min": 0.18, "max": 0.46},
    )
    knee_distance = RewTerm(
        func=mdp.knee_distance_y,
        weight=0.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=[".*_knee_link"]), "min": 0.14, "max": 0.40},
    )
    sound_suppression = RewTerm(
        func=mdp.sound_suppression_acc_per_foot,
        weight=0.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link")},
    )
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=0.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll_link"),
        },
    )

    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-1.0)
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=0.0,
        params={
            "threshold": 1.0,
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["(?!.*ankle.*).*"]),
        },
    )


@configclass
class Lens110AmpEnvCfg(LocomotionAmpEnvCfg):
    rewards: Lens110AmpRewards = Lens110AmpRewards()

    def __post_init__(self):
        super().__post_init__()

        self.scene.num_envs = 2048
        self.scene.robot = LENS110_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None

        self.actions.joint_pos.scale = {
            "torso_yaw_joint": 0.125,
            ".*_shoulder_.*_joint": 0.18,
            ".*_elbow_joint": 0.18,
            "^(?!torso_yaw_joint|.*_shoulder_.*_joint|.*_elbow_joint).*": 0.25,
        }

        self.motion_data.motion_dataset.motion_data_dir = LENS110_MOTION_DATA_DIR
        self.motion_data.motion_dataset.motion_data_weights = {
            motion_name: (1.0, "auto") for motion_name in LENS110_MOTION_NAMES
        }

        self.animation.animation.num_steps_to_use = AMP_NUM_STEPS
        self.observations.disc.history_length = AMP_NUM_STEPS
        self.observations.disc.base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        self.observations.disc.key_body_pos_b = ObsTerm(
            func=mdp.key_body_pos_b,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    body_names=LENS110_KEY_BODY_NAMES,
                    preserve_order=True,
                ),
            },
        )
        self.observations.disc_demo.ref_root_lin_vel_b = ObsTerm(
            func=mdp.ref_root_lin_vel_b,
            params={
                "animation": ANIMATION_TERM_NAME,
                "flatten_steps_dim": False,
            },
        )
        self.observations.disc_demo.ref_key_body_pos_b = ObsTerm(
            func=mdp.ref_key_body_pos_b,
            params={
                "animation": ANIMATION_TERM_NAME,
                "flatten_steps_dim": False,
            },
        )
        self.observations.disc_demo.ref_root_ang_vel_b.params["animation"] = ANIMATION_TERM_NAME
        self.observations.disc_demo.ref_joint_pos.params["animation"] = ANIMATION_TERM_NAME
        self.observations.disc_demo.ref_joint_vel.params["animation"] = ANIMATION_TERM_NAME

        self.events.add_base_mass.params["asset_cfg"].body_names = "pelvis"
        self.events.add_base_mass.params["mass_distribution_params"] = (
            -LENS110_BASE_MASS_RANDOMIZATION_KG,
            LENS110_BASE_MASS_RANDOMIZATION_KG,
        )
        self.events.randomize_rigid_body_com.params["asset_cfg"].body_names = ["pelvis", "torso_yaw_link"]
        self.events.base_external_force_torque.params["asset_cfg"].body_names = ["torso_yaw_link"]
        self.events.scale_link_mass.params["mass_distribution_params"] = (0.9, 1.1)

        self.rewards.track_lin_vel_xy_exp.weight = 1.25
        self.rewards.track_ang_vel_z_exp.weight = 1.25
        self.rewards.alive.weight = 0.1
        self.rewards.ang_vel_xy_l2.weight = -0.1
        self.rewards.flat_orientation_l2.weight = -1.0
        self.rewards.base_height.params["target_height"] = LENS110_BASE_HEIGHT
        self.rewards.base_height.weight = -2.0
        self.rewards.joint_vel_l2.weight = -2e-4
        self.rewards.joint_acc_l2.weight = -2.5e-7
        self.rewards.action_rate_l2.weight = -0.01
        self.rewards.joint_pos_limits.weight = -1.0
        self.rewards.joint_torques_l2.weight = -1e-5
        self.rewards.joint_regularization.weight = -2e-3  # 增强: -1e-3 -> -2e-3
        self.rewards.joint_deviation_hip.weight = -0.03  # 增强: -0.02 -> -0.03
        self.rewards.joint_deviation_arms.weight = -0.025
        self.rewards.joint_deviation_torso.weight = -0.01
        self.rewards.arm_symmetry.weight = -0.04
        self.rewards.arm_vel_symmetry.weight = -0.001
        self.rewards.hip_yaw_symmetry.weight = -0.15  # 增强对称性: -0.08 -> -0.15
        self.rewards.feet_air_time.weight = 0.6  # 增强步态节奏: 0.4 -> 0.6
        self.rewards.feet_air_time.params["threshold"] = 0.35
        self.rewards.feet_slide.weight = -0.12  # 减少滑步: -0.08 -> -0.12
        self.rewards.sound_suppression.weight = -1e-4
        self.rewards.feet_distance.weight = 0.15  # 增强步宽一致性: 0.10 -> 0.15
        self.rewards.knee_distance.weight = 0.10  # 增强步态一致性: 0.06 -> 0.10
        self.rewards.undesired_contacts.weight = -1.0
        self.rewards.undesired_contacts.params["threshold"] = 1.0
        self.rewards.undesired_contacts.params["sensor_cfg"] = SceneEntityCfg(
            "contact_forces",
            body_names=["(?!.*ankle.*).*"],
        )

        self.commands.base_velocity.ranges.lin_vel_x = (-0.6, 2.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.6, 0.6)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.2, 1.2)
        self.commands.base_velocity.ranges.zero_prob = (0.1, 0.1, 0.1)
        self.commands.base_velocity.ranges.heading = None

        self.curriculum.lin_vel_cmd_levels.params["lin_vel_x_limit"] = [-0.6, 2.0]
        self.curriculum.lin_vel_cmd_levels.params["lin_vel_y_limit"] = [-0.6, 0.6]

        self.terminations.base_contact.params["sensor_cfg"].body_names = [
            "pelvis",
            "torso_yaw_link",
            ".*_hip_.*_link",
            ".*_knee_link",
            ".*_shoulder_.*_link",
            ".*_elbow_link",
        ]
        self.terminations.base_height.params["minimum_height"] = 0.34

        if self.__class__.__name__ == "Lens110AmpEnvCfg":
            self.disable_zero_weight_rewards()


@configclass
class Lens110AmpEnvCfg_PLAY(Lens110AmpEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.num_envs = 8
        self.scene.env_spacing = 2.5
        self.episode_length_s = 40.0

        self.commands.base_velocity.ranges.lin_vel_x = (0.8, 0.8)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        self.commands.base_velocity.ranges.zero_prob = (0.0, 0.0, 0.0)
        self.commands.base_velocity.ranges.heading = None
        self.commands.base_velocity.rel_standing_envs = 0.0
        self.commands.base_velocity.rel_heading_envs = 0.0
        self.commands.base_velocity.debug_vis = False

        self.observations.policy.enable_corruption = False
        self.events.physics_material = None
        self.events.add_base_mass = None
        self.events.randomize_rigid_body_com = None
        self.events.scale_link_mass = None
        self.events.scale_actuator_gains = None
        self.events.scale_joint_parameters = None
        self.events.base_external_force_torque = None
        self.events.push_robot = None
        self.events.reset_base.params["pose_range"] = {"x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (0.0, 0.0)}
        self.events.reset_base.params["velocity_range"] = {
            "x": (0.0, 0.0),
            "y": (0.0, 0.0),
            "z": (0.0, 0.0),
            "roll": (0.0, 0.0),
            "pitch": (0.0, 0.0),
            "yaw": (0.0, 0.0),
        }
        self.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
        self.events.reset_robot_joints.params["velocity_range"] = (0.0, 0.0)
        self.curriculum.lin_vel_cmd_levels = None
