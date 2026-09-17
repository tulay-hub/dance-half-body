from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import legged_lab.tasks.locomotion.amp.mdp as mdp
from legged_lab.data.Robots.model_humanoid_lens110.lens110 import LENS110_CFG

from .lens110_amp_env_cfg import ANIMATION_TERM_NAME, Lens110AmpEnvCfg, Lens110AmpEnvCfg_PLAY


XML_TENDON_SOLVER_PATH = mdp.DEFAULT_XML_TENDON_SOLVER_PATH
XML_TENDON_XML_PATH = mdp.DEFAULT_XML_TENDON_XML_PATH
XML_TENDON_MLP_MODEL_DIR = mdp.DEFAULT_XML_TENDON_MLP_MODEL_DIR


@configclass
class Lens110XmlTendonUpperLowerAmpEnvCfg(Lens110AmpEnvCfg):
    """Lens110 AMP config with upper/lower policy IO from the MJCF tendon geometry solver.

    The simulated plant remains the URDF pitch/roll ankle robot. Policy observations expose
    ankle slots as upper/lower joints using pitchRoll2UpperLower/solve/xml_tendon_pr_to_ul.py,
    and policy ankle actions are upper/lower targets that are solved back to pitch/roll targets.
    """

    def __post_init__(self):
        super().__post_init__()
        self._use_xml_tendon_upper_lower_ankle_io()
        if self.__class__.__name__ == "Lens110XmlTendonUpperLowerAmpEnvCfg":
            self.disable_zero_weight_rewards()

    def _xml_tendon_params(self) -> dict:
        return {
            "model_dir": XML_TENDON_MLP_MODEL_DIR,
            "solver_path": XML_TENDON_SOLVER_PATH,
            "xml_path": XML_TENDON_XML_PATH,
        }

    def _use_xml_tendon_upper_lower_ankle_io(self):
        self.scene.robot = LENS110_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        self.actions.joint_pos = mdp.Lens110XmlTendonUpperLowerJointPositionActionCfg(
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
                "left_ankle_upper_joint": (-0.36, 0.73),
                "left_ankle_lower_joint": (-0.36, 0.73),
                "right_ankle_upper_joint": (-0.36, 0.73),
                "right_ankle_lower_joint": (-0.36, 0.73),
            },
            model_dir=XML_TENDON_MLP_MODEL_DIR,
            solver_path=XML_TENDON_SOLVER_PATH,
            xml_path=XML_TENDON_XML_PATH,
        )

        tendon_params = self._xml_tendon_params()
        self.observations.policy.joint_pos = ObsTerm(
            func=mdp.joint_pos_xml_tendon_upper_lower_rel,
            noise=Unoise(n_min=-0.03, n_max=0.03),
            params=tendon_params,
        )
        self.observations.policy.joint_vel = ObsTerm(
            func=mdp.joint_vel_xml_tendon_upper_lower_rel,
            noise=Unoise(n_min=-1.75, n_max=1.75),
            params=tendon_params,
        )

        self.observations.critic.joint_pos = ObsTerm(
            func=mdp.joint_pos_xml_tendon_upper_lower,
            params=tendon_params,
        )
        self.observations.critic.joint_vel = ObsTerm(
            func=mdp.joint_vel_xml_tendon_upper_lower,
            params=tendon_params,
        )
        self.observations.disc.joint_pos = ObsTerm(
            func=mdp.joint_pos_xml_tendon_upper_lower,
            params=tendon_params,
        )
        self.observations.disc.joint_vel = ObsTerm(
            func=mdp.joint_vel_xml_tendon_upper_lower,
            params=tendon_params,
        )
        self.observations.disc_demo.ref_joint_pos = ObsTerm(
            func=mdp.ref_joint_pos_xml_tendon_upper_lower,
            params={
                "animation": ANIMATION_TERM_NAME,
                "flatten_steps_dim": False,
                **tendon_params,
            },
        )
        self.observations.disc_demo.ref_joint_vel = ObsTerm(
            func=mdp.ref_joint_vel_xml_tendon_upper_lower,
            params={
                "animation": ANIMATION_TERM_NAME,
                "flatten_steps_dim": False,
                **tendon_params,
            },
        )


@configclass
class Lens110XmlTendonUpperLowerAmpEnvCfg_PLAY(Lens110AmpEnvCfg_PLAY, Lens110XmlTendonUpperLowerAmpEnvCfg):
    def __post_init__(self):
        super().__post_init__()
