"""Flat-ground, X2-native lower-body velocity locomotion environment."""

from __future__ import annotations

from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab_tasks.manager_based.locomotion.velocity.config.g1.flat_env_cfg import G1FlatEnvCfg
from isaaclab_tasks.manager_based.locomotion.velocity.config.g1.rough_env_cfg import G1Rewards
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import MySceneCfg, TerminationsCfg

from gear_sonic.envs.manager_env.robots.x2 import X2_ULTRA_CFG, apply_x2_pd_profile
from gear_sonic.envs.manager_env.mdp.observations import (
    x2_actuator_response_context,
    x2_actuator_tracking_state,
)

from .actions import (
    GaitTemplateLowerBodyJointPositionActionCfg,
    LowerBodyJointPositionActionCfg,
)
from .gait import gait_phase_observation
from .rewards import (
    GroundContactDwellPenalty,
    GroundContactPhasePenalty,
    GroundFilteredBipedAirTime,
    base_yaw_rate_l2,
    base_lin_vel_xy_l2,
    ground_filtered_feet_slide,
    heading_error_l2,
)


X2_LOWER_JOINTS_15 = [
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_pitch_joint",
    "waist_roll_joint",
]

# Twice the torque-normalized range used by the static foundation.  The output
# is clipped to [-1, 1], so these are also hard per-step residual-angle bounds.
X2_FOUNDATION_LOWER_ACTION_SCALE = {
    ".*_hip_pitch_joint": 0.20,
    ".*_hip_roll_joint": 0.20,
    ".*_hip_yaw_joint": 0.20,
    ".*_knee_joint": 0.20,
    ".*_ankle_pitch_joint": 0.06,
    ".*_ankle_roll_joint": 0.04,
    "waist_yaw_joint": 0.20,
    "waist_pitch_joint": 0.08,
    "waist_roll_joint": 0.08,
}

X2_GAIT_CYCLE_TIME_S = 0.80
X2_GAIT_DOUBLE_SUPPORT_FRACTION = 0.30


@configclass
class X2LowerSceneCfg(MySceneCfg):
    """Velocity scene with ground-only sensors for the two X2 feet."""

    left_foot_ground_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/left_ankle_roll_link",
        filter_prim_paths_expr=["/World/ground/terrain/GroundPlane/CollisionPlane"],
        history_length=3,
        track_air_time=True,
        force_threshold=10.0,
        debug_vis=False,
    )
    right_foot_ground_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/right_ankle_roll_link",
        filter_prim_paths_expr=["/World/ground/terrain/GroundPlane/CollisionPlane"],
        history_length=3,
        track_air_time=True,
        force_threshold=10.0,
        debug_vis=False,
    )


@configclass
class X2LowerActionsCfg:
    """Fifteen deployable lower-body actions; upper body is held internally."""

    joint_pos = LowerBodyJointPositionActionCfg(
        asset_name="robot",
        joint_names=X2_LOWER_JOINTS_15,
        scale=X2_FOUNDATION_LOWER_ACTION_SCALE,
        use_default_offset=True,
        preserve_order=True,
        clip={".*": (-1.0, 1.0)},
        hold_uncontrolled_joints=True,
    )


@configclass
class X2LowerTemplateActionsCfg:
    """Stage182 15-DOF residual actions around an X2-native gait cycle."""

    joint_pos = GaitTemplateLowerBodyJointPositionActionCfg(
        asset_name="robot",
        joint_names=X2_LOWER_JOINTS_15,
        scale=X2_FOUNDATION_LOWER_ACTION_SCALE,
        use_default_offset=True,
        preserve_order=True,
        clip={".*": (-1.0, 1.0)},
        hold_uncontrolled_joints=True,
        template_path="data/processed/x2_official_forward_gait_phase_template_15dof.npz",
        template_scale=0.20,
        command_name="base_velocity",
        command_threshold=0.1,
    )


@configclass
class X2LowerObservationsCfg:
    """Deployable actor observations plus a privileged training critic."""

    @configclass
    class PolicyCfg(ObsGroup):
        # Do not expose simulator root linear velocity to the actor: the
        # currently available X2 SDK/log contract has no reliable odometry.
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5))
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
        # The critic may use simulator-only velocity during training.  It is
        # discarded when the actor is exported.
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class X2LowerTeacherObservationsCfg:
    """Simulator-only locomotion teacher observations.

    Unlike the deployable actor, this actor can see true base linear velocity.
    It is a diagnostic/teacher contract, never a direct X2 deployment contract.
    """

    @configclass
    class PolicyCfg(ObsGroup):
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5))
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    critic: X2LowerObservationsCfg.CriticCfg = X2LowerObservationsCfg.CriticCfg()


@configclass
class X2LowerTeacherPhaseObservationsCfg:
    """Teacher observations plus a four-dimensional deployable gait clock."""

    @configclass
    class PolicyCfg(X2LowerTeacherObservationsCfg.PolicyCfg):
        gait_phase = ObsTerm(
            func=gait_phase_observation,
            params={
                "command_name": "base_velocity",
                "cycle_time_s": X2_GAIT_CYCLE_TIME_S,
                "double_support_fraction": X2_GAIT_DOUBLE_SUPPORT_FRACTION,
            },
        )

    @configclass
    class CriticCfg(X2LowerObservationsCfg.CriticCfg):
        gait_phase = ObsTerm(
            func=gait_phase_observation,
            params={
                "command_name": "base_velocity",
                "cycle_time_s": X2_GAIT_CYCLE_TIME_S,
                "double_support_fraction": X2_GAIT_DOUBLE_SUPPORT_FRACTION,
            },
        )

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class X2LowerTeacherPhaseResponseHistoryObservationsCfg:
    """Phase teacher plus causal lower-joint command/response history.

    The history and grouped response context are appended after the original
    93 values.  This preserves an exact prefix for the frozen Stage192 actor.
    All actor inputs are available from commanded targets, q/dq, the known gait
    clock and the calibrated actuator profile; no foot-force or simulator COM
    truth is exposed.
    """

    @configclass
    class PolicyCfg(X2LowerTeacherPhaseObservationsCfg.PolicyCfg):
        actuator_tracking_history = ObsTerm(
            func=x2_actuator_tracking_state,
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "action_name": "joint_pos",
                "excluded_joint_names": [],
            },
            history_length=10,
            flatten_history_dim=True,
        )
        actuator_response_context = ObsTerm(
            func=x2_actuator_response_context,
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "group_names": ["legs", "feet", "waist"],
                "delay_scale_steps": 4.0,
            },
        )

    policy: PolicyCfg = PolicyCfg()
    critic: X2LowerTeacherPhaseObservationsCfg.CriticCfg = (
        X2LowerTeacherPhaseObservationsCfg.CriticCfg()
    )


@configclass
class X2LowerRewardsCfg(G1Rewards):
    """G1/H1 velocity-task reward family with X2 joint semantics."""

    # Disabled by default.  Targeted profiles may enable this direct term when
    # the Gaussian yaw-tracking reward has saturated and gives little gradient.
    yaw_rate_l2 = RewTerm(
        func=base_yaw_rate_l2,
        weight=0.0,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    stand_lin_vel_xy_l2 = RewTerm(
        func=base_lin_vel_xy_l2,
        weight=0.0,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    heading_error_l2 = RewTerm(
        func=heading_error_l2,
        weight=0.0,
        params={"command_name": "base_velocity"},
    )
    contact_dwell = RewTerm(
        func=GroundContactDwellPenalty,
        weight=0.0,
        params={
            "command_name": "base_velocity",
            "min_dwell_s": 0.12,
            "enter_force_n": 30.0,
            "exit_force_n": 5.0,
            "left_sensor_name": "left_foot_ground_contact",
            "right_sensor_name": "right_foot_ground_contact",
        },
    )
    contact_phase = RewTerm(
        func=GroundContactPhasePenalty,
        weight=0.0,
        params={
            "command_name": "base_velocity",
            "cycle_time_s": X2_GAIT_CYCLE_TIME_S,
            "double_support_fraction": X2_GAIT_DOUBLE_SUPPORT_FRACTION,
            "transition_grace_s": 0.04,
            "enter_force_n": 30.0,
            "exit_force_n": 5.0,
            "left_sensor_name": "left_foot_ground_contact",
            "right_sensor_name": "right_foot_ground_contact",
        },
    )


@configclass
class X2LowerTerminationsCfg(TerminationsCfg):
    """Fall terms that are not corrupted by X2 internal mesh contacts."""

    bad_orientation = DoneTerm(func=mdp.bad_orientation, params={"limit_angle": 0.8})
    root_height = DoneTerm(func=mdp.root_height_below_minimum, params={"minimum_height": 0.45})


@configclass
class X2LowerVelocityFlatEnvCfg(G1FlatEnvCfg):
    """Stage172 foundation task: X2-native forward velocity control."""

    scene: X2LowerSceneCfg = X2LowerSceneCfg(num_envs=4096, env_spacing=2.5)
    actions: X2LowerActionsCfg = X2LowerActionsCfg()
    observations: X2LowerObservationsCfg = X2LowerObservationsCfg()
    rewards: X2LowerRewardsCfg = X2LowerRewardsCfg()
    terminations: X2LowerTerminationsCfg = X2LowerTerminationsCfg()

    def __post_init__(self):
        super().__post_init__()

        # G1RoughEnvCfg patches this inherited term during its own post-init,
        # so remove it only afterwards.  X2 self-collision forces would make
        # the generic all-contact sensor terminate at reset.
        self.terminations.base_contact = None

        # X2 physical substrate.  Keep the already validated mesh collision,
        # self-collision-on and high-gain ideal foundation for the first gate.
        self.scene.robot = X2_ULTRA_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.robot.spawn = self.scene.robot.spawn.replace(
            articulation_props=self.scene.robot.spawn.articulation_props.replace(enabled_self_collisions=True)
        )
        self.x2_pd_profile_report = apply_x2_pd_profile(self.scene.robot, "foundation_stiff_lower")

        # Match the already validated SONIC foundation contact contract.  The
        # generic G1 task multiplies 0.8/0.6 robot material randomization by a
        # 0.8/0.6 ground, which is not the Stage15 X2 zero-action substrate.
        self.events.physics_material = None
        self.scene.terrain.physics_material.static_friction = 1.0
        self.scene.terrain.physics_material.dynamic_friction = 1.0
        self.scene.terrain.physics_material.restitution = 0.0
        self.sim.physics_material = self.scene.terrain.physics_material

        # Exact 20 ms controller contract used by SONIC and the existing X2
        # evaluation substrate.
        self.sim.dt = 0.005
        self.decimation = 4
        self.sim.render_interval = self.decimation
        self.scene.contact_forces.update_period = self.sim.dt
        self.scene.left_foot_ground_contact.update_period = self.sim.dt
        self.scene.right_foot_ground_contact.update_period = self.sim.dt

        # Start with a deliberately small forward-only curriculum.  Lateral
        # and yaw commands are added only after the first walking gate.
        self.commands.base_velocity.heading_command = False
        self.commands.base_velocity.debug_vis = False
        self.commands.base_velocity.rel_heading_envs = 0.0
        self.commands.base_velocity.rel_standing_envs = 0.20
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 0.6)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)

        # Robot-specific reset/body contract.
        self.events.base_external_force_torque.params["asset_cfg"].body_names = ["torso_link"]
        self.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
        self.events.reset_base.params = {
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        }

        # X2 semantic selectors.  Remove the G1 finger term entirely and make
        # waist/arm penalties refer to links that actually exist on X2.
        self.rewards.joint_deviation_fingers = None
        self.rewards.joint_deviation_arms.params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=[".*_shoulder_.*_joint", ".*_elbow_joint", ".*_wrist_.*_joint", "head_.*_joint"]
        )
        self.rewards.joint_deviation_torso.params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=["waist_yaw_joint", "waist_pitch_joint", "waist_roll_joint"]
        )
        self.rewards.dof_acc_l2.params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=[".*_hip_.*_joint", ".*_knee_joint", ".*_ankle_.*_joint", "waist_.*_joint"]
        )
        self.rewards.dof_torques_l2.params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=[".*_hip_.*_joint", ".*_knee_joint", ".*_ankle_.*_joint", "waist_.*_joint"]
        )
        # X2 needs self-collision enabled for the validated standing substrate.
        # Therefore gait rewards must not infer ground contact from the broad
        # all-body sensor: ankle self-contact would create false stance.  Both
        # terms below use only forces filtered against the ground plane.
        self.rewards.feet_air_time = RewTerm(
            func=GroundFilteredBipedAirTime,
            weight=0.75,
            params={
                "command_name": "base_velocity",
                "threshold": 0.4,
                "force_threshold": 10.0,
                "left_sensor_name": "left_foot_ground_contact",
                "right_sensor_name": "right_foot_ground_contact",
            },
        )
        self.rewards.feet_slide = RewTerm(
            func=ground_filtered_feet_slide,
            weight=-0.1,
            params={
                "left_sensor_name": "left_foot_ground_contact",
                "right_sensor_name": "right_foot_ground_contact",
                "force_threshold": 10.0,
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    body_names=["left_ankle_roll_link", "right_ankle_roll_link"],
                    preserve_order=True,
                ),
            },
        )

@configclass
class X2LowerVelocityFlatEnvCfg_PLAY(X2LowerVelocityFlatEnvCfg):
    """Deterministic one/small-scene configuration for contract probes."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None
        self.events.reset_base.params = {
            "pose_range": {"x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (0.0, 0.0)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        }


@configclass
class X2LowerVelocityTeacherFlatEnvCfg(X2LowerVelocityFlatEnvCfg):
    """Simulator-only teacher task whose actor observes base linear velocity."""

    observations: X2LowerTeacherObservationsCfg = X2LowerTeacherObservationsCfg()


@configclass
class X2LowerVelocityTeacherFlatEnvCfg_PLAY(X2LowerVelocityTeacherFlatEnvCfg):
    """Deterministic playback configuration for privileged teacher checkpoints."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None
        self.events.reset_base.params = {
            "pose_range": {"x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (0.0, 0.0)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        }


@configclass
class X2LowerVelocityTeacherPhaseFlatEnvCfg(X2LowerVelocityFlatEnvCfg):
    """Simulator teacher with the deployable Stage181 gait-clock contract."""

    observations: X2LowerTeacherPhaseObservationsCfg = X2LowerTeacherPhaseObservationsCfg()


@configclass
class X2LowerVelocityTeacherPhaseTemplateFlatEnvCfg(X2LowerVelocityTeacherPhaseFlatEnvCfg):
    """Phase-conditioned teacher with fixed native gait and learned residual."""

    actions: X2LowerTemplateActionsCfg = X2LowerTemplateActionsCfg()


@configclass
class X2LowerVelocityTeacherPhaseTemplateResponseHistoryFlatEnvCfg(
    X2LowerVelocityTeacherPhaseTemplateFlatEnvCfg
):
    """Stage196 frozen-base policy with a causal actuator-history adapter."""

    observations: X2LowerTeacherPhaseResponseHistoryObservationsCfg = (
        X2LowerTeacherPhaseResponseHistoryObservationsCfg()
    )


@configclass
class X2LowerVelocityTeacherPhaseFlatEnvCfg_PLAY(X2LowerVelocityTeacherPhaseFlatEnvCfg):
    """Deterministic playback for phase-conditioned teacher checkpoints."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None
        self.events.reset_base.params = {
            "pose_range": {"x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (0.0, 0.0)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        }


@configclass
class X2LowerVelocityTeacherPhaseTemplateFlatEnvCfg_PLAY(
    X2LowerVelocityTeacherPhaseTemplateFlatEnvCfg
):
    """Deterministic playback for native-template residual checkpoints."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None
        self.events.reset_base.params = {
            "pose_range": {"x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (0.0, 0.0)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        }


@configclass
class X2LowerVelocityTeacherPhaseTemplateResponseHistoryFlatEnvCfg_PLAY(
    X2LowerVelocityTeacherPhaseTemplateResponseHistoryFlatEnvCfg
):
    """Deterministic Stage196 response-history playback configuration."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None
        self.events.reset_base.params = {
            "pose_range": {"x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (0.0, 0.0)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        }
