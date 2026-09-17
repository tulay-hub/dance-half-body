#!/usr/bin/env python3
"""Simplified MuJoCo sim2sim runner for Lens110 21-DoF ONNX policy."""

# 运行命令：
# python deploy_debug.py
# 添加imu配置

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import yaml

try:
    import mujoco
except ImportError as exc:
    raise SystemExit("Missing dependency: mujoco") from exc

try:
    import onnxruntime as ort
except ImportError as exc:
    raise SystemExit("Missing dependency: onnxruntime") from exc


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_path(value: str | Path, base_dir: Path) -> Path:
    path = Path(value).expanduser()
    return (base_dir / path if not path.is_absolute() else path).resolve()


def quat_to_rotmat_wxyz(quat: np.ndarray) -> np.ndarray:
    quat = quat / max(np.linalg.norm(quat), 1.0e-12)
    w, x, y, z = quat
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - w * z), 2.0 * (x * z + w * y)],
            [2.0 * (x * y + w * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - w * x)],
            [2.0 * (x * z - w * y), 2.0 * (y * z + w * x), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float32,
    )


def name_to_id(model: mujoco.MjModel, obj_type: mujoco.mjtObj, name: str) -> int:
    obj_id = mujoco.mj_name2id(model, obj_type, name)
    if obj_id < 0:
        raise ValueError(f"Missing {obj_type.name}: {name}")
    return obj_id


def make_mapping(source_names: list[str], target_names: list[str]) -> np.ndarray:
    target_index = {name: i for i, name in enumerate(target_names)}
    return np.array([target_index[name] for name in source_names], dtype=np.int32)


class Joints:
    """Access MuJoCo joints by name, independent of XML order."""

    def __init__(self, model: mujoco.MjModel, joint_names: list[str]):
        self.names = joint_names
        self.joint_ids = np.array(
            [name_to_id(model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in joint_names], dtype=np.int32
        )
        self.qpos_ids = model.jnt_qposadr[self.joint_ids].astype(np.int32)
        self.qvel_ids = model.jnt_dofadr[self.joint_ids].astype(np.int32)
        self.actuator_ids = self._find_actuators(model)

    def _find_actuators(self, model: mujoco.MjModel) -> np.ndarray:
        actuator_ids = np.full(len(self.joint_ids), -1, dtype=np.int32)
        for actuator_id in range(model.nu):
            joint_id = int(model.actuator_trnid[actuator_id, 0])
            matches = np.where(self.joint_ids == joint_id)[0]
            if len(matches):
                actuator_ids[matches[0]] = actuator_id
        if np.any(actuator_ids < 0):
            missing = [name for name, actuator_id in zip(self.names, actuator_ids) if actuator_id < 0]
            raise ValueError(f"Missing actuators for joints: {missing}")
        return actuator_ids

    def qpos(self, data: mujoco.MjData) -> np.ndarray:
        return data.qpos[self.qpos_ids].copy()

    def qvel(self, data: mujoco.MjData) -> np.ndarray:
        return data.qvel[self.qvel_ids].copy()

    def set_qpos(self, data: mujoco.MjData, values: np.ndarray) -> None:
        data.qpos[self.qpos_ids] = values

    def set_torque(self, data: mujoco.MjData, torques: np.ndarray) -> None:
        data.ctrl[self.actuator_ids] = torques


class IMU:
    """Access MuJoCo IMU sensor data from separate sensors."""
    
    def __init__(self, model: mujoco.MjModel, sensor_config: dict):
        """
        初始化IMU传感器
        
        Args:
            model: MuJoCo模型
            sensor_config: 传感器配置字典，包含各传感器名称
        """
        # 获取各个传感器的ID和地址
        self.sensors = {}
        
        # 获取方向传感器 (四元数)
        if 'orientation' in sensor_config:
            name = sensor_config['orientation']
            sensor_id = name_to_id(model, mujoco.mjtObj.mjOBJ_SENSOR, name)
            self.sensors['orientation'] = {
                'id': sensor_id,
                'adr': model.sensor_adr[sensor_id],
                'dim': model.sensor_dim[sensor_id]
            }
            print(f"Orientation sensor: {name}, dim={model.sensor_dim[sensor_id]}")
        
        # 获取陀螺仪 (角速度)
        if 'gyro' in sensor_config:
            name = sensor_config['gyro']
            sensor_id = name_to_id(model, mujoco.mjtObj.mjOBJ_SENSOR, name)
            self.sensors['gyro'] = {
                'id': sensor_id,
                'adr': model.sensor_adr[sensor_id],
                'dim': model.sensor_dim[sensor_id]
            }
            print(f"Gyro sensor: {name}, dim={model.sensor_dim[sensor_id]}")
        
        # 获取加速度计
        if 'accelerometer' in sensor_config:
            name = sensor_config['accelerometer']
            sensor_id = name_to_id(model, mujoco.mjtObj.mjOBJ_SENSOR, name)
            self.sensors['accelerometer'] = {
                'id': sensor_id,
                'adr': model.sensor_adr[sensor_id],
                'dim': model.sensor_dim[sensor_id]
            }
            print(f"Accelerometer sensor: {name}, dim={model.sensor_dim[sensor_id]}")
        
        # 获取速度计 (线速度)
        if 'velocimeter' in sensor_config:
            name = sensor_config['velocimeter']
            sensor_id = name_to_id(model, mujoco.mjtObj.mjOBJ_SENSOR, name)
            self.sensors['velocimeter'] = {
                'id': sensor_id,
                'adr': model.sensor_adr[sensor_id],
                'dim': model.sensor_dim[sensor_id]
            }
            print(f"Velocimeter sensor: {name}, dim={model.sensor_dim[sensor_id]}")
        
        # 检查必要的传感器
        required = ['gyro', 'accelerometer']
        for req in required:
            if req not in self.sensors:
                raise ValueError(f"Required sensor '{req}' not found in sensor config")
    
    def get_gyro(self, data: mujoco.MjData) -> np.ndarray:
        """获取角速度 (在传感器局部坐标系中)"""
        if 'gyro' not in self.sensors:
            return np.zeros(3, dtype=np.float32)
        sensor = self.sensors['gyro']
        return data.sensordata[sensor['adr']:sensor['adr'] + 3].copy()
    
    def get_accelerometer(self, data: mujoco.MjData) -> np.ndarray:
        """获取加速度 (在传感器局部坐标系中)"""
        if 'accelerometer' not in self.sensors:
            return np.zeros(3, dtype=np.float32)
        sensor = self.sensors['accelerometer']
        return data.sensordata[sensor['adr']:sensor['adr'] + 3].copy()
    
    def get_velocimeter(self, data: mujoco.MjData) -> np.ndarray:
        """获取线速度 (在传感器局部坐标系中)"""
        if 'velocimeter' in self.sensors:
            sensor = self.sensors['velocimeter']
            return data.sensordata[sensor['adr']:sensor['adr'] + 3].copy()
        else:
            # 如果没有速度计，从仿真状态获取
            _, root_qvel = freejoint_addresses(data.model)
            return data.qvel[root_qvel:root_qvel + 3].copy()
    
    def get_quat(self, data: mujoco.MjData) -> np.ndarray:
        """获取姿态四元数 (w, x, y, z)"""
        if 'orientation' in self.sensors:
            sensor = self.sensors['orientation']
            quat = data.sensordata[sensor['adr']:sensor['adr'] + 4].copy()
            # 确保四元数是wxyz格式
            if len(quat) == 4:
                return quat
        # 如果orientation传感器不可用，从仿真状态获取
        root_qpos, _ = freejoint_addresses(data.model)
        return data.qpos[root_qpos + 3:root_qpos + 7].copy()


def freejoint_addresses(model: mujoco.MjModel) -> tuple[int, int]:
    for joint_id in range(model.njnt):
        if model.jnt_type[joint_id] == mujoco.mjtJoint.mjJNT_FREE:
            return int(model.jnt_qposadr[joint_id]), int(model.jnt_dofadr[joint_id])
    raise ValueError("Model has no freejoint root.")


def reset_robot(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    joints: Joints,
    base_pos: np.ndarray,
    default_qpos: np.ndarray,
) -> None:
    root_qpos, root_qvel = freejoint_addresses(model)
    data.qpos[root_qpos : root_qpos + 3] = base_pos
    data.qpos[root_qpos + 3 : root_qpos + 7] = np.array([1.0, 0.0, 0.0, 0.0])
    data.qvel[root_qvel : root_qvel + 6] = 0.0
    joints.set_qpos(data, default_qpos)
    data.qvel[:] = 0.0
    data.ctrl[:] = 0.0
    mujoco.mj_forward(model, data)


def root_state(model: mujoco.MjModel, data: mujoco.MjData) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    root_qpos, root_qvel = freejoint_addresses(model)
    quat = data.qpos[root_qpos + 3 : root_qpos + 7].copy()
    lin_vel_w = data.qvel[root_qvel : root_qvel + 3].copy()
    ang_vel_w = data.qvel[root_qvel + 3 : root_qvel + 6].copy()
    return lin_vel_w, ang_vel_w, quat


def build_observation(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    joints: Joints,
    imu: IMU,
    policy_to_model: np.ndarray,
    default_qpos_model: np.ndarray,
    last_action_policy: np.ndarray,
    command: np.ndarray,
    scales: dict,
) -> np.ndarray:
    # 从IMU获取数据
    # 1. 角速度 (在IMU局部坐标系中)
    ang_vel_b = imu.get_gyro(data)
    # print(f"ang_vel_b: {ang_vel_b}")
    
    # 2. 加速度 (在IMU局部坐标系中)
    accel_b = imu.get_accelerometer(data)
    
    # 3. 获取线速度 - 优先使用速度计
    lin_vel_b = imu.get_velocimeter(data)
    # print(f"Velocimeter: {lin_vel_b}")
    
    # 4. 获取姿态四元数
    imu_quat = imu.get_quat(data)
    
    # 使用四元数计算重力向量在IMU局部坐标系中
    imu_rot = quat_to_rotmat_wxyz(imu_quat)
    gravity_b = imu_rot.T @ np.array([0.0, 0.0, -1.0], dtype=np.float32)
    # print(f"gravity_b: {gravity_b}")
    # 关节位置和速度
    qpos_policy = (joints.qpos(data) - default_qpos_model)[policy_to_model]
    qvel_policy = joints.qvel(data)[policy_to_model]

    obs = np.concatenate(
        [
            # lin_vel_b * scales["lin_vel"], 
            ang_vel_b * scales["ang_vel"],
            gravity_b,
            command * scales["cmd"],
            qpos_policy * scales["dof_pos"],
            qvel_policy * scales["dof_vel"],
            last_action_policy,
        ]
    )
    return obs.astype(np.float32)


def pd_torque(qpos: np.ndarray, qvel: np.ndarray, target: np.ndarray, kp: np.ndarray, kd: np.ndarray, limit: np.ndarray):
    return np.clip(kp * (target - qpos) - kd * qvel, -limit, limit)


def launch_viewer(model: mujoco.MjModel, data: mujoco.MjData):
    from mujoco import viewer

    handle = viewer.launch_passive(model, data)
    handle.cam.distance = 3.5
    handle.cam.azimuth = 90
    handle.cam.elevation = -18
    return handle


def main() -> None:
    # 加载配置
    config_path = Path("configs/lens_110_sim.yaml").resolve()
    with open(config_path, "r") as f:
        cfg = load_yaml(config_path)

    # 读取路径配置
    policy_path = resolve_path(cfg["policy_path"], config_path.parent)
    xml_path = resolve_path(cfg["xml_path"], config_path.parent)
    
    # 验证文件存在
    if not policy_path.exists():
        raise FileNotFoundError(f"Policy file not found: {policy_path}")
    if not xml_path.exists():
        raise FileNotFoundError(f"XML file not found: {xml_path}")
    
    # 加载 MJCF 模型
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    model.opt.timestep = cfg["simulation_dt"]

    # 初始化IMU传感器 - 使用XML中的传感器名称
    sensor_config = {
        'orientation': cfg.get('imu_orientation_sensor', 'orientation'),
        'gyro': cfg.get('imu_gyro_sensor', 'angular-velocity'),
        'accelerometer': cfg.get('imu_accelerometer_sensor', 'linear-acceleration'),
        'velocimeter': cfg.get('imu_velocimeter_sensor', 'linear-velocity'),
    }
    imu = IMU(model, sensor_config)

    # 关节映射
    model_joint_names = cfg["model_joint_names"]
    policy_joint_names = cfg["policy_joint_names"]
    model_to_policy = make_mapping(model_joint_names, policy_joint_names)
    policy_to_model = make_mapping(policy_joint_names, model_joint_names)
    joints = Joints(model, model_joint_names)

    # 控制参数
    num_actions = len(policy_joint_names)
    default_qpos = np.asarray(cfg["default_angles"], dtype=np.float32)
    kp = np.asarray(cfg["walk_kps"], dtype=np.float32)
    kd = np.asarray(cfg["walk_kds"], dtype=np.float32)
    torque_limit = np.asarray(cfg["torque_limits"], dtype=np.float32)
    action_scale = cfg["action_scale"]

    decimation = cfg["control_decimation"]
    control_dt = model.opt.timestep * decimation
    duration = cfg["simulation_duration"]
    base_pos = np.asarray(cfg["init_base_pos"], dtype=np.float32)
    command = np.asarray(cfg["cmd_init"], dtype=np.float32)
    
    scales = {
        "lin_vel": cfg["lin_vel_scale"],
        "ang_vel": cfg["ang_vel_scale"],
        "dof_pos": cfg["dof_pos_scale"],
        "dof_vel": cfg["dof_vel_scale"],
        "cmd": np.asarray(cfg["cmd_scale"], dtype=np.float32),
    }

    # 加载 ONNX 策略
    session = ort.InferenceSession(str(policy_path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name

    # 启动可视化
    viewer = launch_viewer(model, data)

    print(f"Policy: {policy_path}")
    print(f"XML Model: {xml_path}")
    print(f"Control: sim_dt={model.opt.timestep:g}s, decimation={decimation}, policy_rate={1.0 / control_dt:.1f}Hz")
    print(f"Command: vx={command[0]:.3f}, vy={command[1]:.3f}, wz={command[2]:.3f}")

    # 重置机器人
    reset_robot(model, data, joints, base_pos, default_qpos)

    # 主循环
    policy_start = data.time    
    wall_start = time.time()
    last_action_policy = np.zeros(num_actions, dtype=np.float32)
    target_qpos = default_qpos.copy()
    step = 0
    control_step = 0
    fall_reported = False

    try:
        while data.time - policy_start < duration:
            tick = time.time()
            policy_time = data.time - policy_start

            if step % decimation == 0:
                obs = build_observation(
                    model, data, joints, imu, policy_to_model, default_qpos, 
                    last_action_policy, command, scales
                )
                raw_action = session.run(None, {input_name: obs.reshape(1, -1)})[0][0].astype(np.float32)
                last_action_policy = raw_action
                # print(last_action_policy[0])
                target_qpos = default_qpos + action_scale * raw_action[model_to_policy]
                control_step += 1
                # print(target_qpos[3])

            # 应用 PD 控制
            torque = pd_torque(joints.qpos(data), joints.qvel(data), target_qpos, kp, kd, torque_limit)
            # print(torque)
            joints.set_torque(data, torque)
            mujoco.mj_step(model, data)

            # 更新可视化
            if viewer.is_running():
                viewer.sync()
            else:
                break

            # 打印状态 (包含IMU数据)
            if step % max(1, int(0.5 / model.opt.timestep)) == 0:
                lin_vel_w, _, quat = root_state(model, data)
                vx_body = float((quat_to_rotmat_wxyz(quat).T @ lin_vel_w)[0])
                
                # 获取IMU数据用于打印
                imu_gyro = imu.get_gyro(data)
                imu_accel = imu.get_accelerometer(data)
                imu_vel = imu.get_velocimeter(data)
                
                print(
                    f"t={policy_time:6.2f}s ctrl={control_step:5d} "
                    f"x={data.qpos[0]:.3f} y={data.qpos[1]:.3f} vx_b={vx_body:.3f} "
                    f"height={data.qpos[2]:.3f} action_norm={np.linalg.norm(last_action_policy):.3f} "
                    f"gyro_x={imu_gyro[0]:.2f} accel_z={imu_accel[2]:.2f} vel_x={imu_vel[0]:.2f}",
                    flush=True,
                )

            # 摔倒检测
            if data.qpos[2] < cfg["fall_height"] and not fall_reported:
                print(f"\nFall detected: base height={data.qpos[2]:.3f}m")
                fall_reported = True

            # 实时同步
            sleep_time = model.opt.timestep - (time.time() - tick)
            if sleep_time > 0.0:
                time.sleep(sleep_time)
                # print(f"Sleeping for {sleep_time:.3f}s")
            else:
                print(f"---------------------------------------------------")
            
            step += 1
            
    except KeyboardInterrupt:
        print("\nInterrupted.")

    print(
        f"\nFinished: policy_time={data.time - policy_start:.2f}s, "
        f"steps={step}, control_steps={control_step}, wall_time={time.time() - wall_start:.2f}s"
    )


if __name__ == "__main__":
    main()