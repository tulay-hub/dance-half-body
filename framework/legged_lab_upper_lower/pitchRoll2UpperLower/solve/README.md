# Lens110 XML Tendon 求解器说明

本目录用于从 Lens110 的 MJCF/XML 结构中，直接计算脚踝 `pitch/roll` 和 `upper/lower` 之间的几何关系。

核心代码：

```text
projects/02_dance_half_body/framework/legged_lab_upper_lower/pitchRoll2UpperLower/solve/xml_tendon_pr_to_ul.py
```

这个求解器不使用多项式模型，也不使用 MLP 模型。它直接加载 Lens110 的 XML，通过 MuJoCo 正运动学计算 spatial tendon 长度，然后做数值求解。

## 1. 解决的问题

Lens110 脚踝 XML 里同时有：

```text
pitch/roll 被动脚踝关节
upper/lower 真实电机关节
```

它们之间不是简单线性关系，而是由 XML 里的 spatial tendon 几何约束决定。

XML 中四条 tendon 是：

```text
left upper:  lleg_Link4_1 -> lleg_Link6_a, target length = 0.192
left lower:  lleg_Link4_2 -> lleg_Link6_b, target length = 0.132
right upper: rleg_Link4_1 -> rleg_Link6_a, target length = 0.192
right lower: rleg_Link4_2 -> rleg_Link6_b, target length = 0.132
```

对应 XML 片段：

```xml
<tendon>
  <spatial range="0.192 0.192001">
    <site site="lleg_Link4_1"/>
    <site site="lleg_Link6_a"/>
  </spatial>
  <spatial range="0.132 0.132001">
    <site site="lleg_Link4_2"/>
    <site site="lleg_Link6_b"/>
  </spatial>

  <spatial range="0.192 0.192001">
    <site site="rleg_Link4_1"/>
    <site site="rleg_Link6_a"/>
  </spatial>
  <spatial range="0.132 0.132001">
    <site site="rleg_Link4_2"/>
    <site site="rleg_Link6_b"/>
  </spatial>
</tendon>
```

数学上可以写成：

```text
|lleg_Link4_1(u) - lleg_Link6_a(p,r)| = 0.192
|lleg_Link4_2(l) - lleg_Link6_b(p,r)| = 0.132
|rleg_Link4_1(u) - rleg_Link6_a(p,r)| = 0.192
|rleg_Link4_2(l) - rleg_Link6_b(p,r)| = 0.132
```

其中：

```text
p = ankle pitch
r = ankle roll
u = ankle upper
l = ankle lower
```

这个求解器支持两个方向：

```text
PR -> UL: 给定 pitch/roll，求 upper/lower
UL -> PR: 给定 upper/lower，求 pitch/roll
```

## 2. 文件结构

```text
pitchRoll2UpperLower/solve/
├── README.md
└── xml_tendon_pr_to_ul.py
```

`README.md` 是当前说明文件。

`xml_tendon_pr_to_ul.py` 同时提供：

```text
Python API
命令行工具
单点求解
batch 求解
grid 生成
npz motion 转换
奇异性/限位分析
```

## 3. 默认 XML

代码默认使用：

```text
projects/02_dance_half_body/framework/legged_lab_upper_lower/source/legged_lab/legged_lab/data/Robots/model_humanoid_lens110/mjcf/lens110.xml
```

也可以显式指定 AMP_mjlab 里的 XML：

```text
frameworks/shared/lens110_isaaclab/lens110/legged_lab_lbot/source/legged_lab/legged_lab/data/Robots/model_humanoid_lens110/mjcf/lens110.xml
```

命令行使用方式：

```bash
--xml frameworks/shared/lens110_isaaclab/lens110/legged_lab_lbot/source/legged_lab/legged_lab/data/Robots/model_humanoid_lens110/mjcf/lens110.xml
```

注意：AMP_mjlab 这份 XML 里 `meshdir="./meshes/"` 和实际目录不完全一致，代码里的 `_load_model()` 会自动修正这种 mesh 路径问题。

## 4. 重要常量

### 4.1 `PR_NAMES`

```python
PR_NAMES = (
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
)
```

表示四个 pitch/roll 脚踝关节。

顺序固定为：

```text
left_pitch, left_roll, right_pitch, right_roll
```

### 4.2 `UL_NAMES`

```python
UL_NAMES = (
    "left_ankle_upper_joint",
    "left_ankle_lower_joint",
    "right_ankle_upper_joint",
    "right_ankle_lower_joint",
)
```

表示四个 upper/lower 电机关节。

顺序固定为：

```text
left_upper, left_lower, right_upper, right_lower
```

### 4.3 `CHANNELS`

`CHANNELS` 把每条 tendon 与对应关节绑定起来。

每个通道包含：

```text
side: left/right
name: upper/lower
pitch_joint
roll_joint
solve_joint
tendon_id
```

四个通道对应：

```text
left upper  -> tendon 0
left lower  -> tendon 1
right upper -> tendon 2
right lower -> tendon 3
```

## 5. 数据结构

### 5.1 `TendonChannel`

定义位置：

```python
@dataclass(frozen=True)
class TendonChannel:
```

作用：描述一条 tendon 约束和它相关的关节。

字段：

```text
side: 左腿还是右腿，取值 left/right
name: upper 还是 lower
pitch_joint: 该侧 pitch 关节名
roll_joint: 该侧 roll 关节名
solve_joint: 正向求解时要解的 UL 关节名
tendon_id: MuJoCo data.ten_length 里的 tendon 编号
```

例子：

```python
TendonChannel(
    "left",
    "upper",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "left_ankle_upper_joint",
    0,
)
```

表示：

```text
左腿 upper tendon:
给定 left pitch/roll，求 left upper，让 tendon 0 长度等于 0.192。
```

## 6. 主类：`Lens110XmlTendonSolver`

这是整个文件最核心的类。

初始化：

```python
solver = Lens110XmlTendonSolver(xml_path)
```

主要能力：

```text
1. 加载 Lens110 XML
2. 获取 PR/UL qpos 地址
3. 获取 joint range 限位
4. 获取 tendon 目标长度
5. 正向求解 PR -> UL
6. 反向求解 UL -> PR
7. 计算 tendon length
8. 检查 tendon error
9. 分析限位和奇异性风险
```

### 6.1 `__init__`

```python
def __init__(self, xml_path=DEFAULT_XML, grid_samples=801, length_tol=1e-8)
```

作用：初始化求解器。

参数：

```text
xml_path: MJCF/XML 文件路径
grid_samples: 全局扫描时每个关节区间采样点数
length_tol: tendon 长度误差容忍度
```

内部做的事：

```python
self.model = self._load_model(self.xml_path)
self.data = mujoco.MjData(self.model)
self.qpos_addr = ...
self.joint_range = ...
self.tendon_targets = self.model.tendon_range[:4, 0]
```

其中：

```text
self.qpos_addr: joint name -> data.qpos 地址
self.joint_range: joint name -> XML 限位
self.tendon_targets: [0.192, 0.132, 0.192, 0.132]
```

### 6.2 `_load_model`

```python
@staticmethod
def _load_model(xml_path: Path) -> mujoco.MjModel
```

作用：加载 MuJoCo XML。

它先尝试：

```python
mujoco.MjModel.from_xml_path(str(xml_path))
```

如果失败，会调用 `_xml_with_resolved_meshdir()` 修正 meshdir，然后用：

```python
mujoco.MjModel.from_xml_string(fixed_xml)
```

这个函数主要是为了兼容 AMP_mjlab 那份 XML。

### 6.3 `_xml_with_resolved_meshdir`

```python
@staticmethod
def _xml_with_resolved_meshdir(xml_path: Path) -> str | None
```

作用：修正 XML 里的 `compiler meshdir`。

如果 XML 写的是：

```xml
<compiler meshdir="./meshes/"/>
```

但实际目录是：

```text
../meshes
```

这个函数会把 meshdir 改成绝对路径，然后返回修正后的 XML 字符串。

如果不需要修正，返回 `None`。

### 6.4 `_joint_id`

```python
def _joint_id(self, name: str) -> int
```

作用：通过 joint name 找 MuJoCo joint id。

内部用：

```python
mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
```

如果找不到，会抛异常。

### 6.5 `_qpos_addr`

```python
def _qpos_addr(self, name: str) -> int
```

作用：获取某个 joint 在 `data.qpos` 里的地址。

内部逻辑：

```python
joint_id = self._joint_id(name)
return self.model.jnt_qposadr[joint_id]
```

### 6.6 `_joint_range`

```python
def _joint_range(self, name: str) -> tuple[float, float]
```

作用：读取 XML 里的 joint range。

如果 XML 里 joint 有 limit，则返回：

```text
(lower, upper)
```

如果没有 limit，则返回：

```text
(-pi, pi)
```

这个范围用于求解时的边界，避免解跑到 XML 限位外。

### 6.7 `reset_qpos`

```python
def reset_qpos(self) -> None
```

作用：把内部 `MjData` 状态重置到 XML 默认姿态。

做的事：

```python
self.data.qpos[:] = self.model.qpos0
self.data.qvel[:] = 0.0
```

每次计算 tendon length 前都会重置，避免上一次求解的状态污染这一次。

### 6.8 `set_ankle_pose`

```python
def set_ankle_pose(
    left_pitch=0.0,
    left_roll=0.0,
    left_upper=0.0,
    left_lower=0.0,
    right_pitch=0.0,
    right_roll=0.0,
    right_upper=0.0,
    right_lower=0.0,
)
```

作用：设置 8 个脚踝相关 joint 的 qpos。

它只是设置角度，不会自动 `mj_forward()`。

通常后面会调用：

```python
mujoco.mj_forward(self.model, self.data)
```

### 6.9 `tendon_lengths`

```python
def tendon_lengths(...) -> np.ndarray
```

作用：给定一组 PR/UL 角度，返回当前四条 tendon 的长度。

返回顺序：

```text
[left_upper_tendon, left_lower_tendon, right_upper_tendon, right_lower_tendon]
```

内部流程：

```text
1. reset_qpos()
2. set_ankle_pose(...)
3. mujoco.mj_forward(...)
4. 返回 data.ten_length[:4]
```

这是检查几何关系最直接的函数。

### 6.10 `_channel_error`

```python
def _channel_error(self, channel, pitch, roll, motor_angle) -> float
```

作用：计算单条 tendon 的长度误差。

定义：

```text
error = 当前 tendon length - XML target length
```

例如左上：

```text
error(u; p, r) = |lleg_Link4_1(u) - lleg_Link6_a(p,r)| - 0.192
```

正向 `PR -> UL` 求解时，就是找一个 `motor_angle`，让：

```text
error = 0
```

### 6.11 `_side_tendon_error`

```python
def _side_tendon_error(self, side, pitch, roll, upper, lower) -> np.ndarray
```

作用：计算某一侧两条 tendon 的误差。

左腿返回：

```text
[
  left_upper_tendon_length - 0.192,
  left_lower_tendon_length - 0.132,
]
```

右腿返回：

```text
[
  right_upper_tendon_length - 0.192,
  right_lower_tendon_length - 0.132,
]
```

反向 `UL -> PR` 求解时使用这个函数。

因为反向是：

```text
给定 upper/lower
未知 pitch/roll
同时让两条 tendon error = 0
```

这是一个 2 元非线性方程。

### 6.12 `channel_length_derivative`

```python
def channel_length_derivative(self, channel, pitch, roll, motor_angle, eps=1e-5) -> float
```

作用：数值计算某条 tendon 长度对 motor angle 的导数。

近似：

```text
d(length) / d(motor_angle)
```

用途：判断奇异性。

如果这个值接近 0，说明电机转动几乎不改变 tendon 长度，几何映射会非常不稳定，属于接近奇异的位置。

### 6.13 `_bisect`

```python
@staticmethod
def _bisect(func, lo, hi, flo, fhi, tol, max_iter=80) -> float
```

作用：二分法求根。

要求：

```text
func(lo) 和 func(hi) 异号，或者某一端已经接近 0。
```

正向 `PR -> UL` 中，用它求单个 upper/lower 角。

### 6.14 `_golden_minimize`

```python
@staticmethod
def _golden_minimize(func, lo, hi, max_iter=80) -> float
```

作用：黄金分割搜索，最小化一个一维函数。

它是 fallback。

当全局扫描没有找到明显变号区间时，会找 tendon error 最小的点。如果最小误差足够小，也接受这个解；否则报错。

### 6.15 `solve_channel`

```python
def solve_channel(self, channel, pitch, roll, initial=None) -> float
```

作用：正向求解单条通道。

也就是：

```text
给定 pitch/roll
求 left_upper 或 left_lower 或 right_upper 或 right_lower
```

流程：

```text
1. 读取该 UL joint 的 XML range。
2. 如果有 initial，先在 initial 附近扩张搜索。
3. 如果找到 error 变号区间，用二分法求根。
4. 如果局部找不到，就在整个 joint range 上采样。
5. 找到变号区间后继续二分。
6. 如果找不到根，就找最小误差点。
7. 误差仍然太大则报错。
```

`initial` 的作用：让连续轨迹求解时不要跳到另一支解。

### 6.16 `solve_side`

```python
def solve_side(self, side, pitch, roll, initial=None) -> np.ndarray
```

作用：正向求解某一侧的两个 UL 关节。

输入：

```text
side: left/right
pitch
roll
initial: 可选，上一次的 [upper, lower]
```

输出：

```text
[upper, lower]
```

内部会调用两次 `solve_channel()`：

```text
upper tendon -> upper joint
lower tendon -> lower joint
```

### 6.17 `solve_pr_to_ul`

```python
def solve_pr_to_ul(
    left_pitch,
    left_roll,
    right_pitch,
    right_roll,
    initial=None,
) -> np.ndarray
```

作用：正向总接口。

输入顺序：

```text
left_pitch, left_roll, right_pitch, right_roll
```

输出顺序：

```text
left_upper, left_lower, right_upper, right_lower
```

这是 sim2sim 中 PR policy 转 UL 电机目标时主要使用的接口。

### 6.18 `_inverse_seed_points`

```python
def _inverse_seed_points(self, side, initial=None) -> list[np.ndarray]
```

作用：给反向求解 `UL -> PR` 准备初值。

反向是 2 元非线性方程，可能有多解，初值很重要。

这个函数会生成一组候选初值：

```text
用户传入的 initial
XML 默认 qpos0
常见站立姿态 [-0.15, 0]
当前训练站立附近 [-0.22, 0]
零姿态 [0, 0]
range 中点
range 边界附近
```

所有初值都会 clip 到 pitch/roll joint range 内。

### 6.19 `_solve_ul_side_custom`

```python
def _solve_ul_side_custom(self, side, upper, lower, seed, bounds) -> tuple[np.ndarray, float]
```

作用：反向求解的自写 fallback。

如果环境没有 SciPy，或者 SciPy 求解结果不好，会使用这个函数。

方法是阻尼 Gauss-Newton：

```text
1. 计算当前 residual = 两条 tendon error。
2. 用有限差分计算 2x2 Jacobian。
3. 解最小二乘步长。
4. 做 line search，保证误差下降。
5. 重复直到误差足够小。
```

输出：

```text
(best_pitch_roll, residual_norm)
```

### 6.20 `solve_ul_side`

```python
def solve_ul_side(self, side, upper, lower, initial=None) -> np.ndarray
```

作用：反向求解某一侧的 pitch/roll。

输入：

```text
side: left/right
upper
lower
initial: 可选，上一次的 [pitch, roll]
```

输出：

```text
[pitch, roll]
```

求解目标：

```text
side_upper_tendon_error(pitch, roll, upper, lower) = 0
side_lower_tendon_error(pitch, roll, upper, lower) = 0
```

优先使用：

```python
scipy.optimize.least_squares(...)
```

并带 bounds：

```text
pitch range
roll range
```

如果 SciPy 不可用，或误差较大，则用 `_solve_ul_side_custom()`。

### 6.21 `solve_ul_to_pr`

```python
def solve_ul_to_pr(
    left_upper,
    left_lower,
    right_upper,
    right_lower,
    initial=None,
) -> np.ndarray
```

作用：反向总接口。

输入顺序：

```text
left_upper, left_lower, right_upper, right_lower
```

输出顺序：

```text
left_pitch, left_roll, right_pitch, right_roll
```

它内部会调用：

```text
solve_ul_side("left", ...)
solve_ul_side("right", ...)
```

### 6.22 `solve_ul_to_pr_batch`

```python
def solve_ul_to_pr_batch(ul_pos, ul_vel=None, dt=None) -> tuple[np.ndarray, np.ndarray | None]
```

作用：批量反向转换。

输入：

```text
ul_pos shape = (N, 4)
顺序 = left_upper,left_lower,right_upper,right_lower
```

输出：

```text
pr_pos shape = (N, 4)
顺序 = left_pitch,left_roll,right_pitch,right_roll
```

如果传入 `ul_vel`，会用局部 Jacobian 的最小二乘反解速度：

```text
ul_vel = J * pr_vel
pr_vel = least_squares(J, ul_vel)
```

如果没有 `ul_vel`，但传了 `dt`，则对 `pr_pos` 做有限差分得到 `pr_vel`。

### 6.23 `jacobian_side`

```python
def jacobian_side(self, side, pitch, roll, eps=1e-5) -> np.ndarray
```

作用：计算某侧 `PR -> UL` 的局部雅可比矩阵。

输出矩阵形状：

```text
2 x 2
```

含义：

```text
[d_upper/d_pitch, d_upper/d_roll]
[d_lower/d_pitch, d_lower/d_roll]
```

用途：

```text
1. PR velocity -> UL velocity
2. UL velocity -> PR velocity 的最小二乘反解
3. 分析局部映射增益
```

### 6.24 `solve_pr_to_ul_batch`

```python
def solve_pr_to_ul_batch(pr_pos, pr_vel=None, dt=None) -> tuple[np.ndarray, np.ndarray | None]
```

作用：批量正向转换。

输入：

```text
pr_pos shape = (N, 4)
顺序 = left_pitch,left_roll,right_pitch,right_roll
```

输出：

```text
ul_pos shape = (N, 4)
顺序 = left_upper,left_lower,right_upper,right_lower
```

如果传入 `pr_vel`，会通过 `jacobian_side()` 计算：

```text
ul_vel = J * pr_vel
```

如果没有 `pr_vel` 但传了 `dt`，则对 `ul_pos` 做有限差分。

批量转换时会把上一帧解作为下一帧 initial，保证连续性。

### 6.25 `check_solution`

```python
def check_solution(self, pr, ul) -> np.ndarray
```

作用：检查一组 PR 和 UL 是否满足 XML tendon 长度约束。

输入：

```text
pr = [left_pitch, left_roll, right_pitch, right_roll]
ul = [left_upper, left_lower, right_upper, right_lower]
```

输出：

```text
四条 tendon 的长度误差
```

理想结果接近：

```text
[0, 0, 0, 0]
```

### 6.26 `limit_margin`

```python
def limit_margin(self, joint_name, value) -> tuple[float, float, float]
```

作用：计算某个 joint 角度离上下限还有多少余量。

返回：

```text
distance_to_low
distance_to_high
min_margin
```

用于判断是否接近 joint limit。

### 6.27 `analyze_solution`

```python
def analyze_solution(self, pr, ul) -> list[dict]
```

作用：分析一个 PR/UL 解的安全性。

每个 UL joint 返回：

```text
joint
angle
low_margin
high_margin
min_margin
dlength_dmotor
inv_gain
```

其中：

```text
dlength_dmotor: tendon 长度对 motor 角的导数
inv_gain: 1 / abs(dlength_dmotor)
```

如果 `dlength_dmotor` 很小，说明接近奇异位置。

## 7. 顶层辅助函数

### 7.1 `_finite_diff`

```python
def _finite_diff(values, dt) -> np.ndarray
```

作用：对位置序列做有限差分，得到速度。

规则：

```text
中间点: 中心差分
第一个点: 前向差分
最后一个点: 后向差分
```

### 7.2 `_as_scalar`

```python
def _as_scalar(value) -> float
```

作用：把 numpy array、list、标量统一转成 Python float。

主要用于读取 npz 里的 `fps` 等字段。

### 7.3 `_save_grid`

```python
def _save_grid(args) -> None
```

命令行 `grid` 的实现。

作用：在一段 pitch/roll 范围内采样，生成 PR -> UL 查表文件。

输出支持：

```text
.npz
.csv 或其它文本后缀
```

默认输出：

```text
pitchRoll2UpperLower/solve/lens110_xml_pr_to_ul_grid.npz
```

### 7.4 `_convert_npz_dir`

```python
def _convert_npz_dir(args) -> None
```

命令行 `convert-npz-dir` 的实现。

作用：批量读取一个目录下的 MJLab `.npz` motion，如果里面有 PR ankle joints，就用 XML tendon solver 生成或替换 UL ankle joints。

输入 motion 需要包含：

```text
dof_names
dof_pos
可选 dof_vel
可选 fps
```

转换后会写：

```text
dof_pos 中的 upper/lower
dof_vel 中的 upper/lower 速度
ankle_ul_source = lens110_xml_spatial_tendon_solve
ankle_ul_order = UL_NAMES
```

## 8. 命令行函数

### 8.1 `_cmd_solve`

命令：

```bash
python xml_tendon_pr_to_ul.py solve ...
```

作用：单点正向求解。

输入：

```text
--left-pitch
--left-roll
--right-pitch
--right-roll
```

输出：

```text
left_upper
left_lower
right_upper
right_lower
tendon_error
```

### 8.2 `_cmd_solve_inverse`

命令：

```bash
python xml_tendon_pr_to_ul.py solve-inverse ...
```

作用：单点反向求解。

输入：

```text
--left-upper
--left-lower
--right-upper
--right-lower
```

输出：

```text
left_pitch
left_roll
right_pitch
right_roll
tendon_error
```

### 8.3 `_cmd_check`

命令：

```bash
python xml_tendon_pr_to_ul.py check
```

作用：跑几组固定样本，检查正向 `PR -> UL` 是否能让 tendon error 接近 0。

### 8.4 `_cmd_check_inverse`

命令：

```bash
python xml_tendon_pr_to_ul.py check-inverse
```

作用：跑几组固定样本，检查闭环：

```text
PR -> UL -> PR
```

会打印：

```text
max_abs_pr_error
max_abs_tendon_error
```

如果这两个数都很小，说明正反解一致。

### 8.5 `_cmd_analyze`

命令：

```bash
python xml_tendon_pr_to_ul.py analyze ...
```

作用：对单个 PR pose 做分析。

输出内容：

```text
UL 解
tendon error
每个 UL joint 离上下限的余量
dlength_dmotor
inv_gain
```

用于判断：

```text
是否接近 joint limit
是否接近奇异位置
```

### 8.6 `_cmd_scan_singularity`

命令：

```bash
python xml_tendon_pr_to_ul.py scan-singularity ...
```

作用：扫描一片 pitch/roll 区域，找出：

```text
1. 哪些点 tendon 几何闭不上
2. 哪个点最接近 joint limit
3. 哪个点最接近奇异位置
```

默认扫描左右镜像姿态：

```text
left:  pitch, roll
right: pitch, -roll
```

打印：

```text
failures
WORST_LIMIT
WORST_SINGULAR
```

### 8.7 `build_arg_parser`

```python
def build_arg_parser() -> argparse.ArgumentParser
```

作用：定义所有命令行参数。

全局参数：

```text
--xml
--grid-samples
--length-tol
```

子命令：

```text
solve
solve-inverse
check
check-inverse
analyze
scan-singularity
grid
convert-npz-dir
```

### 8.8 `main`

```python
def main(argv=None) -> int
```

作用：命令行入口。

流程：

```text
1. 解析参数
2. 找到对应子命令函数
3. 执行 args.func(args)
4. 返回 0
```

## 9. 常用命令

### 9.1 正向单点 PR -> UL

```bash
python \
  projects/02_dance_half_body/framework/legged_lab_upper_lower/pitchRoll2UpperLower/solve/xml_tendon_pr_to_ul.py \
  solve --left-pitch -0.15 --left-roll 0.0 --right-pitch -0.15 --right-roll 0.0
```

典型输出：

```text
left_upper  = +0.161341435
left_lower  = +0.162273347
right_upper = +0.161341435
right_lower = +0.162273347
```

### 9.2 反向单点 UL -> PR

```bash
python \
  projects/02_dance_half_body/framework/legged_lab_upper_lower/pitchRoll2UpperLower/solve/xml_tendon_pr_to_ul.py \
  solve-inverse \
  --left-upper 0.161341435 --left-lower 0.162273347 \
  --right-upper 0.161341435 --right-lower 0.162273347
```

典型输出：

```text
left_pitch  = -0.150000084
left_roll   = +0.000000259
right_pitch = -0.150000084
right_roll  = -0.000000259
```

### 9.3 正向检查

```bash
python \
  projects/02_dance_half_body/framework/legged_lab_upper_lower/pitchRoll2UpperLower/solve/xml_tendon_pr_to_ul.py \
  check
```

### 9.4 反向闭环检查

```bash
python \
  projects/02_dance_half_body/framework/legged_lab_upper_lower/pitchRoll2UpperLower/solve/xml_tendon_pr_to_ul.py \
  check-inverse
```

### 9.5 指定 AMP_mjlab XML

```bash
python \
  projects/02_dance_half_body/framework/legged_lab_upper_lower/pitchRoll2UpperLower/solve/xml_tendon_pr_to_ul.py \
  --xml frameworks/shared/lens110_isaaclab/lens110/legged_lab_lbot/source/legged_lab/legged_lab/data/Robots/model_humanoid_lens110/mjcf/lens110.xml \
  solve --left-pitch -0.15 --left-roll 0.0 --right-pitch -0.15 --right-roll 0.0
```

### 9.6 分析限位和奇异性

```bash
python \
  projects/02_dance_half_body/framework/legged_lab_upper_lower/pitchRoll2UpperLower/solve/xml_tendon_pr_to_ul.py \
  analyze --left-pitch -0.15 --left-roll 0.0 --right-pitch -0.15 --right-roll 0.0
```

### 9.7 扫描奇异性

```bash
python \
  projects/02_dance_half_body/framework/legged_lab_upper_lower/pitchRoll2UpperLower/solve/xml_tendon_pr_to_ul.py \
  scan-singularity --pitch-min -0.8 --pitch-max 0.3 --roll-min -0.2 --roll-max 0.2
```

### 9.8 生成 PR -> UL lookup grid

```bash
python \
  projects/02_dance_half_body/framework/legged_lab_upper_lower/pitchRoll2UpperLower/solve/xml_tendon_pr_to_ul.py \
  grid --output projects/02_dance_half_body/framework/legged_lab_upper_lower/pitchRoll2UpperLower/solve/lens110_xml_pr_to_ul_grid.npz
```

### 9.9 转换 motion npz

```bash
python \
  projects/02_dance_half_body/framework/legged_lab_upper_lower/pitchRoll2UpperLower/solve/xml_tendon_pr_to_ul.py \
  convert-npz-dir \
  --input-dir /path/to/pr_npz_dir \
  --output-dir /path/to/xml_solved_ul_npz_dir \
  --float32
```

## 10. Python API 示例

### 10.1 PR -> UL

```python
from xml_tendon_pr_to_ul import Lens110XmlTendonSolver

solver = Lens110XmlTendonSolver(
    "frameworks/shared/lens110_isaaclab/lens110/legged_lab_lbot/source/legged_lab/legged_lab/data/Robots/model_humanoid_lens110/mjcf/lens110.xml"
)

ul = solver.solve_pr_to_ul(
    left_pitch=-0.15,
    left_roll=0.0,
    right_pitch=-0.15,
    right_roll=0.0,
)

print(ul)
```

输出顺序：

```text
[left_upper, left_lower, right_upper, right_lower]
```

### 10.2 UL -> PR

```python
pr = solver.solve_ul_to_pr(
    left_upper=0.161341435,
    left_lower=0.162273347,
    right_upper=0.161341435,
    right_lower=0.162273347,
)

print(pr)
```

输出顺序：

```text
[left_pitch, left_roll, right_pitch, right_roll]
```

### 10.3 检查 tendon error

```python
err = solver.check_solution(pr, ul)
print(err)
```

理想输出接近：

```text
[0, 0, 0, 0]
```

## 11. 注意事项

### 11.1 多解问题

四连杆/tendon 结构可能存在多解。

因此批量求解时建议传入上一帧结果作为 `initial`，让解保持在同一支上。

代码里的 batch 函数已经这么做了：

```text
上一帧 UL -> 下一帧 PR->UL 初值
上一帧 PR -> 下一帧 UL->PR 初值
```

### 11.2 找不到根

如果某个目标姿态在 tendon 几何上闭不上，代码会报错：

```text
No tendon-length root
No pitch/roll root
```

这通常说明：

```text
1. 输入角度超出机构可达空间
2. 接近奇异位置
3. 初值落到了错误分支
4. XML tendon 约束和输入数据不一致
```

### 11.3 反向速度

`solve_ul_to_pr_batch()` 如果传入 `ul_vel`，会用局部 Jacobian 最小二乘反推 `pr_vel`。

如果不传 `ul_vel`，但传 `dt`，会用 `pr_pos` 有限差分得到速度。

### 11.4 奇异性判断

`analyze_solution()` 里打印的：

```text
dlength_dmotor
inv_gain
```

可以用来判断奇异性。

一般来说：

```text
dlength_dmotor 越接近 0，越危险。
inv_gain 越大，映射越敏感。
```

### 11.5 这个求解器和 MLP/多项式模型的区别

MLP/多项式模型是数据拟合。

这个求解器是 XML 几何求解：

```text
设置 joint qpos
MuJoCo mj_forward
读取 data.ten_length
数值求根
```

所以它反映的是当前 XML 文件里的机构几何，而不是采集数据的统计拟合。

