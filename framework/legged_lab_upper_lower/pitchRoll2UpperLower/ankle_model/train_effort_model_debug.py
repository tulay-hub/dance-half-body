import numpy as np
import joblib
import os
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline

# ============================================
# 模型加载（单例模式，避免重复加载）
# ============================================

_models_cache = {}

def load_model(model_path):
    """加载模型（带缓存）"""
    if model_path not in _models_cache:
        _models_cache[model_path] = joblib.load(model_path)
    return _models_cache[model_path]

def get_model(model_name, model_dir="models"):
    """获取模型"""
    model_path = os.path.join(model_dir, model_name)
    return load_model(model_path)


# ============================================
# 预测函数（使用训练好的模型）
# ============================================

def predict_pos_ul2pr_left(left_upper, left_lower, model_dir="models"):
    """逆向预测位置 - 左腿：2个主动关节角度 → 2个被动关节角度"""
    model = get_model("pos_ul2pr_left.pkl", model_dir)
    X = np.array([[left_upper, left_lower]])
    Y_pred = model.predict(X)[0]
    return Y_pred[0], Y_pred[1]

def predict_pos_ul2pr_right(right_upper, right_lower, model_dir="models"):
    """逆向预测位置 - 右腿：2个主动关节角度 → 2个被动关节角度"""
    model = get_model("pos_ul2pr_right.pkl", model_dir)
    X = np.array([[right_upper, right_lower]])
    Y_pred = model.predict(X)[0]
    return Y_pred[0], Y_pred[1]

def predict_pos_pr2ul_left(left_pitch, left_roll, model_dir="models"):
    """正向预测位置 - 左腿：2个被动关节角度 → 2个主动关节角度"""
    model = get_model("pos_pr2ul_left.pkl", model_dir)
    X = np.array([[left_pitch, left_roll]])
    Y_pred = model.predict(X)[0]
    return Y_pred[0], Y_pred[1]

def predict_pos_pr2ul_right(right_pitch, right_roll, model_dir="models"):
    """正向预测位置 - 右腿：2个被动关节角度 → 2个主动关节角度"""
    model = get_model("pos_pr2ul_right.pkl", model_dir)
    X = np.array([[right_pitch, right_roll]])
    Y_pred = model.predict(X)[0]
    return Y_pred[0], Y_pred[1]


# ============================================
# 雅可比矩阵计算函数
# ============================================

def get_jacobian_pr2ul(p, r, model_dir="models", side="left"):
    """计算从被动关节（pitch, roll）到主动关节（upper, lower）的雅可比矩阵"""
    if side == "left":
        model = get_model("pos_pr2ul_left.pkl", model_dir)
    else:
        model = get_model("pos_pr2ul_right.pkl", model_dir)
    
    ridge = model.named_steps['ridge']
    coef = ridge.coef_
    w_u = coef[0]
    w_l = coef[1]
    
    # 特征顺序: [p, r, p^2, pr, r^2, p^3, p^2*r, p*r^2, r^3]
    du_dp = (w_u[0] + 2*w_u[2]*p + w_u[3]*r + 3*w_u[5]*p**2 + 2*w_u[6]*p*r + w_u[7]*r**2)
    du_dr = (w_u[1] + w_u[3]*p + 2*w_u[4]*r + w_u[6]*p**2 + 2*w_u[7]*p*r + 3*w_u[8]*r**2)
    dl_dp = (w_l[0] + 2*w_l[2]*p + w_l[3]*r + 3*w_l[5]*p**2 + 2*w_l[6]*p*r + w_l[7]*r**2)
    dl_dr = (w_l[1] + w_l[3]*p + 2*w_l[4]*r + w_l[6]*p**2 + 2*w_l[7]*p*r + 3*w_l[8]*r**2)
    
    return np.array([[du_dp, du_dr], [dl_dp, dl_dr]])


# ============================================
# 核心接口：从电机状态计算等效关节增益
# ============================================

def compute_joint_gains_from_motor_state(
    u_act, l_act,           # 电机当前位置 (rad)
    u_vel, l_vel,           # 电机当前速度 (rad/s)
    kp_pitch, kd_pitch,     # Pitch 的 kp/kd (任务空间)
    kp_roll, kd_roll,       # Roll 的 kp/kd (任务空间)
    model_dir="models",
    side="left",
    return_debug=False
):
    """
    核心接口：从电机状态计算 Upper/Lower 的等效 kp/kd
    
    功能：
    1. 从电机位置推算脚踝位置 (使用逆向位置模型)
    2. 从电机速度推算脚踝速度 (使用雅可比矩阵)
    3. 将任务空间 kp/kd 映射到关节空间
    
    参数:
        u_act, l_act: Upper/Lower 电机当前位置 (rad)
        u_vel, l_vel: Upper/Lower 电机当前速度 (rad/s)
        kp_pitch, kd_pitch: Pitch 的刚度和阻尼 (任务空间)
        kp_roll, kd_roll: Roll 的刚度和阻尼 (任务空间)
        model_dir: 模型目录
        side: "left" 或 "right"
        return_debug: 是否返回调试信息
    
    返回:
        kp_u, kd_u: Upper 电机的等效刚度和阻尼
        kp_l, kd_l: Lower 电机的等效刚度和阻尼
        debug_info: 调试信息字典 (如果 return_debug=True)
    """
    
    # ========== 步骤1: 从电机位置推算脚踝位置 ==========
    if side == "left":
        p_act, r_act = predict_pos_ul2pr_left(u_act, l_act, model_dir)
    else:
        p_act, r_act = predict_pos_ul2pr_right(u_act, l_act, model_dir)
    
    # ========== 步骤2: 从电机速度推算脚踝速度 ==========
    J_pr2ul = get_jacobian_pr2ul(p_act, r_act, model_dir, side)
    J_ul2pr = np.linalg.inv(J_pr2ul)
    
    pr_vel = J_ul2pr @ np.array([u_vel, l_vel])
    p_vel, r_vel = pr_vel[0], pr_vel[1]
    
    # ========== 步骤3: 将任务空间 kp/kd 映射到关节空间 ==========
    # K_x = diag(kp_pitch, kp_roll), D_x = diag(kd_pitch, kd_roll)
    # K_q = J.T @ K_x @ J, D_q = J.T @ D_x @ J
    
    K_x = np.diag([kp_pitch, kp_roll])
    D_x = np.diag([kd_pitch, kd_roll])
    
    K_q = J_ul2pr.T @ K_x @ J_ul2pr
    D_q = J_ul2pr.T @ D_x @ J_ul2pr
    
    # 提取 Upper/Lower 的 kp/kd (对角元素)
    kp_u = K_q[0, 0]
    kp_l = K_q[1, 1]
    kd_u = D_q[0, 0]
    kd_l = D_q[1, 1]
    
    # ========== 可选: 调试信息 ==========
    if return_debug:
        debug_info = {
            'p_act': p_act,
            'r_act': r_act,
            'p_vel': p_vel,
            'r_vel': r_vel,
            'J_pr2ul': J_pr2ul,
            'J_ul2pr': J_ul2pr,
            'K_q': K_q,
            'D_q': D_q,
            'kp_u': kp_u,
            'kp_l': kp_l,
            'kd_u': kd_u,
            'kd_l': kd_l,
            'u_act': u_act,
            'l_act': l_act,
            'u_vel': u_vel,
            'l_vel': l_vel,
        }
        return kp_u, kd_u, kp_l, kd_l, debug_info
    
    return kp_u, kd_u, kp_l, kd_l


# ============================================
# 完整控制接口：计算力矩
# ============================================

def compute_torque_from_motor_state(
    u_act, l_act,           # 电机当前位置 (rad)
    u_vel, l_vel,           # 电机当前速度 (rad/s)
    p_des, r_des,           # 期望脚踝姿态 (rad)
    kp_pitch, kd_pitch,     # Pitch 的 kp/kd
    kp_roll, kd_roll,       # Roll 的 kp/kd
    model_dir="models",
    side="left",
    return_debug=False
):
    """
    完整控制接口：从电机状态计算力矩指令
    
    功能：
    1. 从电机位置推算脚踝位置
    2. 从电机速度推算脚踝速度
    3. 计算任务空间控制力
    4. 映射到关节空间力矩
    
    返回:
        tau_u, tau_l: 电机力矩指令
        debug_info: 调试信息 (如果 return_debug=True)
    """
    
    # ========== 步骤1: 推算脚踝状态 ==========
    if side == "left":
        p_act, r_act = predict_pos_ul2pr_left(u_act, l_act, model_dir)
    else:
        p_act, r_act = predict_pos_ul2pr_right(u_act, l_act, model_dir)
    
    J_pr2ul = get_jacobian_pr2ul(p_act, r_act, model_dir, side)
    J_ul2pr = np.linalg.inv(J_pr2ul)
    
    pr_vel = J_ul2pr @ np.array([u_vel, l_vel])
    p_vel, r_vel = pr_vel[0], pr_vel[1]
    
    # ========== 步骤2: 任务空间控制力 ==========
    F_pitch = kp_pitch * (p_des - p_act) - kd_pitch * p_vel
    F_roll = kp_roll * (r_des - r_act) - kd_roll * r_vel
    
    # ========== 步骤3: 映射到关节空间力矩 ==========
    tau_ul = J_pr2ul.T @ np.array([F_pitch, F_roll])
    tau_u, tau_l = tau_ul[0], tau_ul[1]
    
    if return_debug:
        debug_info = {
            'p_act': p_act,
            'r_act': r_act,
            'p_vel': p_vel,
            'r_vel': r_vel,
            'p_des': p_des,
            'r_des': r_des,
            'F_pitch': F_pitch,
            'F_roll': F_roll,
            'J_pr2ul': J_pr2ul,
            'J_ul2pr': J_ul2pr,
            'tau_u': tau_u,
            'tau_l': tau_l,
            'u_act': u_act,
            'l_act': l_act,
            'u_vel': u_vel,
            'l_vel': l_vel,
        }
        return tau_u, tau_l, debug_info
    
    return tau_u, tau_l


# ============================================
# 使用示例
# ============================================

if __name__ == "__main__":
    print("=" * 70)
    print("关节增益计算接口示例")
    print("=" * 70)
    
    # ============================================
    # 示例1：计算 Upper/Lower 的等效 kp/kd
    # ============================================
    
    print("\n【示例1】计算 Upper/Lower 的等效 kp/kd")
    print("-" * 50)
    
    # 输入：电机状态
    u_act = -0.600615   # Upper 当前位置 (rad)
    l_act = -0.109792   # Lower 当前位置 (rad)
    u_vel = -0.130831   # Upper 当前速度 (rad/s)
    l_vel = -0.042722   # Lower 当前速度 (rad/s)
    
    # 输入：任务空间 kp/kd (你在仿真中训练的值)
    kp_pitch = 100.0    # Pitch 刚度
    kd_pitch = 10.0     # Pitch 阻尼
    kp_roll = 80.0      # Roll 刚度
    kd_roll = 8.0       # Roll 阻尼
    
    # 计算等效关节增益
    kp_u, kd_u, kp_l, kd_l, debug = compute_joint_gains_from_motor_state(
        u_act, l_act, u_vel, l_vel,
        kp_pitch, kd_pitch, kp_roll, kd_roll,
        model_dir="models",
        side="left",
        return_debug=True
    )
    
    print(f"输入电机状态:")
    print(f"  Upper: pos={u_act:.6f} rad, vel={u_vel:.6f} rad/s")
    print(f"  Lower: pos={l_act:.6f} rad, vel={l_vel:.6f} rad/s")
    
    print(f"\n推算脚踝状态:")
    print(f"  Pitch: pos={debug['p_act']:.6f} rad, vel={debug['p_vel']:.6f} rad/s")
    print(f"  Roll:  pos={debug['r_act']:.6f} rad, vel={debug['r_vel']:.6f} rad/s")
    
    print(f"\n任务空间 kp/kd:")
    print(f"  Pitch: kp={kp_pitch:.2f}, kd={kd_pitch:.2f}")
    print(f"  Roll:  kp={kp_roll:.2f}, kd={kd_roll:.2f}")
    
    print(f"\n关节空间增益 (等效 kp/kd):")
    print(f"  Upper: kp={kp_u:.4f}, kd={kd_u:.4f}")
    print(f"  Lower: kp={kp_l:.4f}, kd={kd_l:.4f}")
    
    print(f"\n关节空间增益矩阵 K_q:")
    print(f"  {debug['K_q']}")
    print(f"\n关节空间阻尼矩阵 D_q:")
    print(f"  {debug['D_q']}")
    
    # ============================================
    # 示例2：完整控制 - 计算力矩
    # ============================================
    
    print("\n" + "=" * 70)
    print("【示例2】完整控制 - 计算力矩指令")
    print("=" * 70)
    
    # 期望脚踝姿态
    p_des = 0.45
    r_des = -0.35
    
    # 计算力矩
    tau_u, tau_l, debug = compute_torque_from_motor_state(
        u_act, l_act, u_vel, l_vel,
        p_des, r_des,
        kp_pitch, kd_pitch, kp_roll, kd_roll,
        model_dir="models",
        side="left",
        return_debug=True
    )
    
    print(f"期望脚踝姿态:")
    print(f"  Pitch: {p_des:.4f} rad, Roll: {r_des:.4f} rad")
    
    print(f"\n当前脚踝状态:")
    print(f"  Pitch: pos={debug['p_act']:.6f} rad, vel={debug['p_vel']:.6f} rad/s")
    print(f"  Roll:  pos={debug['r_act']:.6f} rad, vel={debug['r_vel']:.6f} rad/s")
    
    print(f"\n任务空间控制力:")
    print(f"  F_pitch = {debug['F_pitch']:.6f} N")
    print(f"  F_roll  = {debug['F_roll']:.6f} N")
    
    print(f"\n输出力矩:")
    print(f"  tau_u = {tau_u:.6f} N·m")
    print(f"  tau_l = {tau_l:.6f} N·m")
    
    # ============================================
    # 示例3：批量处理 - 多个姿态
    # ============================================
    
    print("\n" + "=" * 70)
    print("【示例3】批量处理 - 多个姿态的增益映射")
    print("=" * 70)
    
    # 模拟不同姿态下的电机状态
    test_cases = [
        {"u_act": -0.4, "l_act": -0.1, "u_vel": -0.1, "l_vel": -0.05},
        {"u_act": -0.5, "l_act": -0.1, "u_vel": -0.1, "l_vel": -0.05},
        {"u_act": -0.6, "l_act": -0.1, "u_vel": -0.1, "l_vel": -0.05},
        {"u_act": -0.7, "l_act": -0.1, "u_vel": -0.1, "l_vel": -0.05},
    ]
    
    print(f"{'u_act':>8} {'l_act':>8} | {'kp_u':>10} {'kd_u':>10} {'kp_l':>10} {'kd_l':>10}")
    print("-" * 60)
    
    for case in test_cases:
        kp_u, kd_u, kp_l, kd_l = compute_joint_gains_from_motor_state(
            case["u_act"], case["l_act"],
            case["u_vel"], case["l_vel"],
            kp_pitch, kd_pitch, kp_roll, kd_roll,
            model_dir="models",
            side="left"
        )
        print(f"{case['u_act']:8.4f} {case['l_act']:8.4f} | "
              f"{kp_u:10.4f} {kd_u:10.4f} {kp_l:10.4f} {kd_l:10.4f}")
    
    print("\n" + "=" * 70)
    print("✅ 所有示例运行完成!")