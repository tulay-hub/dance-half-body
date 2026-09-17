import numpy as np
import joblib
import os

# ============================================
# 全局变量（预加载的pkl模型）
# ============================================
pkl_models = {}
pkl_coefs = {}
pkl_is_initialized = False


# ============================================
# 初始化接口（程序启动时调用一次）
# ============================================
def init_model_interface(model_dir="models", sides=None):
    """预加载所有模型，程序启动时调用一次"""
    global pkl_models, pkl_coefs, pkl_is_initialized
    
    if sides is None:
        sides = ["left", "right"]
    
    for side in sides:
        # 逆向模型: upper/lower → pitch/roll
        pkl_models[f"pos_ul2pr_{side}"] = joblib.load(
            os.path.join(model_dir, f"pos_ul2pr_{side}.pkl")
        )
        # 正向模型: pitch/roll → upper/lower
        key = f"pos_pr2ul_{side}"
        pkl_models[key] = joblib.load(os.path.join(model_dir, f"{key}.pkl"))
        # 提取权重系数
        pkl_coefs[side] = pkl_models[key].named_steps['ridge'].coef_
    
    pkl_is_initialized = True
    print(f"✅ 模型预加载完成: {sides}")


# ============================================
# 计算雅可比矩阵（内部函数）
# ============================================
def _compute_jacobian(p, r, side):
    """计算雅可比矩阵 J_pr2ul"""
    w_u, w_l = pkl_coefs[side][0], pkl_coefs[side][1]
    
    du_dp = (w_u[0] + 2*w_u[2]*p + w_u[3]*r + 3*w_u[5]*p**2 + 
             2*w_u[6]*p*r + w_u[7]*r**2)
    du_dr = (w_u[1] + w_u[3]*p + 2*w_u[4]*r + w_u[6]*p**2 + 
             2*w_u[7]*p*r + 3*w_u[8]*r**2)
    dl_dp = (w_l[0] + 2*w_l[2]*p + w_l[3]*r + 3*w_l[5]*p**2 + 
             2*w_l[6]*p*r + w_l[7]*r**2)
    dl_dr = (w_l[1] + w_l[3]*p + 2*w_l[4]*r + w_l[6]*p**2 + 
             2*w_l[7]*p*r + 3*w_l[8]*r**2)
    
    return np.array([[du_dp, du_dr], [dl_dp, dl_dr]])


# ============================================
# 核心接口1：计算等效关节增益
# ============================================

def compute_joint_gains(u_act, l_act, u_vel, l_vel,
                        kp_pitch, kd_pitch, kp_roll, kd_roll,
                        side="left"):
    """
    从电机状态计算 Upper/Lower 的等效 kp/kd
    
    返回: kp_u, kd_u, kp_l, kd_l, p_act, r_act, p_vel, r_vel
    """
    if not pkl_is_initialized:
        raise RuntimeError("请先调用 init_model_interface()")
    
    # 1. 推算脚踝位置
    model = pkl_models[f"pos_ul2pr_{side}"]
    p_act, r_act = model.predict(np.array([[u_act, l_act]]))[0]
    
    # 2. 计算雅可比
    J_pr2ul = _compute_jacobian(p_act, r_act, side)
    J_ul2pr = np.linalg.inv(J_pr2ul)
    
    # 3. 推算脚踝速度
    p_vel, r_vel = J_ul2pr @ np.array([u_vel, l_vel])
    
    # 4. 映射增益
    K_q = J_ul2pr.T @ np.diag([kp_pitch, kp_roll]) @ J_ul2pr
    D_q = J_ul2pr.T @ np.diag([kd_pitch, kd_roll]) @ J_ul2pr
    
    return K_q[0,0], D_q[0,0], K_q[1,1], D_q[1,1], p_act, r_act, p_vel, r_vel


# ============================================
# 核心接口2：计算力矩
# ============================================

def compute_torque(u_act, l_act, u_vel, l_vel,
                   p_des, r_des,
                   kp_pitch, kd_pitch, kp_roll, kd_roll,
                   side="left"):
    """
    从电机状态计算力矩指令
    
    返回: tau_u, tau_l, p_act, r_act, p_vel, r_vel
    """
    if not pkl_is_initialized:
        raise RuntimeError("请先调用 init_model_interface()")
    
    # 1. 推算脚踝位置
    model = pkl_models[f"pos_ul2pr_{side}"]
    p_act, r_act = model.predict(np.array([[u_act, l_act]]))[0]
    
    # 2. 计算雅可比
    J_pr2ul = _compute_jacobian(p_act, r_act, side)
    J_ul2pr = np.linalg.inv(J_pr2ul)
    
    # 3. 推算脚踝速度
    p_vel, r_vel = J_ul2pr @ np.array([u_vel, l_vel])
    
    # 4. 任务空间控制力
    F_pitch = kp_pitch * (p_des - p_act) - kd_pitch * p_vel
    F_roll = kp_roll * (r_des - r_act) - kd_roll * r_vel
    
    # 5. 映射到关节空间力矩
    tau_u, tau_l = J_pr2ul.T @ np.array([F_pitch, F_roll])
    
    return tau_u, tau_l, p_act, r_act, p_vel, r_vel


# ============================================
# 使用示例
# ============================================

if __name__ == "__main__":
    # 初始化（只执行一次）
    init_model_interface("models")
    
    # 输入参数
    u_act, l_act = -0.600615, -0.109792
    u_vel, l_vel = -0.130831, -0.042722
    kp_pitch, kd_pitch = 100.0, 10.0
    kp_roll, kd_roll = 80.0, 8.0
    
    # 计算等效增益
    kp_u, kd_u, kp_l, kd_l, p_act, r_act, p_vel, r_vel = compute_joint_gains(
        u_act, l_act, u_vel, l_vel,
        kp_pitch, kd_pitch, kp_roll, kd_roll
    )
    
    print(f"Upper: kp={kp_u:.4f}, kd={kd_u:.4f}")
    print(f"Lower: kp={kp_l:.4f}, kd={kd_l:.4f}")
    print(f"Pitch: pos={p_act:.6f}, vel={p_vel:.6f}")
    print(f"Roll:  pos={r_act:.6f}, vel={r_vel:.6f}")
    
    # 计算力矩
    p_des, r_des = 0.45, -0.35
    tau_u, tau_l, _, _, _, _ = compute_torque(
        u_act, l_act, u_vel, l_vel,
        p_des, r_des,
        kp_pitch, kd_pitch, kp_roll, kd_roll
    )
    
    print(f"力矩: tau_u={tau_u:.6f}, tau_l={tau_l:.6f}")