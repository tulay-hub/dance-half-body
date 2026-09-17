# retrain_models.py
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
import os

def train_all_models(left_file, right_file, save_dir="models"):
    """训练所有模型（正向和逆向映射）"""
    os.makedirs(save_dir, exist_ok=True)
    
    # 加载数据
    df_left = pd.read_csv(left_file)
    df_right = pd.read_csv(right_file)
    
    # ============================================
    # 1. 正向映射：pitch/roll → upper/lower (位置)
    # ============================================
    print("=" * 60)
    print("Training forward position models (pitch/roll → upper/lower)")
    print("=" * 60)
    
    # 输入：4个被动关节角度 [left_pitch, left_roll, right_pitch, right_roll]
    # 输出：4个主动关节角度 [left_upper, left_lower, right_upper, right_lower]
    X_pos = np.hstack([
        df_left[['left_ankle_pitch_joint_pos', 'left_ankle_roll_joint_pos']].values,
        df_right[['right_ankle_pitch_joint_pos', 'right_ankle_roll_joint_pos']].values
    ])
    Y_pos = np.hstack([
        df_left[['left_ankle_upper_joint_pos', 'left_ankle_lower_joint_pos']].values,
        df_right[['right_ankle_upper_joint_pos', 'right_ankle_lower_joint_pos']].values
    ])
    
    print(f"  X_pos shape: {X_pos.shape}, Y_pos shape: {Y_pos.shape}")
    print("Training combined position model...")
    model_pos = Pipeline([
        ('poly', PolynomialFeatures(degree=3, include_bias=False)),
        ('ridge', Ridge(alpha=0.01))
    ])
    model_pos.fit(X_pos, Y_pos)
    joblib.dump(model_pos, os.path.join(save_dir, "pos_pr2ul.pkl"))
    print(f"  Saved to pos_pr2ul.pkl (samples: {len(X_pos)})")
    
    # ============================================
    # 2. 正向映射：pitch/roll → upper/lower (速度)
    # ============================================
    print("\n" + "=" * 60)
    print("Training forward velocity models (pitch/roll → upper/lower)")
    print("=" * 60)
    
    # 输入：4个被动关节速度 [left_pitch_vel, left_roll_vel, right_pitch_vel, right_roll_vel]
    # 输出：4个主动关节速度 [left_upper_vel, left_lower_vel, right_upper_vel, right_lower_vel]
    X_vel = np.hstack([
        df_left[['left_ankle_pitch_joint_vel', 'left_ankle_roll_joint_vel']].values,
        df_right[['right_ankle_pitch_joint_vel', 'right_ankle_roll_joint_vel']].values
    ])
    Y_vel = np.hstack([
        df_left[['left_ankle_upper_joint_vel', 'left_ankle_lower_joint_vel']].values,
        df_right[['right_ankle_upper_joint_vel', 'right_ankle_lower_joint_vel']].values
    ])
    
    print(f"  X_vel shape: {X_vel.shape}, Y_vel shape: {Y_vel.shape}")
    print("Training combined velocity model...")
    model_vel = Pipeline([
        ('poly', PolynomialFeatures(degree=3, include_bias=False)),
        ('ridge', Ridge(alpha=0.01))
    ])
    model_vel.fit(X_vel, Y_vel)
    joblib.dump(model_vel, os.path.join(save_dir, "vel_pr2ul.pkl"))
    print(f"  Saved to vel_pr2ul.pkl (samples: {len(X_vel)})")
    
    # ============================================
    # 3. 逆向映射：upper/lower → pitch/roll (位置)
    # ============================================
    print("\n" + "=" * 60)
    print("Training inverse position models (upper/lower → pitch/roll)")
    print("=" * 60)
    
    # 输入：4个主动关节角度 [left_upper, left_lower, right_upper, right_lower]
    # 输出：4个被动关节角度 [left_pitch, left_roll, right_pitch, right_roll]
    X_inv_pos = np.hstack([
        df_left[['left_ankle_upper_joint_pos', 'left_ankle_lower_joint_pos']].values,
        df_right[['right_ankle_upper_joint_pos', 'right_ankle_lower_joint_pos']].values
    ])
    Y_inv_pos = np.hstack([
        df_left[['left_ankle_pitch_joint_pos', 'left_ankle_roll_joint_pos']].values,
        df_right[['right_ankle_pitch_joint_pos', 'right_ankle_roll_joint_pos']].values
    ])
    
    print(f"  X_inv_pos shape: {X_inv_pos.shape}, Y_inv_pos shape: {Y_inv_pos.shape}")
    print("Training combined inverse position model...")
    model_inv_pos = Pipeline([
        ('poly', PolynomialFeatures(degree=3, include_bias=False)),
        ('ridge', Ridge(alpha=0.01))
    ])
    model_inv_pos.fit(X_inv_pos, Y_inv_pos)
    joblib.dump(model_inv_pos, os.path.join(save_dir, "pos_ul2pr.pkl"))
    print(f"  Saved to pos_ul2pr.pkl (samples: {len(X_inv_pos)})")
    
    # ============================================
    # 4. 逆向映射：upper/lower → pitch/roll (速度)
    # ============================================
    print("\n" + "=" * 60)
    print("Training inverse velocity models (upper/lower → pitch/roll)")
    print("=" * 60)
    
    # 输入：4个主动关节速度 [left_upper_vel, left_lower_vel, right_upper_vel, right_lower_vel]
    # 输出：4个被动关节速度 [left_pitch_vel, left_roll_vel, right_pitch_vel, right_roll_vel]
    X_inv_vel = np.hstack([
        df_left[['left_ankle_upper_joint_vel', 'left_ankle_lower_joint_vel']].values,
        df_right[['right_ankle_upper_joint_vel', 'right_ankle_lower_joint_vel']].values
    ])
    Y_inv_vel = np.hstack([
        df_left[['left_ankle_pitch_joint_vel', 'left_ankle_roll_joint_vel']].values,
        df_right[['right_ankle_pitch_joint_vel', 'right_ankle_roll_joint_vel']].values
    ])
    
    print(f"  X_inv_vel shape: {X_inv_vel.shape}, Y_inv_vel shape: {Y_inv_vel.shape}")
    print("Training combined inverse velocity model...")
    model_inv_vel = Pipeline([
        ('poly', PolynomialFeatures(degree=3, include_bias=False)),
        ('ridge', Ridge(alpha=0.01))
    ])
    model_inv_vel.fit(X_inv_vel, Y_inv_vel)
    joblib.dump(model_inv_vel, os.path.join(save_dir, "vel_ul2pr.pkl"))
    print(f"  Saved to vel_ul2pr.pkl (samples: {len(X_inv_vel)})")
    
    print("\n" + "✅ All models training complete!")
    print(f"   Models saved to {save_dir}/")
    
    return {
        'pos_pr2ul': model_pos,
        'vel_pr2ul': model_vel,
        'pos_ul2pr': model_inv_pos,
        'vel_ul2pr': model_inv_vel
    }


# ============================================
# 预测函数（使用训练好的模型）
# ============================================

def predict_pos_pr2ul(left_pitch, left_roll, right_pitch, right_roll, model_dir="models"):
    """
    正向预测位置：4个被动关节角度 → 4个主动关节角度
    
    返回:
        left_upper, left_lower, right_upper, right_lower
    """
    model = joblib.load(os.path.join(model_dir, "pos_pr2ul.pkl"))
    X = np.array([[left_pitch, left_roll, right_pitch, right_roll]])
    Y_pred = model.predict(X)[0]
    return Y_pred[0], Y_pred[1], Y_pred[2], Y_pred[3]

def predict_vel_pr2ul(left_pitch_vel, left_roll_vel, right_pitch_vel, right_roll_vel, model_dir="models"):
    """
    正向预测速度：4个被动关节速度 → 4个主动关节速度
    
    返回:
        left_upper_vel, left_lower_vel, right_upper_vel, right_lower_vel
    """
    model = joblib.load(os.path.join(model_dir, "vel_pr2ul.pkl"))
    X = np.array([[left_pitch_vel, left_roll_vel, right_pitch_vel, right_roll_vel]])
    Y_pred = model.predict(X)[0]
    return Y_pred[0], Y_pred[1], Y_pred[2], Y_pred[3]

def predict_pos_ul2pr(left_upper, left_lower, right_upper, right_lower, model_dir="models"):
    """
    逆向预测位置：4个主动关节角度 → 4个被动关节角度
    
    返回:
        left_pitch, left_roll, right_pitch, right_roll
    """
    model = joblib.load(os.path.join(model_dir, "pos_ul2pr.pkl"))
    X = np.array([[left_upper, left_lower, right_upper, right_lower]])
    Y_pred = model.predict(X)[0]
    return Y_pred[0], Y_pred[1], Y_pred[2], Y_pred[3]

def predict_vel_ul2pr(left_upper_vel, left_lower_vel, right_upper_vel, right_lower_vel, model_dir="models"):
    """
    逆向预测速度：4个主动关节速度 → 4个被动关节速度
    
    返回:
        left_pitch_vel, left_roll_vel, right_pitch_vel, right_roll_vel
    """
    model = joblib.load(os.path.join(model_dir, "vel_ul2pr.pkl"))
    X = np.array([[left_upper_vel, left_lower_vel, right_upper_vel, right_lower_vel]])
    Y_pred = model.predict(X)[0]
    return Y_pred[0], Y_pred[1], Y_pred[2], Y_pred[3]


# ============================================
# 主程序：训练并测试
# ============================================

if __name__ == "__main__":
    # 训练所有模型
    models = train_all_models("left_joint_data_all.csv", "right_joint_data_all.csv")
    
    # 测试所有预测函数
    print("\n" + "=" * 60)
    print("Testing all prediction functions")
    print("=" * 60)
    
    df = pd.read_csv("left_joint_data_all.csv")
    df_right = pd.read_csv("right_joint_data_all.csv")
    idx = 6000
    
    # 获取数据
    left_pitch_pos = df['left_ankle_pitch_joint_pos'].iloc[idx]
    left_roll_pos = df['left_ankle_roll_joint_pos'].iloc[idx]
    right_pitch_pos = df_right['right_ankle_pitch_joint_pos'].iloc[idx]
    right_roll_pos = df_right['right_ankle_roll_joint_pos'].iloc[idx]
    
    left_upper_pos = df['left_ankle_upper_joint_pos'].iloc[idx]
    left_lower_pos = df['left_ankle_lower_joint_pos'].iloc[idx]
    right_upper_pos = df_right['right_ankle_upper_joint_pos'].iloc[idx]
    right_lower_pos = df_right['right_ankle_lower_joint_pos'].iloc[idx]
    
    # 正向位置预测测试
    print("\n1. Forward Position (4 pitch/roll → 4 upper/lower):")
    pred_left_upper, pred_left_lower, pred_right_upper, pred_right_lower = predict_pos_pr2ul(
        left_pitch_pos, left_roll_pos, right_pitch_pos, right_roll_pos
    )
    
    print(f"   Input:  L_Pitch={left_pitch_pos:.6f}, L_Roll={left_roll_pos:.6f}, R_Pitch={right_pitch_pos:.6f}, R_Roll={right_roll_pos:.6f}")
    print(f"   True:   L_Up={left_upper_pos:.6f}, L_Low={left_lower_pos:.6f}, R_Up={right_upper_pos:.6f}, R_Low={right_lower_pos:.6f}")
    print(f"   Pred:   L_Up={pred_left_upper:.6f}, L_Low={pred_left_lower:.6f}, R_Up={pred_right_upper:.6f}, R_Low={pred_right_lower:.6f}")
    
    # 逆向位置预测测试
    print("\n2. Inverse Position (4 upper/lower → 4 pitch/roll):")
    pred_left_pitch, pred_left_roll, pred_right_pitch, pred_right_roll = predict_pos_ul2pr(
        left_upper_pos, left_lower_pos, right_upper_pos, right_lower_pos
    )
    
    print(f"   Input:  L_Up={left_upper_pos:.6f}, L_Low={left_lower_pos:.6f}, R_Up={right_upper_pos:.6f}, R_Low={right_lower_pos:.6f}")
    print(f"   True:   L_Pitch={left_pitch_pos:.6f}, L_Roll={left_roll_pos:.6f}, R_Pitch={right_pitch_pos:.6f}, R_Roll={right_roll_pos:.6f}")
    print(f"   Pred:   L_Pitch={pred_left_pitch:.6f}, L_Roll={pred_left_roll:.6f}, R_Pitch={pred_right_pitch:.6f}, R_Roll={pred_right_roll:.6f}")
    print(f"   Error:  L_Pitch={abs(left_pitch_pos-pred_left_pitch):.6f} rad, L_Roll={abs(left_roll_pos-pred_left_roll):.6f} rad")
    print(f"          R_Pitch={abs(right_pitch_pos-pred_right_pitch):.6f} rad, R_Roll={abs(right_roll_pos-pred_right_roll):.6f} rad")
    
    # 速度数据
    left_pitch_vel = df['left_ankle_pitch_joint_vel'].iloc[idx]
    left_roll_vel = df['left_ankle_roll_joint_vel'].iloc[idx]
    right_pitch_vel = df_right['right_ankle_pitch_joint_vel'].iloc[idx]
    right_roll_vel = df_right['right_ankle_roll_joint_vel'].iloc[idx]
    
    left_upper_vel = df['left_ankle_upper_joint_vel'].iloc[idx]
    left_lower_vel = df['left_ankle_lower_joint_vel'].iloc[idx]
    right_upper_vel = df_right['right_ankle_upper_joint_vel'].iloc[idx]
    right_lower_vel = df_right['right_ankle_lower_joint_vel'].iloc[idx]
    
    # 正向速度预测测试
    print("\n3. Forward Velocity (4 pitch/roll velocities → 4 upper/lower velocities):")
    pred_left_upper_v, pred_left_lower_v, pred_right_upper_v, pred_right_lower_v = predict_vel_pr2ul(
        left_pitch_vel, left_roll_vel, right_pitch_vel, right_roll_vel
    )
    
    print(f"   Input:  L_Pitch={left_pitch_vel:.6f}, L_Roll={left_roll_vel:.6f}, R_Pitch={right_pitch_vel:.6f}, R_Roll={right_roll_vel:.6f}")
    print(f"   True:   L_Up={left_upper_vel:.6f}, L_Low={left_lower_vel:.6f}, R_Up={right_upper_vel:.6f}, R_Low={right_lower_vel:.6f}")
    print(f"   Pred:   L_Up={pred_left_upper_v:.6f}, L_Low={pred_left_lower_v:.6f}, R_Up={pred_right_upper_v:.6f}, R_Low={pred_right_lower_v:.6f}")
    
    # 逆向速度预测测试
    print("\n4. Inverse Velocity (4 upper/lower velocities → 4 pitch/roll velocities):")
    pred_left_pitch_v, pred_left_roll_v, pred_right_pitch_v, pred_right_roll_v = predict_vel_ul2pr(
        left_upper_vel, left_lower_vel, right_upper_vel, right_lower_vel
    )
    
    print(f"   Input:  L_Up={left_upper_vel:.6f}, L_Low={left_lower_vel:.6f}, R_Up={right_upper_vel:.6f}, R_Low={right_lower_vel:.6f}")
    print(f"   True:   L_Pitch={left_pitch_vel:.6f}, L_Roll={left_roll_vel:.6f}, R_Pitch={right_pitch_vel:.6f}, R_Roll={right_roll_vel:.6f}")
    print(f"   Pred:   L_Pitch={pred_left_pitch_v:.6f}, L_Roll={pred_left_roll_v:.6f}, R_Pitch={pred_right_pitch_v:.6f}, R_Roll={pred_right_roll_v:.6f}")
    print(f"   Error:  L_Pitch={abs(left_pitch_vel-pred_left_pitch_v):.6f} rad/s, L_Roll={abs(left_roll_vel-pred_left_roll_v):.6f} rad/s")
    print(f"          R_Pitch={abs(right_pitch_vel-pred_right_pitch_v):.6f} rad/s, R_Roll={abs(right_roll_vel-pred_right_roll_v):.6f} rad/s")
    
    print("\n" + "=" * 60)
    print("✅ All tests completed!")