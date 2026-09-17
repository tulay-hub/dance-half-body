# retrain_models.py
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
import os

def train_all_models(left_file, right_file, save_dir="models"):
    """训练所有模型（正向和逆向映射，左右分开）"""
    os.makedirs(save_dir, exist_ok=True)
    
    # 加载数据
    df_left = pd.read_csv(left_file)
    df_right = pd.read_csv(right_file)
    
    # ============================================
    # 1. 正向映射：pitch/roll → upper/lower (位置) - 左腿
    # ============================================
    print("=" * 60)
    print("Training forward position models - LEFT leg (pitch/roll → upper/lower)")
    print("=" * 60)
    
    # 输入：2个被动关节角度 [left_pitch, left_roll]
    # 输出：2个主动关节角度 [left_upper, left_lower]
    X_pos_left = df_left[['left_ankle_pitch_joint_pos', 'left_ankle_roll_joint_pos']].values
    Y_pos_left = df_left[['left_ankle_upper_joint_pos', 'left_ankle_lower_joint_pos']].values
    
    print(f"  X_pos_left shape: {X_pos_left.shape}, Y_pos_left shape: {Y_pos_left.shape}")
    print("Training left leg position model...")
    model_pos_left = Pipeline([
        ('poly', PolynomialFeatures(degree=3, include_bias=False)),
        ('ridge', Ridge(alpha=0.01))
    ])
    model_pos_left.fit(X_pos_left, Y_pos_left)
    joblib.dump(model_pos_left, os.path.join(save_dir, "pos_pr2ul_left.pkl"))
    print(f"  Saved to pos_pr2ul_left.pkl (samples: {len(X_pos_left)})")
    
    # ============================================
    # 2. 正向映射：pitch/roll → upper/lower (位置) - 右腿
    # ============================================
    print("\n" + "=" * 60)
    print("Training forward position models - RIGHT leg (pitch/roll → upper/lower)")
    print("=" * 60)
    
    # 输入：2个被动关节角度 [right_pitch, right_roll]
    # 输出：2个主动关节角度 [right_upper, right_lower]
    X_pos_right = df_right[['right_ankle_pitch_joint_pos', 'right_ankle_roll_joint_pos']].values
    Y_pos_right = df_right[['right_ankle_upper_joint_pos', 'right_ankle_lower_joint_pos']].values
    
    print(f"  X_pos_right shape: {X_pos_right.shape}, Y_pos_right shape: {Y_pos_right.shape}")
    print("Training right leg position model...")
    model_pos_right = Pipeline([
        ('poly', PolynomialFeatures(degree=3, include_bias=False)),
        ('ridge', Ridge(alpha=0.01))
    ])
    model_pos_right.fit(X_pos_right, Y_pos_right)
    joblib.dump(model_pos_right, os.path.join(save_dir, "pos_pr2ul_right.pkl"))
    print(f"  Saved to pos_pr2ul_right.pkl (samples: {len(X_pos_right)})")
    
    # ============================================
    # 3. 逆向映射：upper/lower → pitch/roll (位置) - 左腿
    # ============================================
    print("\n" + "=" * 60)
    print("Training inverse position models - LEFT leg (upper/lower → pitch/roll)")
    print("=" * 60)
    
    # 输入：2个主动关节角度 [left_upper, left_lower]
    # 输出：2个被动关节角度 [left_pitch, left_roll]
    X_inv_pos_left = df_left[['left_ankle_upper_joint_pos', 'left_ankle_lower_joint_pos']].values
    Y_inv_pos_left = df_left[['left_ankle_pitch_joint_pos', 'left_ankle_roll_joint_pos']].values
    
    print(f"  X_inv_pos_left shape: {X_inv_pos_left.shape}, Y_inv_pos_left shape: {Y_inv_pos_left.shape}")
    print("Training left leg inverse position model...")
    model_inv_pos_left = Pipeline([
        ('poly', PolynomialFeatures(degree=3, include_bias=False)),
        ('ridge', Ridge(alpha=0.01))
    ])
    model_inv_pos_left.fit(X_inv_pos_left, Y_inv_pos_left)
    joblib.dump(model_inv_pos_left, os.path.join(save_dir, "pos_ul2pr_left.pkl"))
    print(f"  Saved to pos_ul2pr_left.pkl (samples: {len(X_inv_pos_left)})")
    
    # ============================================
    # 4. 逆向映射：upper/lower → pitch/roll (位置) - 右腿
    # ============================================
    print("\n" + "=" * 60)
    print("Training inverse position models - RIGHT leg (upper/lower → pitch/roll)")
    print("=" * 60)
    
    # 输入：2个主动关节角度 [right_upper, right_lower]
    # 输出：2个被动关节角度 [right_pitch, right_roll]
    X_inv_pos_right = df_right[['right_ankle_upper_joint_pos', 'right_ankle_lower_joint_pos']].values
    Y_inv_pos_right = df_right[['right_ankle_pitch_joint_pos', 'right_ankle_roll_joint_pos']].values
    
    print(f"  X_inv_pos_right shape: {X_inv_pos_right.shape}, Y_inv_pos_right shape: {Y_inv_pos_right.shape}")
    print("Training right leg inverse position model...")
    model_inv_pos_right = Pipeline([
        ('poly', PolynomialFeatures(degree=3, include_bias=False)),
        ('ridge', Ridge(alpha=0.01))
    ])
    model_inv_pos_right.fit(X_inv_pos_right, Y_inv_pos_right)
    joblib.dump(model_inv_pos_right, os.path.join(save_dir, "pos_ul2pr_right.pkl"))
    print(f"  Saved to pos_ul2pr_right.pkl (samples: {len(X_inv_pos_right)})")
    
    print("\n" + "✅ All models training complete!")
    print(f"   Models saved to {save_dir}/")
    
    return {
        'pos_pr2ul_left': model_pos_left,
        'pos_pr2ul_right': model_pos_right,
        'pos_ul2pr_left': model_inv_pos_left,
        'pos_ul2pr_right': model_inv_pos_right
    }


# ============================================
# 预测函数（使用训练好的模型）- 左右分开
# ============================================

def predict_pos_pr2ul_left(left_pitch, left_roll, model_dir="models"):
    """
    正向预测位置 - 左腿：2个被动关节角度 → 2个主动关节角度
    
    返回:
        left_upper, left_lower
    """
    model = joblib.load(os.path.join(model_dir, "pos_pr2ul_left.pkl"))
    X = np.array([[left_pitch, left_roll]])
    Y_pred = model.predict(X)[0]
    return Y_pred[0], Y_pred[1]

def predict_pos_pr2ul_right(right_pitch, right_roll, model_dir="models"):
    """
    正向预测位置 - 右腿：2个被动关节角度 → 2个主动关节角度
    
    返回:
        right_upper, right_lower
    """
    model = joblib.load(os.path.join(model_dir, "pos_pr2ul_right.pkl"))
    X = np.array([[right_pitch, right_roll]])
    Y_pred = model.predict(X)[0]
    return Y_pred[0], Y_pred[1]

def predict_pos_pr2ul_both(left_pitch, left_roll, right_pitch, right_roll, model_dir="models"):
    """
    正向预测位置 - 双腿：4个被动关节角度 → 4个主动关节角度
    
    返回:
        left_upper, left_lower, right_upper, right_lower
    """
    left_upper, left_lower = predict_pos_pr2ul_left(left_pitch, left_roll, model_dir)
    right_upper, right_lower = predict_pos_pr2ul_right(right_pitch, right_roll, model_dir)
    return left_upper, left_lower, right_upper, right_lower

def predict_pos_ul2pr_left(left_upper, left_lower, model_dir="models"):
    """
    逆向预测位置 - 左腿：2个主动关节角度 → 2个被动关节角度
    
    返回:
        left_pitch, left_roll
    """
    model = joblib.load(os.path.join(model_dir, "pos_ul2pr_left.pkl"))
    X = np.array([[left_upper, left_lower]])
    Y_pred = model.predict(X)[0]
    return Y_pred[0], Y_pred[1]

def predict_pos_ul2pr_right(right_upper, right_lower, model_dir="models"):
    """
    逆向预测位置 - 右腿：2个主动关节角度 → 2个被动关节角度
    
    返回:
        right_pitch, right_roll
    """
    model = joblib.load(os.path.join(model_dir, "pos_ul2pr_right.pkl"))
    X = np.array([[right_upper, right_lower]])
    Y_pred = model.predict(X)[0]
    return Y_pred[0], Y_pred[1]

def predict_pos_ul2pr_both(left_upper, left_lower, right_upper, right_lower, model_dir="models"):
    """
    逆向预测位置 - 双腿：4个主动关节角度 → 4个被动关节角度
    
    返回:
        left_pitch, left_roll, right_pitch, right_roll
    """
    left_pitch, left_roll = predict_pos_ul2pr_left(left_upper, left_lower, model_dir)
    right_pitch, right_roll = predict_pos_ul2pr_right(right_upper, right_lower, model_dir)
    return left_pitch, left_roll, right_pitch, right_roll


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
    
    df_left = pd.read_csv("left_joint_data_all.csv")
    df_right = pd.read_csv("right_joint_data_all.csv")
    idx = 6000
    
    # 获取数据 - 左腿
    left_pitch_pos = df_left['left_ankle_pitch_joint_pos'].iloc[idx]
    left_roll_pos = df_left['left_ankle_roll_joint_pos'].iloc[idx]
    left_upper_pos = df_left['left_ankle_upper_joint_pos'].iloc[idx]
    left_lower_pos = df_left['left_ankle_lower_joint_pos'].iloc[idx]
    
    # 获取数据 - 右腿
    right_pitch_pos = df_right['right_ankle_pitch_joint_pos'].iloc[idx]
    right_roll_pos = df_right['right_ankle_roll_joint_pos'].iloc[idx]
    right_upper_pos = df_right['right_ankle_upper_joint_pos'].iloc[idx]
    right_lower_pos = df_right['right_ankle_lower_joint_pos'].iloc[idx]
    
    # ============================================
    # 测试1：正向位置预测 - 左腿
    # ============================================
    print("\n1. Forward Position - LEFT leg (pitch/roll → upper/lower):")
    pred_left_upper, pred_left_lower = predict_pos_pr2ul_left(
        left_pitch_pos, left_roll_pos
    )
    
    print(f"   Input:  L_Pitch={left_pitch_pos:.6f}, L_Roll={left_roll_pos:.6f}")
    print(f"   True:   L_Up={left_upper_pos:.6f}, L_Low={left_lower_pos:.6f}")
    print(f"   Pred:   L_Up={pred_left_upper:.6f}, L_Low={pred_left_lower:.6f}")
    print(f"   Error:  L_Up={abs(left_upper_pos-pred_left_upper):.6f}, L_Low={abs(left_lower_pos-pred_left_lower):.6f}")
    
    # ============================================
    # 测试2：正向位置预测 - 右腿
    # ============================================
    print("\n2. Forward Position - RIGHT leg (pitch/roll → upper/lower):")
    pred_right_upper, pred_right_lower = predict_pos_pr2ul_right(
        right_pitch_pos, right_roll_pos
    )
    
    print(f"   Input:  R_Pitch={right_pitch_pos:.6f}, R_Roll={right_roll_pos:.6f}")
    print(f"   True:   R_Up={right_upper_pos:.6f}, R_Low={right_lower_pos:.6f}")
    print(f"   Pred:   R_Up={pred_right_upper:.6f}, R_Low={pred_right_lower:.6f}")
    print(f"   Error:  R_Up={abs(right_upper_pos-pred_right_upper):.6f}, R_Low={abs(right_lower_pos-pred_right_lower):.6f}")
    
    # ============================================
    # 测试3：逆向位置预测 - 左腿
    # ============================================
    print("\n3. Inverse Position - LEFT leg (upper/lower → pitch/roll):")
    pred_left_pitch, pred_left_roll = predict_pos_ul2pr_left(
        left_upper_pos, left_lower_pos
    )
    
    print(f"   Input:  L_Up={left_upper_pos:.6f}, L_Low={left_lower_pos:.6f}")
    print(f"   True:   L_Pitch={left_pitch_pos:.6f}, L_Roll={left_roll_pos:.6f}")
    print(f"   Pred:   L_Pitch={pred_left_pitch:.6f}, L_Roll={pred_left_roll:.6f}")
    print(f"   Error:  L_Pitch={abs(left_pitch_pos-pred_left_pitch):.6f}, L_Roll={abs(left_roll_pos-pred_left_roll):.6f}")
    
    # ============================================
    # 测试4：逆向位置预测 - 右腿
    # ============================================
    print("\n4. Inverse Position - RIGHT leg (upper/lower → pitch/roll):")
    pred_right_pitch, pred_right_roll = predict_pos_ul2pr_right(
        right_upper_pos, right_lower_pos
    )
    
    print(f"   Input:  R_Up={right_upper_pos:.6f}, R_Low={right_lower_pos:.6f}")
    print(f"   True:   R_Pitch={right_pitch_pos:.6f}, R_Roll={right_roll_pos:.6f}")
    print(f"   Pred:   R_Pitch={pred_right_pitch:.6f}, R_Roll={pred_right_roll:.6f}")
    print(f"   Error:  R_Pitch={abs(right_pitch_pos-pred_right_pitch):.6f}, R_Roll={abs(right_roll_pos-pred_right_roll):.6f}")
    
    # ============================================
    # 测试5：双腿一起预测
    # ============================================
    print("\n5. Both legs prediction:")
    left_upper, left_lower, right_upper, right_lower = predict_pos_pr2ul_both(
        left_pitch_pos, left_roll_pos, right_pitch_pos, right_roll_pos
    )
    print(f"   Left:  L_Up={left_upper:.6f}, L_Low={left_lower:.6f}")
    print(f"   Right: R_Up={right_upper:.6f}, R_Low={right_lower:.6f}")
    
    left_pitch, left_roll, right_pitch, right_roll = predict_pos_ul2pr_both(
        left_upper_pos, left_lower_pos, right_upper_pos, right_lower_pos
    )
    print(f"   Left:  L_Pitch={left_pitch:.6f}, L_Roll={left_roll:.6f}")
    print(f"   Right: R_Pitch={right_pitch:.6f}, R_Roll={right_roll:.6f}")
    
    print("\n" + "=" * 60)
    print("✅ All tests completed!")