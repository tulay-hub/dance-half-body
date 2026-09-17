# retrain_models.py
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
import os

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

def predict_pos_pr2ul(left_pitch, left_roll, right_pitch, right_roll, model_dir="models"):
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

def predict_pos_ul2pr(left_upper, left_lower, right_upper, right_lower, model_dir="models"):
    """
    逆向预测位置 - 双腿：4个主动关节角度 → 4个被动关节角度
    
    返回:
        left_pitch, left_roll, right_pitch, right_roll
    """
    left_pitch, left_roll = predict_pos_ul2pr_left(left_upper, left_lower, model_dir)
    right_pitch, right_roll = predict_pos_ul2pr_right(right_upper, right_lower, model_dir)
    return left_pitch, left_roll, right_pitch, right_roll


# ============================================
# 主程序：测试预测函数
# ============================================

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Testing position prediction functions (split left/right)")
    print("=" * 60)
    
    # 加载测试数据
    df_left = pd.read_csv("left_joint_data_all.csv")
    df_right = pd.read_csv("right_joint_data_all.csv")
    idx = 6000
    
    # 获取位置数据 - 左腿
    left_pitch_pos = df_left['left_ankle_pitch_joint_pos'].iloc[idx]
    left_roll_pos = df_left['left_ankle_roll_joint_pos'].iloc[idx]
    left_upper_pos = df_left['left_ankle_upper_joint_pos'].iloc[idx]
    left_lower_pos = df_left['left_ankle_lower_joint_pos'].iloc[idx]
    
    # 获取位置数据 - 右腿
    right_pitch_pos = df_right['right_ankle_pitch_joint_pos'].iloc[idx]
    right_roll_pos = df_right['right_ankle_roll_joint_pos'].iloc[idx]
    right_upper_pos = df_right['right_ankle_upper_joint_pos'].iloc[idx]
    right_lower_pos = df_right['right_ankle_lower_joint_pos'].iloc[idx]
    
    # ============================================
    # 测试1：正向位置预测 - 左腿单独
    # ============================================
    print("\n1. Forward Position - LEFT leg (2 pitch/roll → 2 upper/lower):")
    pred_left_upper, pred_left_lower = predict_pos_pr2ul_left(
        left_pitch_pos, left_roll_pos
    )
    
    print(f"   Input:  L_Pitch={left_pitch_pos:.6f}, L_Roll={left_roll_pos:.6f}")
    print(f"   True:   L_Up={left_upper_pos:.6f}, L_Low={left_lower_pos:.6f}")
    print(f"   Pred:   L_Up={pred_left_upper:.6f}, L_Low={pred_left_lower:.6f}")
    print(f"   Error:  L_Up={abs(left_upper_pos-pred_left_upper):.6f}, L_Low={abs(left_lower_pos-pred_left_lower):.6f}")
    
    # ============================================
    # 测试2：正向位置预测 - 右腿单独
    # ============================================
    print("\n2. Forward Position - RIGHT leg (2 pitch/roll → 2 upper/lower):")
    pred_right_upper, pred_right_lower = predict_pos_pr2ul_right(
        right_pitch_pos, right_roll_pos
    )
    
    print(f"   Input:  R_Pitch={right_pitch_pos:.6f}, R_Roll={right_roll_pos:.6f}")
    print(f"   True:   R_Up={right_upper_pos:.6f}, R_Low={right_lower_pos:.6f}")
    print(f"   Pred:   R_Up={pred_right_upper:.6f}, R_Low={pred_right_lower:.6f}")
    print(f"   Error:  R_Up={abs(right_upper_pos-pred_right_upper):.6f}, R_Low={abs(right_lower_pos-pred_right_lower):.6f}")
    
    # ============================================
    # 测试3：逆向位置预测 - 左腿单独
    # ============================================
    print("\n3. Inverse Position - LEFT leg (2 upper/lower → 2 pitch/roll):")
    pred_left_pitch, pred_left_roll = predict_pos_ul2pr_left(
        left_upper_pos, left_lower_pos
    )
    
    print(f"   Input:  L_Up={left_upper_pos:.6f}, L_Low={left_lower_pos:.6f}")
    print(f"   True:   L_Pitch={left_pitch_pos:.6f}, L_Roll={left_roll_pos:.6f}")
    print(f"   Pred:   L_Pitch={pred_left_pitch:.6f}, L_Roll={pred_left_roll:.6f}")
    print(f"   Error:  L_Pitch={abs(left_pitch_pos-pred_left_pitch):.6f}, L_Roll={abs(left_roll_pos-pred_left_roll):.6f}")
    
    # ============================================
    # 测试4：逆向位置预测 - 右腿单独
    # ============================================
    print("\n4. Inverse Position - RIGHT leg (2 upper/lower → 2 pitch/roll):")
    pred_right_pitch, pred_right_roll = predict_pos_ul2pr_right(
        right_upper_pos, right_lower_pos
    )
    
    print(f"   Input:  R_Up={right_upper_pos:.6f}, R_Low={right_lower_pos:.6f}")
    print(f"   True:   R_Pitch={right_pitch_pos:.6f}, R_Roll={right_roll_pos:.6f}")
    print(f"   Pred:   R_Pitch={pred_right_pitch:.6f}, R_Roll={pred_right_roll:.6f}")
    print(f"   Error:  R_Pitch={abs(right_pitch_pos-pred_right_pitch):.6f}, R_Roll={abs(right_roll_pos-pred_right_roll):.6f}")
    
    # ============================================
    # 测试5：双腿同时预测（调用合并函数）
    # ============================================
    print("\n5. Both legs prediction (using combined function):")
    left_upper, left_lower, right_upper, right_lower = predict_pos_pr2ul(
        left_pitch_pos, left_roll_pos, right_pitch_pos, right_roll_pos
    )
    print(f"   Left:  L_Up={left_upper:.6f}, L_Low={left_lower:.6f}")
    print(f"   Right: R_Up={right_upper:.6f}, R_Low={right_lower:.6f}")
    
    left_pitch, left_roll, right_pitch, right_roll = predict_pos_ul2pr(
        left_upper_pos, left_lower_pos, right_upper_pos, right_lower_pos
    )
    print(f"   Left:  L_Pitch={left_pitch:.6f}, L_Roll={left_roll:.6f}")
    print(f"   Right: R_Pitch={right_pitch:.6f}, R_Roll={right_roll:.6f}")
    
    # ============================================
    # 测试6：手动指定输入 - 左腿
    # ============================================
    print("\n6. Manual Forward Prediction - LEFT leg:")
    left_pitch = -0.22
    left_roll = 0.0
    
    left_upper, left_lower = predict_pos_pr2ul_left(left_pitch, left_roll)
    print(f"   Input:  L_Pitch={left_pitch:.6f}, L_Roll={left_roll:.6f}")
    print(f"   Output: L_Upper={left_upper:.6f}, L_Lower={left_lower:.6f}")
    
    # ============================================
    # 测试7：手动指定输入 - 右腿
    # ============================================
    print("\n7. Manual Forward Prediction - RIGHT leg:")
    right_pitch = -0.22
    right_roll = 0.0
    
    right_upper, right_lower = predict_pos_pr2ul_right(right_pitch, right_roll)
    print(f"   Input:  R_Pitch={right_pitch:.6f}, R_Roll={right_roll:.6f}")
    print(f"   Output: R_Upper={right_upper:.6f}, R_Lower={right_lower:.6f}")
    
    # ============================================
    # 测试8：手动指定输入进行逆向预测
    # ============================================
    print("\n8. Manual Inverse Prediction - Both legs:")
    left_upper = 0.7
    left_lower = 0.7 
    right_upper = 0.7 
    right_lower = 0.7
    
    left_pitch, left_roll, right_pitch, right_roll = predict_pos_ul2pr(
        left_upper, left_lower, right_upper, right_lower
    )
    
    print(f"   Input:  L_Upper={left_upper:.6f}, L_Lower={left_lower:.6f}, R_Upper={right_upper:.6f}, R_Lower={right_lower:.6f}")
    print(f"   Output: L_Pitch={left_pitch:.6f}, L_Roll={left_roll:.6f}, R_Pitch={right_pitch:.6f}, R_Roll={right_roll:.6f}")
    
    print("\n" + "=" * 60)
    print("✅ All position tests completed!")