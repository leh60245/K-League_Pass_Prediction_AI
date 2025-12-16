"""
LightGBM V2.1 - Feature Selection 적용

목표: 중요도 낮은 피처 제거 및 하이퍼파라미터 튜닝
"""

import pandas as pd
import numpy as np
import pickle
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GroupKFold
import warnings
warnings.filterwarnings('ignore')

def euclidean_distance(y_true, y_pred):
    """유클리드 거리 계산"""
    distances = np.sqrt((y_true[:, 0] - y_pred[:, 0])**2 +
                       (y_true[:, 1] - y_pred[:, 1])**2)
    return distances.mean()

def main():
    print("=" * 80)
    print("  LightGBM V2.1 - Feature Selection + 하이퍼파라미터 튜닝")
    print("=" * 80)
    print()

    # 1. 데이터 로딩
    print("📊 V2 데이터 로딩...")
    data = pd.read_csv('processed_train_data_v2.csv')
    print(f"✅ 데이터: {data.shape}")

    # 2. 피처 선택 (V1에서 효과적이었던 피처 + V2 새 피처 중 중요한 것만)
    print("\n🎯 Feature Selection...")

    # V1 핵심 피처
    core_features = [
        'start_x', 'start_y',
        'delta_x', 'delta_y', 'distance',
        'distance_to_goal_start', 'distance_to_goal_end',
        'goal_approach',
        'shooting_angle',
        'in_penalty_area', 'in_final_third',
        'episode_length', 'event_order',
        'x_progression', 'x_total_progression',
        'velocity', 'velocity_x', 'velocity_y',
        'tempo',
        'direction_consistency',
        'forward_momentum',
        'goal_approach_trend',
        'prev_start_x', 'prev_start_y',
        'prev_end_x', 'prev_end_y',
        'period_id', 'is_home'
    ]

    # V2 새 피처 중 핵심만 선택
    new_features_v2 = [
        'distance_to_goal_inv',  # 골문 거리 역수
        'shooting_angle_sin',  # 각도 변환
        'shooting_angle_cos',
        'start_x_squared',  # 비선형
        'goal_dist_angle_interaction',  # 상호작용
        'goal_urgency',  # 위치 특화
        'is_central_corridor',
        'player_avg_x',  # 컨텍스트
        'player_avg_pass_dist',
        'team_aggression',
        'time_pressure',
        'player_position_deviation'
    ]

    selected_features = core_features + new_features_v2

    # 실제 존재하는 피처만
    available_features = [col for col in selected_features if col in data.columns]
    print(f"✅ 선택된 피처: {len(available_features)}개 (전체 75개 → {len(available_features)}개)")

    X = data[available_features]
    y = data[['end_x', 'end_y']].values

    # 3. Train/Val Split
    print("\n📊 5-Fold Cross Validation...")

    gkf = GroupKFold(n_splits=5)
    fold_scores = []

    # 개선된 하이퍼파라미터
    params = {
        'n_estimators': 800,  # 증가
        'learning_rate': 0.03,  # 감소 (더 세밀하게)
        'max_depth': 10,  # 증가
        'num_leaves': 127,  # 증가
        'subsample': 0.85,
        'colsample_bytree': 0.85,
        'min_child_samples': 20,
        'reg_alpha': 0.1,  # L1 regularization
        'reg_lambda': 0.1,  # L2 regularization
        'random_state': 42,
        'verbose': -1
    }

    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, groups=data['game_id'])):
        print(f"\nFold {fold+1}/5", end=" ")

        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # X 좌표 모델
        model_x = LGBMRegressor(**params)
        model_x.fit(X_train, y_train[:, 0])

        # Y 좌표 모델
        model_y = LGBMRegressor(**params)
        model_y.fit(X_train, y_train[:, 1])

        # 예측
        y_pred_x = model_x.predict(X_val)
        y_pred_y = model_y.predict(X_val)
        y_pred = np.column_stack([y_pred_x, y_pred_y])

        # 평가
        eucl_dist = euclidean_distance(y_val, y_pred)
        mse_x = mean_squared_error(y_val[:, 0], y_pred[:, 0])
        mse_y = mean_squared_error(y_val[:, 1], y_pred[:, 1])

        print(f"→ {eucl_dist:.4f}m")

        fold_scores.append({
            'fold': fold + 1,
            'euclidean': eucl_dist,
            'mse_x': mse_x,
            'mse_y': mse_y
        })

        # Feature Importance (첫 fold)
        if fold == 0:
            print("\n📊 Feature Importance Top 15:")
            importance_x = model_x.feature_importances_
            importance_df = pd.DataFrame({
                'feature': available_features,
                'importance': importance_x
            }).sort_values('importance', ascending=False)

            for i, row in importance_df.head(15).iterrows():
                print(f"  {i+1:2d}. {row['feature']:35s}: {row['importance']:8.0f}")

    # 4. 결과 요약
    print("\n" + "="*80)
    print("  결과 요약")
    print("="*80)

    scores_df = pd.DataFrame(fold_scores)
    mean_score = scores_df['euclidean'].mean()
    std_score = scores_df['euclidean'].std()

    print(f"\n평균 유클리드 거리: {mean_score:.4f}m ± {std_score:.4f}m")
    print(f"평균 MSE X: {scores_df['mse_x'].mean():.4f}")
    print(f"평균 MSE Y: {scores_df['mse_y'].mean():.4f}")

    print("\n" + "="*80)
    print("  성능 비교")
    print("="*80)
    print(f"V1 (54개 피처):  0.93m")
    print(f"V2 (75개 피처):  1.06m")
    print(f"V2.1 ({len(available_features)}개 피처): {mean_score:.4f}m")

    improvement_from_v1 = (0.93 - mean_score) / 0.93 * 100
    improvement_from_v2 = (1.06 - mean_score) / 1.06 * 100

    if mean_score < 0.93:
        print(f"\n✅ V1 대비 {-improvement_from_v1:.2f}% 개선!")
    elif mean_score < 1.06:
        print(f"\n✅ V2 대비 {-improvement_from_v2:.2f}% 개선")
    else:
        print(f"\n⚠️  추가 튜닝 필요")

    # 5. 최종 모델 학습
    if mean_score <= 0.93:
        print("\n💾 성능이 개선되어 최종 모델 저장...")

        model_x_final = LGBMRegressor(**params)
        model_y_final = LGBMRegressor(**params)

        model_x_final.fit(X, y[:, 0])
        model_y_final.fit(X, y[:, 1])

        with open('lightgbm_model_v2.1.pkl', 'wb') as f:
            pickle.dump({
                'model_x': model_x_final,
                'model_y': model_y_final,
                'feature_cols': available_features,
                'val_score': mean_score
            }, f)
        print("✅ 모델 저장: lightgbm_model_v2.1.pkl")

    print("\n" + "="*80)
    print("✅ V2.1 학습 완료!")
    print("="*80)

if __name__ == "__main__":
    main()

