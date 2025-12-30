"""
LightGBM 최종 학습 (최적화된 파라미터 사용)

작성일: 2025-12-19
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
import pickle

print("=" * 80)
print("  LightGBM 최종 학습 (최적 파라미터)")
print("=" * 80)
print()


def euclidean_distance(y_true, y_pred):
    y_true = y_true.reshape(-1, 2)
    y_pred = y_pred.reshape(-1, 2)
    distances = np.sqrt(np.sum((y_pred - y_true) ** 2, axis=1))
    return distances.mean()


def main():
    # 최적 파라미터 로드
    print("📦 최적 파라미터 로딩...")
    with open('best_params_lightgbm_optimized.pkl', 'rb') as f:
        best_params = pickle.load(f)

    print("✅ 최적 파라미터:")
    for key, value in best_params.items():
        print(f"   - {key}: {value}")
    print()

    # 기본 파라미터 추가
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'verbosity': -1,
        'seed': 42,
        **best_params
    }

    # 데이터 로드
    print("📊 데이터 로딩...")
    data = pd.read_csv('processed_train_data_v4.csv')
    print(f"데이터 Shape: {data.shape}")
    print()

    game_ids = data['game_id'].values
    X = data.drop(columns=['game_episode', 'game_id', 'target_x', 'target_y', 'final_team_id'])
    y_x = data['target_x'].values
    y_y = data['target_y'].values

    # 5-Fold 학습
    print("🚀 5-Fold 학습 시작...")
    gkf = GroupKFold(n_splits=5)

    models_x = []
    models_y = []
    fold_scores = []

    for fold_idx, (train_idx, val_idx) in enumerate(gkf.split(X, groups=game_ids)):
        print(f"\nFold {fold_idx + 1}/5")
        print("-" * 60)

        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_x_train, y_x_val = y_x[train_idx], y_x[val_idx]
        y_y_train, y_y_val = y_y[train_idx], y_y[val_idx]

        # X 좌표 모델
        print("  Training X model...")
        model_x = lgb.LGBMRegressor(**params)
        model_x.fit(
            X_train, y_x_train,
            eval_set=[(X_val, y_x_val)],
            callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(100)]
        )
        models_x.append(model_x)

        # Y 좌표 모델
        print("  Training Y model...")
        model_y = lgb.LGBMRegressor(**params)
        model_y.fit(
            X_train, y_y_train,
            eval_set=[(X_val, y_y_val)],
            callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(100)]
        )
        models_y.append(model_y)

        # 평가
        pred_x = model_x.predict(X_val)
        pred_y = model_y.predict(X_val)

        y_val = np.stack([y_x_val, y_y_val], axis=1)
        y_pred = np.stack([pred_x, pred_y], axis=1)

        score = euclidean_distance(y_val, y_pred)
        fold_scores.append(score)

        print(f"  ✅ Fold {fold_idx + 1} Score: {score:.4f}m")

    # 결과
    print("\n" + "=" * 80)
    print("  학습 완료!")
    print("=" * 80)

    print("\nFold별 결과:")
    for i, score in enumerate(fold_scores):
        print(f"   Fold {i+1}: {score:.4f}m")

    avg_score = np.mean(fold_scores)
    std_score = np.std(fold_scores)

    print(f"\n✅ 평균 Score: {avg_score:.4f}m ± {std_score:.4f}m")

    # 모델 저장
    print("\n💾 모델 저장...")
    with open('lightgbm_optimized_5fold_models.pkl', 'wb') as f:
        pickle.dump({
            'models_x': models_x,
            'models_y': models_y,
            'params': params,
            'fold_scores': fold_scores,
            'avg_score': avg_score
        }, f)

    print("✅ 모델 저장: lightgbm_optimized_5fold_models.pkl")

    # 성능 비교
    print("\n" + "=" * 80)
    print("  성능 비교")
    print("=" * 80)
    print(f"   - LightGBM V4 (기존): ~1.5m (Val), 14.138m (Public LB)")
    print(f"   - LightGBM Optimized: {avg_score:.4f}m (Val)")

    # Public LB 예상
    # Val 1.5m → Public 14.138m (약 9.4배)
    # 비율 유지 가정
    expected_public = avg_score * (14.138 / 1.5)
    print(f"\n📊 예상 Public LB: {expected_public:.4f}m")

    if expected_public < 13.8:
        print("\n🎉🎉🎉 목표 달성 예상! (< 13.8m)")
    elif expected_public < 14.0:
        print("\n✅ 매우 좋은 성능 예상! (< 14.0m)")
    elif expected_public < 14.138:
        print("\n✅ 기존 성능 초과 예상!")
    else:
        print("\n📊 비슷한 성능 예상")

    print("\n" + "=" * 80)
    print("다음 단계: inference_lightgbm_optimized.py (Test 추론)")
    print("=" * 80)


if __name__ == "__main__":
    main()

