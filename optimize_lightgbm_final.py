"""
LightGBM 최적화 전략 - Optuna 하이퍼파라미터 튜닝

목표: 14.138m → 13.5~14.0m
방법: Bayesian Optimization으로 최적 파라미터 탐색

작성일: 2025-12-19
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
import optuna
from optuna.samplers import TPESampler
import pickle

print("=" * 80)
print("  LightGBM 하이퍼파라미터 최적화 (Optuna)")
print("  목표: 14.138m → 13.5~14.0m")
print("=" * 80)
print()


def euclidean_distance(y_true, y_pred):
    """유클리드 거리 계산"""
    y_true = y_true.reshape(-1, 2)
    y_pred = y_pred.reshape(-1, 2)
    distances = np.sqrt(np.sum((y_pred - y_true) ** 2, axis=1))
    return distances.mean()


def objective(trial):
    """Optuna objective function"""

    # 하이퍼파라미터 탐색 공간
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'verbosity': -1,
        'seed': 42,

        # 탐색할 파라미터
        'num_leaves': trial.suggest_int('num_leaves', 20, 100),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),

        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),

        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),

        'min_split_gain': trial.suggest_float('min_split_gain', 0.0, 1.0),
        'min_child_weight': trial.suggest_float('min_child_weight', 1e-3, 10.0, log=True),
    }

    # 데이터 로드
    data = pd.read_csv('processed_train_data_v4.csv')
    game_ids = data['game_id'].values

    X = data.drop(columns=['game_episode', 'game_id', 'target_x', 'target_y', 'final_team_id'])
    y_x = data['target_x'].values
    y_y = data['target_y'].values

    # 5-Fold CV
    gkf = GroupKFold(n_splits=5)
    fold_scores = []

    for fold_idx, (train_idx, val_idx) in enumerate(gkf.split(X, groups=game_ids)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_x_train, y_x_val = y_x[train_idx], y_x[val_idx]
        y_y_train, y_y_val = y_y[train_idx], y_y[val_idx]

        # X 좌표 모델
        model_x = lgb.LGBMRegressor(**params)
        model_x.fit(X_train, y_x_train, eval_set=[(X_val, y_x_val)], callbacks=[lgb.early_stopping(50, verbose=False)])

        # Y 좌표 모델
        model_y = lgb.LGBMRegressor(**params)
        model_y.fit(X_train, y_y_train, eval_set=[(X_val, y_y_val)], callbacks=[lgb.early_stopping(50, verbose=False)])

        # 예측
        pred_x = model_x.predict(X_val)
        pred_y = model_y.predict(X_val)

        # 평가
        y_val = np.stack([y_x_val, y_y_val], axis=1)
        y_pred = np.stack([pred_x, pred_y], axis=1)

        score = euclidean_distance(y_val, y_pred)
        fold_scores.append(score)

    avg_score = np.mean(fold_scores)
    return avg_score


def main():
    print("📊 데이터 로딩...")
    data = pd.read_csv('processed_train_data_v4.csv')
    print(f"데이터 Shape: {data.shape}")
    print()

    print("🔍 Optuna 하이퍼파라미터 탐색 시작...")
    print("   - Trials: 100")
    print("   - Sampler: TPE (Bayesian Optimization)")
    print("   - 예상 시간: 1~2시간")
    print()

    # Optuna Study
    study = optuna.create_study(
        direction='minimize',
        sampler=TPESampler(seed=42)
    )

    study.optimize(objective, n_trials=100, show_progress_bar=True)

    # 결과
    print("\n" + "=" * 80)
    print("  최적화 완료!")
    print("=" * 80)
    print(f"\n✅ Best Score: {study.best_value:.4f}m")
    print("\n📊 Best Parameters:")
    for key, value in study.best_params.items():
        print(f"   - {key}: {value}")

    # 최적 파라미터 저장
    with open('best_params_lightgbm_optimized.pkl', 'wb') as f:
        pickle.dump(study.best_params, f)

    print(f"\n✅ 최적 파라미터 저장: best_params_lightgbm_optimized.pkl")

    # 성능 비교
    print("\n" + "=" * 80)
    print("  성능 비교")
    print("=" * 80)
    print(f"   - LightGBM V4 (기존): 14.138m (Public LB)")
    print(f"   - LightGBM Optimized: {study.best_value:.4f}m (5-Fold CV)")

    improvement = 14.138 - study.best_value
    print(f"\n📈 예상 개선: {improvement:+.4f}m ({improvement/14.138*100:.1f}%)")

    if study.best_value < 13.8:
        print("\n🎉🎉🎉 목표 달성! (< 13.8m)")
    elif study.best_value < 14.0:
        print("\n✅ 매우 좋은 성능! (< 14.0m)")
    elif study.best_value < 14.138:
        print("\n✅ 기존 모델 초과!")
    else:
        print("\n📊 추가 전략 필요 (피처 엔지니어링, 앙상블)")

    print("\n" + "=" * 80)
    print("다음 단계: train_lightgbm_optimized.py (최적 파라미터로 전체 학습)")
    print("=" * 80)


if __name__ == "__main__":
    main()

