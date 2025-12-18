"""
LightGBM V4 - Optuna 하이퍼파라미터 최적화

목표: V4의 잠재력 최대한 활용
예상 개선: 0.2-0.5점
시간: 3-5시간 (100 trials)
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GroupKFold
import optuna
from optuna.samplers import TPESampler
import pickle
import warnings
warnings.filterwarnings('ignore')


def euclidean_distance(y_true, y_pred):
    """유클리드 거리 계산"""
    distances = np.sqrt((y_true[:, 0] - y_pred[:, 0])**2 +
                       (y_true[:, 1] - y_pred[:, 1])**2)
    return distances.mean()


class LightGBMOptimizer:
    def __init__(self, X_train, y_train_x, y_train_y, game_ids):
        self.X_train = X_train
        self.y_train_x = y_train_x
        self.y_train_y = y_train_y
        self.game_ids = game_ids
        self.best_score = float('inf')

    def objective(self, trial):
        """Optuna objective function"""

        # 하이퍼파라미터 탐색 공간
        params = {
            'objective': 'regression',
            'metric': 'rmse',
            'verbosity': -1,
            'boosting_type': 'gbdt',

            # 학습률
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),

            # 트리 구조
            'num_leaves': trial.suggest_int('num_leaves', 31, 255),
            'max_depth': trial.suggest_int('max_depth', 5, 15),
            'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 20, 100),

            # 정규화
            'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 10.0, log=True),
            'lambda_l2': trial.suggest_float('lambda_l2', 1e-8, 10.0, log=True),

            # 샘플링
            'feature_fraction': trial.suggest_float('feature_fraction', 0.6, 1.0),
            'bagging_fraction': trial.suggest_float('bagging_fraction', 0.6, 1.0),
            'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),

            # 기타
            'min_gain_to_split': trial.suggest_float('min_gain_to_split', 0.0, 15.0),
        }

        # 5-Fold Cross Validation
        gkf = GroupKFold(n_splits=5)
        fold_scores = []

        for fold, (train_idx, val_idx) in enumerate(gkf.split(self.X_train, groups=self.game_ids)):
            X_tr, X_val = self.X_train.iloc[train_idx], self.X_train.iloc[val_idx]
            y_tr_x, y_val_x = self.y_train_x[train_idx], self.y_train_x[val_idx]
            y_tr_y, y_val_y = self.y_train_y[train_idx], self.y_train_y[val_idx]

            # X 좌표 모델
            dtrain_x = lgb.Dataset(X_tr, label=y_tr_x)
            dvalid_x = lgb.Dataset(X_val, label=y_val_x, reference=dtrain_x)

            model_x = lgb.train(
                params,
                dtrain_x,
                num_boost_round=3000,
                valid_sets=[dvalid_x],
                callbacks=[
                    lgb.early_stopping(stopping_rounds=100, verbose=False),
                    lgb.log_evaluation(0)
                ]
            )

            # Y 좌표 모델
            dtrain_y = lgb.Dataset(X_tr, label=y_tr_y)
            dvalid_y = lgb.Dataset(X_val, label=y_val_y, reference=dtrain_y)

            model_y = lgb.train(
                params,
                dtrain_y,
                num_boost_round=3000,
                valid_sets=[dvalid_y],
                callbacks=[
                    lgb.early_stopping(stopping_rounds=100, verbose=False),
                    lgb.log_evaluation(0)
                ]
            )

            # 검증
            pred_x = model_x.predict(X_val, num_iteration=model_x.best_iteration)
            pred_y = model_y.predict(X_val, num_iteration=model_y.best_iteration)
            y_pred = np.column_stack([pred_x, pred_y])
            y_val = np.column_stack([y_val_x, y_val_y])

            eucl_dist = euclidean_distance(y_val, y_pred)
            fold_scores.append(eucl_dist)

        mean_score = np.mean(fold_scores)

        # Best score 업데이트
        if mean_score < self.best_score:
            self.best_score = mean_score
            print(f"\n🎯 New Best Score: {mean_score:.4f}m")
            print(f"   Params: {params}")

        return mean_score


def main():
    print("=" * 80)
    print("  LightGBM V4 - Optuna 하이퍼파라미터 최적화")
    print("  목표: 0.2-0.5점 추가 개선")
    print("=" * 80)
    print()

    # 1. 데이터 로딩
    print("📊 데이터 로딩...")
    data = pd.read_csv('processed_train_data_v4.csv')
    print(f"데이터: {data.shape}\n")

    # 2. 피처/타겟 분리
    print("📊 피처/타겟 분리...")

    y_train_x = data['target_x'].values
    y_train_y = data['target_y'].values
    game_ids = data['game_id'].values

    drop_cols = ['game_episode', 'game_id', 'target_x', 'target_y', 'final_team_id']
    X_train = data.drop(columns=[c for c in drop_cols if c in data.columns])
    X_train = X_train.fillna(0)

    for col in X_train.columns:
        if X_train[col].dtype == 'object':
            X_train[col] = pd.to_numeric(X_train[col], errors='coerce').fillna(0)

    print(f"피처 수: {X_train.shape[1]}")
    print(f"샘플 수: {len(X_train):,}\n")

    # 3. Optuna 최적화
    n_trials = 20  # 빠른 테스트용 (좋은 결과면 100으로 확장)
    print("🔧 Optuna 하이퍼파라미터 최적화 시작...")
    print(f"   - Trials: {n_trials} (빠른 테스트)")
    print("   - 5-Fold CV")
    print("   - 예상 시간: 30-60분\n")

    optimizer = LightGBMOptimizer(X_train, y_train_x, y_train_y, game_ids)

    study = optuna.create_study(
        direction='minimize',
        sampler=TPESampler(seed=42),
        study_name='lightgbm_v4_optimization'
    )

    study.optimize(
        optimizer.objective,
        n_trials=n_trials,
        timeout=None,
        show_progress_bar=True
    )

    # 4. 최적 파라미터 출력
    print("\n" + "=" * 80)
    print("  최적화 완료!")
    print("=" * 80)

    best_params = study.best_params
    best_score = study.best_value

    print(f"\n🏆 Best Score: {best_score:.4f}m")
    print(f"\n📊 Best Parameters:")
    for key, value in best_params.items():
        print(f"   {key:20s}: {value}")

    # 5. 개선도 분석
    baseline_score = 14.36  # V4 baseline
    improvement = baseline_score - best_score
    improvement_pct = (improvement / baseline_score) * 100

    print(f"\n📈 개선도 분석:")
    print(f"   Baseline (V4):     {baseline_score:.4f}m")
    print(f"   Optimized (V4.1):  {best_score:.4f}m")
    print(f"   Improvement:       {improvement:.4f}m ({improvement_pct:.2f}%)")

    if improvement > 0.2:
        print("\n🎉 우수한 개선! 즉시 V4.1로 재학습 권장")
    elif improvement > 0.1:
        print("\n✅ 의미 있는 개선! V4.1 재학습 고려")
    else:
        print("\n📊 소폭 개선. 다른 전략 병행 권장")

    # 6. 최적 파라미터 저장
    best_params_full = {
        'objective': 'regression',
        'metric': 'rmse',
        'verbosity': -1,
        **best_params
    }

    with open('best_params_v4_optuna.pkl', 'wb') as f:
        pickle.dump({
            'params': best_params_full,
            'score': best_score,
            'improvement': improvement,
            'study': study
        }, f)

    print(f"\n💾 최적 파라미터 저장: best_params_v4_optuna.pkl")

    # 7. Optuna 시각화 정보
    print("\n" + "=" * 80)
    print("  Optuna 분석")
    print("=" * 80)

    print(f"\n총 시도 횟수: {len(study.trials)}")
    print(f"완료된 시도: {len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])}")
    print(f"실패한 시도: {len([t for t in study.trials if t.state == optuna.trial.TrialState.FAIL])}")

    # 중요 파라미터 분석
    print("\n📊 파라미터 중요도 (추정):")
    try:
        importances = optuna.importance.get_param_importances(study)
        for i, (param, importance) in enumerate(sorted(importances.items(), key=lambda x: x[1], reverse=True)[:10], 1):
            print(f"   {i:2d}. {param:20s}: {importance:.4f}")
    except:
        print("   파라미터 중요도 계산 실패 (trials 부족)")

    # 8. 다음 단계 안내
    print("\n" + "=" * 80)
    print("  다음 단계")
    print("=" * 80)

    print("\n1. 최적 파라미터로 V4.1 재학습:")
    print("   python train_lightgbm_v4_with_best_params.py")

    print("\n2. Test 추론 및 제출:")
    print("   python inference_v4.py")

    print("\n3. 다른 최적화 전략:")
    print("   - K 값 최적화 (K=15,25,30)")
    print("   - Feature Selection")
    print("   - XGBoost/CatBoost 실험")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()

