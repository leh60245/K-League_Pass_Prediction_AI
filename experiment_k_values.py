"""
K 값 최적화 실험

목표: 마지막 K개 이벤트의 최적 개수 찾기
후보: K = [15, 20, 25, 30]
예상 개선: 0.1-0.3점
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GroupKFold
import pickle
import warnings
warnings.filterwarnings('ignore')

# 기존 preprocessing_v4 import
import sys
sys.path.append('.')
from preprocessing_v4 import DataPreprocessorV4


def euclidean_distance(y_true, y_pred):
    """유클리드 거리 계산"""
    distances = np.sqrt((y_true[:, 0] - y_pred[:, 0])**2 +
                       (y_true[:, 1] - y_pred[:, 1])**2)
    return distances.mean()


def quick_train_eval(X_train, y_train_x, y_train_y, game_ids, k_value):
    """빠른 학습 및 평가 (1-Fold만)"""

    print(f"\n{'='*60}")
    print(f"  K = {k_value} 테스트")
    print(f"{'='*60}")

    # 하이퍼파라미터 (V4 기본값)
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'learning_rate': 0.05,
        'num_leaves': 127,
        'min_data_in_leaf': 80,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 1,
        'verbose': -1,
    }

    # 1-Fold만 사용 (빠른 평가)
    gkf = GroupKFold(n_splits=5)
    train_idx, val_idx = next(gkf.split(X_train, groups=game_ids))

    X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
    y_tr_x, y_val_x = y_train_x[train_idx], y_train_x[val_idx]
    y_tr_y, y_val_y = y_train_y[train_idx], y_train_y[val_idx]

    print(f"Train: {len(X_tr):,}, Val: {len(X_val):,}")

    # X 좌표 모델
    print("end_x 모델 학습 중...")
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
    print(f"  -> {model_x.best_iteration} rounds")

    # Y 좌표 모델
    print("end_y 모델 학습 중...")
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
    print(f"  -> {model_y.best_iteration} rounds")

    # 평가
    pred_x = model_x.predict(X_val, num_iteration=model_x.best_iteration)
    pred_y = model_y.predict(X_val, num_iteration=model_y.best_iteration)
    y_pred = np.column_stack([pred_x, pred_y])
    y_val = np.column_stack([y_val_x, y_val_y])

    eucl_dist = euclidean_distance(y_val, y_pred)
    mse_x = mean_squared_error(y_val_x, pred_x)
    mse_y = mean_squared_error(y_val_y, pred_y)

    print(f"\n결과:")
    print(f"  - 유클리드 거리: {eucl_dist:.4f}m")
    print(f"  - MSE X: {mse_x:.4f}")
    print(f"  - MSE Y: {mse_y:.4f}")

    return {
        'k': k_value,
        'euclidean': eucl_dist,
        'mse_x': mse_x,
        'mse_y': mse_y,
        'best_iter_x': model_x.best_iteration,
        'best_iter_y': model_y.best_iteration
    }


def full_train_eval(k_value, n_folds=5):
    """전체 5-Fold 학습 및 평가 (최종 검증용)"""

    print(f"\n{'='*80}")
    print(f"  K = {k_value} 전체 5-Fold 학습")
    print(f"{'='*80}\n")

    # 전처리 실행
    preprocessor = DataPreprocessorV4(data_dir='./data', K=k_value)
    X_train, X_test = preprocessor.preprocess_pipeline(verbose=True)

    # 데이터 저장
    X_train.to_csv(f'processed_train_data_v4_k{k_value}.csv', index=False)
    X_test.to_csv(f'processed_test_data_v4_k{k_value}.csv', index=False)

    # 학습
    print(f"\n5-Fold 학습 시작...")

    y_train_x = X_train['target_x'].values
    y_train_y = X_train['target_y'].values
    game_ids = X_train['game_id'].values

    drop_cols = ['game_episode', 'game_id', 'target_x', 'target_y', 'final_team_id']
    X_train_feat = X_train.drop(columns=[c for c in drop_cols if c in X_train.columns])
    X_train_feat = X_train_feat.fillna(0)

    for col in X_train_feat.columns:
        if X_train_feat[col].dtype == 'object':
            X_train_feat[col] = pd.to_numeric(X_train_feat[col], errors='coerce').fillna(0)

    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'learning_rate': 0.05,
        'num_leaves': 127,
        'min_data_in_leaf': 80,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 1,
        'verbose': -1,
    }

    gkf = GroupKFold(n_splits=n_folds)
    fold_scores = []

    for fold, (train_idx, val_idx) in enumerate(gkf.split(X_train_feat, groups=game_ids)):
        print(f"\nFold {fold+1}/{n_folds}")

        X_tr, X_val = X_train_feat.iloc[train_idx], X_train_feat.iloc[val_idx]
        y_tr_x, y_val_x = y_train_x[train_idx], y_train_x[val_idx]
        y_tr_y, y_val_y = y_train_y[train_idx], y_train_y[val_idx]

        # X 모델
        dtrain_x = lgb.Dataset(X_tr, label=y_tr_x)
        dvalid_x = lgb.Dataset(X_val, label=y_val_x, reference=dtrain_x)
        model_x = lgb.train(params, dtrain_x, num_boost_round=3000,
                           valid_sets=[dvalid_x],
                           callbacks=[lgb.early_stopping(100, verbose=False)])

        # Y 모델
        dtrain_y = lgb.Dataset(X_tr, label=y_tr_y)
        dvalid_y = lgb.Dataset(X_val, label=y_val_y, reference=dtrain_y)
        model_y = lgb.train(params, dtrain_y, num_boost_round=3000,
                           valid_sets=[dvalid_y],
                           callbacks=[lgb.early_stopping(100, verbose=False)])

        # 평가
        pred_x = model_x.predict(X_val, num_iteration=model_x.best_iteration)
        pred_y = model_y.predict(X_val, num_iteration=model_y.best_iteration)
        y_pred = np.column_stack([pred_x, pred_y])
        y_val = np.column_stack([y_val_x, y_val_y])

        eucl_dist = euclidean_distance(y_val, y_pred)
        fold_scores.append(eucl_dist)
        print(f"  Score: {eucl_dist:.4f}m")

    mean_score = np.mean(fold_scores)
    std_score = np.std(fold_scores)

    print(f"\n평균: {mean_score:.4f}m ± {std_score:.4f}m")

    return mean_score, std_score


def main():
    print("=" * 80)
    print("  K 값 최적화 실험")
    print("  목표: 최적 시퀀스 길이 찾기")
    print("=" * 80)
    print()

    # Phase 1: Quick Test (1-Fold)
    print("📊 Phase 1: Quick Test (1-Fold)")
    print("   각 K 값으로 빠르게 테스트하여 경향 파악\n")

    k_candidates = [15, 20, 25, 30]
    quick_results = []

    # K=20 데이터 로드 (이미 생성됨)
    print("K=20 데이터 로딩 (기존)...")
    data_k20 = pd.read_csv('processed_train_data_v4.csv')

    y_train_x = data_k20['target_x'].values
    y_train_y = data_k20['target_y'].values
    game_ids = data_k20['game_id'].values

    drop_cols = ['game_episode', 'game_id', 'target_x', 'target_y', 'final_team_id']
    X_train = data_k20.drop(columns=[c for c in drop_cols if c in data_k20.columns])
    X_train = X_train.fillna(0)

    for col in X_train.columns:
        if X_train[col].dtype == 'object':
            X_train[col] = pd.to_numeric(X_train[col], errors='coerce').fillna(0)

    # K=20으로 Quick Test
    result_k20 = quick_train_eval(X_train, y_train_x, y_train_y, game_ids, 20)
    quick_results.append(result_k20)

    # 다른 K 값들은 전처리부터 필요
    print("\n⚠️  다른 K 값 테스트를 위해서는 전처리가 필요합니다.")
    print("   각 K 값마다 2-3분 소요 예상\n")

    for k in [15, 25, 30]:
        response = input(f"K={k} 테스트를 진행하시겠습니까? (y/n): ")
        if response.lower() != 'y':
            print(f"K={k} 건너뜀")
            continue

        print(f"\nK={k} 전처리 시작...")
        preprocessor = DataPreprocessorV4(data_dir='./data', K=k)
        X_train_k, _ = preprocessor.preprocess_pipeline(verbose=False)

        # 피처 준비
        y_train_x = X_train_k['target_x'].values
        y_train_y = X_train_k['target_y'].values
        game_ids = X_train_k['game_id'].values

        X_train_feat = X_train_k.drop(columns=[c for c in drop_cols if c in X_train_k.columns])
        X_train_feat = X_train_feat.fillna(0)

        for col in X_train_feat.columns:
            if X_train_feat[col].dtype == 'object':
                X_train_feat[col] = pd.to_numeric(X_train_feat[col], errors='coerce').fillna(0)

        # Quick Test
        result = quick_train_eval(X_train_feat, y_train_x, y_train_y, game_ids, k)
        quick_results.append(result)

    # Phase 1 결과 요약
    print("\n" + "=" * 80)
    print("  Phase 1 결과 요약 (Quick Test)")
    print("=" * 80)

    results_df = pd.DataFrame(quick_results)
    results_df = results_df.sort_values('euclidean')

    print("\n순위 (유클리드 거리 기준):")
    for i, row in results_df.iterrows():
        rank = results_df.index.get_loc(i) + 1
        print(f"  {rank}. K={row['k']:2d}: {row['euclidean']:.4f}m")

    best_k = results_df.iloc[0]['k']
    best_score = results_df.iloc[0]['euclidean']

    print(f"\n🏆 Best K: {int(best_k)} (Score: {best_score:.4f}m)")

    # K=20 대비 개선도
    k20_score = results_df[results_df['k'] == 20]['euclidean'].values[0]
    improvement = k20_score - best_score

    print(f"\nK=20 대비:")
    print(f"  - K=20: {k20_score:.4f}m")
    print(f"  - K={int(best_k)}: {best_score:.4f}m")
    print(f"  - 개선: {improvement:.4f}m")

    # Phase 2: Full Test (선택)
    if improvement > 0.05:  # 의미 있는 개선이 있는 경우
        print("\n" + "=" * 80)
        print("  Phase 2: Full Test 권장")
        print("=" * 80)
        print(f"\nK={int(best_k)}가 K=20보다 {improvement:.4f}m 우수합니다.")
        print("5-Fold 전체 학습으로 검증하시겠습니까?")
        print("(예상 시간: 20-30분)\n")

        response = input("진행하시겠습니까? (y/n): ")
        if response.lower() == 'y':
            mean_score, std_score = full_train_eval(int(best_k), n_folds=5)

            print("\n" + "=" * 80)
            print("  최종 결과")
            print("=" * 80)
            print(f"\nK={int(best_k)} 5-Fold 성능: {mean_score:.4f}m ± {std_score:.4f}m")

            # V4 baseline과 비교
            v4_baseline = 14.36
            final_improvement = v4_baseline - mean_score

            print(f"\nV4 Baseline 대비:")
            print(f"  - V4 (K=20): {v4_baseline:.4f}m")
            print(f"  - V4.2 (K={int(best_k)}): {mean_score:.4f}m")
            print(f"  - 개선: {final_improvement:.4f}m")

            if final_improvement > 0.1:
                print("\n🎉 우수한 개선! V4.2로 업데이트 권장")
            elif final_improvement > 0:
                print("\n✅ 소폭 개선! 상황에 따라 선택")
            else:
                print("\n📊 K=20 유지 권장")
    else:
        print("\n📊 결론: K=20이 최적이거나 큰 차이 없음")
        print("   다른 최적화 전략 (하이퍼파라미터, 피처 등) 추천")

    # 결과 저장
    results_df.to_csv('k_optimization_results.csv', index=False)
    print(f"\n💾 결과 저장: k_optimization_results.csv")

    # 다음 단계 안내
    print("\n" + "=" * 80)
    print("  다음 단계")
    print("=" * 80)

    print("\n1. 하이퍼파라미터 최적화:")
    print("   python train_lightgbm_v4_optuna.py")

    print("\n2. Feature Selection:")
    print("   Feature Importance 기반 상위 피처 선택")

    print("\n3. 다른 모델 실험:")
    print("   XGBoost, CatBoost 구현")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()

