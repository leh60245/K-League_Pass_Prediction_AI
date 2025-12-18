"""
K Value Optimization for V4.1

Test K = [15, 20, 25, 30] to find optimal sequence length
Quick Test: 1-Fold per K
Expected improvement: 0.1-0.3 points
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_squared_error
import sys
sys.path.append('.')
from preprocessing_v4 import DataPreprocessorV4
import warnings
warnings.filterwarnings('ignore')


def euclidean_distance(y_true, y_pred):
    distances = np.sqrt((y_true[:, 0] - y_pred[:, 0])**2 +
                       (y_true[:, 1] - y_pred[:, 1])**2)
    return distances.mean()


def quick_test_k(k_value, params):
    """Quick 1-Fold test for given K"""
    print(f"\n{'='*60}")
    print(f"  Testing K = {k_value}")
    print(f"{'='*60}")

    # Preprocess with K
    print(f"Preprocessing with K={k_value}...")
    preprocessor = DataPreprocessorV4(data_dir='./data', K=k_value)
    X_train, _ = preprocessor.preprocess_pipeline(verbose=False)

    # Prepare data
    y_train_x = X_train['target_x'].values
    y_train_y = X_train['target_y'].values
    game_ids = X_train['game_id'].values

    drop_cols = ['game_episode', 'game_id', 'target_x', 'target_y', 'final_team_id']
    X_feat = X_train.drop(columns=[c for c in drop_cols if c in X_train.columns])
    X_feat = X_feat.fillna(0)

    for col in X_feat.columns:
        if X_feat[col].dtype == 'object':
            X_feat[col] = pd.to_numeric(X_feat[col], errors='coerce').fillna(0)

    # 1-Fold test
    gkf = GroupKFold(n_splits=5)
    train_idx, val_idx = next(gkf.split(X_feat, groups=game_ids))

    X_tr, X_val = X_feat.iloc[train_idx], X_feat.iloc[val_idx]
    y_tr_x, y_val_x = y_train_x[train_idx], y_train_x[val_idx]
    y_tr_y, y_val_y = y_train_y[train_idx], y_train_y[val_idx]

    print(f"Train: {len(X_tr):,}, Val: {len(X_val):,}")

    # Train X
    print("Training end_x...")
    dtrain_x = lgb.Dataset(X_tr, label=y_tr_x)
    dvalid_x = lgb.Dataset(X_val, label=y_val_x, reference=dtrain_x)

    model_x = lgb.train(
        params, dtrain_x, num_boost_round=3000,
        valid_sets=[dvalid_x],
        callbacks=[lgb.early_stopping(100, verbose=False)]
    )
    print(f"  -> {model_x.best_iteration} rounds")

    # Train Y
    print("Training end_y...")
    dtrain_y = lgb.Dataset(X_tr, label=y_tr_y)
    dvalid_y = lgb.Dataset(X_val, label=y_val_y, reference=dtrain_y)

    model_y = lgb.train(
        params, dtrain_y, num_boost_round=3000,
        valid_sets=[dvalid_y],
        callbacks=[lgb.early_stopping(100, verbose=False)]
    )
    print(f"  -> {model_y.best_iteration} rounds")

    # Evaluate
    pred_x = model_x.predict(X_val, num_iteration=model_x.best_iteration)
    pred_y = model_y.predict(X_val, num_iteration=model_y.best_iteration)

    score = euclidean_distance(
        np.column_stack([y_val_x, y_val_y]),
        np.column_stack([pred_x, pred_y])
    )

    print(f"\nK={k_value} Score: {score:.4f}m")

    return {
        'k': k_value,
        'score': score,
        'rounds_x': model_x.best_iteration,
        'rounds_y': model_y.best_iteration,
        'features': X_feat.shape[1]
    }


def main():
    print("=" * 80)
    print("  K Value Optimization (Quick Test)")
    print("  Current: K=20 (V4.1 14.138)")
    print("  Goal: Find optimal K for 0.1-0.3 improvement")
    print("=" * 80)
    print()

    # V4.1 Optuna best parameters
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'verbosity': -1,
        'learning_rate': 0.01389988648190196,
        'num_leaves': 186,
        'max_depth': 8,
        'min_data_in_leaf': 29,
        'lambda_l1': 0.0539460564176539,
        'lambda_l2': 2.0076869308427136e-06,
        'feature_fraction': 0.7521886906472112,
        'bagging_fraction': 0.859189408696891,
        'bagging_freq': 2,
        'min_gain_to_split': 3.490626896293116,
    }

    # Test K values
    k_values = [15, 20, 25, 30]
    results = []

    print("Phase 1: Quick Test (1-Fold per K)")
    print("Estimated time: 30-60 minutes\n")

    for k in k_values:
        try:
            result = quick_test_k(k, params)
            results.append(result)
        except Exception as e:
            print(f"\nError with K={k}: {e}")
            continue

    # Summary
    print("\n" + "="*80)
    print("  Results Summary")
    print("="*80)

    results_df = pd.DataFrame(results).sort_values('score')

    print("\nRanking (by score):")
    for idx, row in results_df.iterrows():
        rank = results_df.index.get_loc(idx) + 1
        print(f"  {rank}. K={int(row['k']):2d}: {row['score']:.4f}m "
              f"(features: {int(row['features'])}, "
              f"rounds: {int(row['rounds_x'])}/{int(row['rounds_y'])})")

    best_k = int(results_df.iloc[0]['k'])
    best_score = results_df.iloc[0]['score']
    baseline_score = results_df[results_df['k'] == 20]['score'].values[0] if 20 in results_df['k'].values else 14.2

    print(f"\nBest K: {best_k}")
    print(f"Best Score: {best_score:.4f}m")
    print(f"Baseline (K=20): {baseline_score:.4f}m")
    print(f"Improvement: {baseline_score - best_score:.4f}m")

    # Recommendation
    print("\n" + "="*80)
    print("  Recommendation")
    print("="*80)

    if abs(best_score - baseline_score) < 0.05:
        print(f"\nK=20 is optimal (difference < 0.05m)")
        print("Recommendation: Keep K=20")
    elif best_k != 20:
        improvement = baseline_score - best_score
        print(f"\nK={best_k} is better than K=20")
        print(f"Improvement: {improvement:.4f}m")

        if improvement > 0.1:
            print("\nRecommendation: Use K={best_k} for V4.2")
            print("Run full 5-Fold training with:")
            print(f"  python train_v4.2_k{best_k}.py")
        else:
            print("\nSmall improvement. Consider:")
            print(f"  1. Full 5-Fold test with K={best_k}")
            print("  2. Or continue with other optimizations")

    # Save results
    results_df.to_csv('k_optimization_results.csv', index=False)
    print(f"\nResults saved: k_optimization_results.csv")

    print("\n" + "="*80)
    print("Next steps:")
    print("  1. If K changed: Full 5-Fold training")
    print("  2. Or proceed to XGBoost/CatBoost")
    print("="*80)


if __name__ == "__main__":
    main()

