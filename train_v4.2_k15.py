"""V4.2 Training with K=15"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
import pickle
import sys
sys.path.append('.')
from preprocessing_v4 import DataPreprocessorV4
import warnings
warnings.filterwarnings('ignore')


def euclidean_distance(y_true, y_pred):
    distances = np.sqrt((y_true[:, 0] - y_pred[:, 0])**2 +
                       (y_true[:, 1] - y_pred[:, 1])**2)
    return distances.mean()


print("=" * 80)
print("  V4.2 - K=15 Full 5-Fold Training")
print("  Quick Test: K=15 best (14.1588m)")
print("  Expected: 13.9-14.0 Test")
print("=" * 80)
print()

# Preprocess K=15
print("Preprocessing with K=15...")
preprocessor = DataPreprocessorV4(data_dir='./data', K=15)
X_train, X_test = preprocessor.preprocess_pipeline(verbose=True)

X_train.to_csv('processed_train_data_v4.2_k15.csv', index=False)
X_test.to_csv('processed_test_data_v4.2_k15.csv', index=False)
print("\nSaved processed data\n")

# Prepare
print("Preparing features...")
y_train_x = X_train['target_x'].values
y_train_y = X_train['target_y'].values
game_ids = X_train['game_id'].values

drop_cols = ['game_episode', 'game_id', 'target_x', 'target_y', 'final_team_id']
X_feat = X_train.drop(columns=[c for c in drop_cols if c in X_train.columns])
X_feat = X_feat.fillna(0)

for col in X_feat.columns:
    if X_feat[col].dtype == 'object':
        X_feat[col] = pd.to_numeric(X_feat[col], errors='coerce').fillna(0)

print(f"Features: {X_feat.shape[1]} (K=20: 775, K=15: {X_feat.shape[1]})")
print(f"Samples: {len(X_feat):,}\n")

# Parameters
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

# Train
print("Starting 5-Fold training...\n")

gkf = GroupKFold(n_splits=5)
models_x = []
models_y = []
fold_scores = []

for fold, (train_idx, val_idx) in enumerate(gkf.split(X_feat, groups=game_ids)):
    print(f"{'='*60}")
    print(f"  Fold {fold+1}/5")
    print(f"{'='*60}")

    X_tr, X_val = X_feat.iloc[train_idx], X_feat.iloc[val_idx]
    y_tr_x, y_val_x = y_train_x[train_idx], y_train_x[val_idx]
    y_tr_y, y_val_y = y_train_y[train_idx], y_train_y[val_idx]

    print(f"Train: {len(X_tr):,}, Val: {len(X_val):,}")

    # X
    print("Training end_x...")
    dtrain_x = lgb.Dataset(X_tr, label=y_tr_x)
    dvalid_x = lgb.Dataset(X_val, label=y_val_x, reference=dtrain_x)

    model_x = lgb.train(
        params, dtrain_x, num_boost_round=5000,
        valid_sets=[dtrain_x, dvalid_x],
        valid_names=['train', 'valid'],
        callbacks=[lgb.early_stopping(150, verbose=False)]
    )
    models_x.append(model_x)
    print(f"  -> {model_x.best_iteration} rounds")

    # Y
    print("Training end_y...")
    dtrain_y = lgb.Dataset(X_tr, label=y_tr_y)
    dvalid_y = lgb.Dataset(X_val, label=y_val_y, reference=dtrain_y)

    model_y = lgb.train(
        params, dtrain_y, num_boost_round=5000,
        valid_sets=[dtrain_y, dvalid_y],
        valid_names=['train', 'valid'],
        callbacks=[lgb.early_stopping(150, verbose=False)]
    )
    models_y.append(model_y)
    print(f"  -> {model_y.best_iteration} rounds")

    # Eval
    pred_x = model_x.predict(X_val, num_iteration=model_x.best_iteration)
    pred_y = model_y.predict(X_val, num_iteration=model_y.best_iteration)

    score = euclidean_distance(
        np.column_stack([y_val_x, y_val_y]),
        np.column_stack([pred_x, pred_y])
    )

    print(f"\nFold {fold+1}: {score:.4f}m\n")
    fold_scores.append({'fold': fold+1, 'euclidean': score})

# Results
print("="*80)
print("  Results")
print("="*80)

scores_df = pd.DataFrame(fold_scores)
mean_score = scores_df['euclidean'].mean()
std_score = scores_df['euclidean'].std()

print(f"\nMean: {mean_score:.4f}m +/- {std_score:.4f}m")
for _, row in scores_df.iterrows():
    print(f"  Fold {int(row['fold'])}: {row['euclidean']:.4f}m")

# Compare
print("\n" + "="*80)
print("  Comparison")
print("="*80)

v41_val = 14.365
improvement = v41_val - mean_score

print(f"\nV4.1 (K=20): {v41_val:.4f}m (Validation)")
print(f"V4.2 (K=15): {mean_score:.4f}m (Validation)")
print(f"Improvement: {improvement:.4f}m ({improvement/v41_val*100:.2f}%)")

if mean_score < 14.2:
    print("\nExcellent! Expected Test: 13.9-14.1")
elif mean_score < 14.3:
    print("\nGood! Expected Test: 14.0-14.2")

# Save
print("\nSaving model...")
with open('lightgbm_model_v4.2_k15_5fold.pkl', 'wb') as f:
    pickle.dump({
        'models_x': models_x,
        'models_y': models_y,
        'val_score': mean_score,
        'fold_scores': fold_scores,
        'params': params,
        'K': 15
    }, f)
print("Saved: lightgbm_model_v4.2_k15_5fold.pkl")

# Decision
print("\n" + "="*80)
if mean_score < 14.25:
    print("Recommendation: Use V4.2 (K=15)")
    print("Next: python inference_v4.2_k15.py")
else:
    print("Keep V4.1, proceed to XGBoost/CatBoost")

print("\nV4.1 Test: 14.138")
print(f"V4.2 Expected: {mean_score - 0.06:.2f}-{mean_score + 0.06:.2f}")
print("="*80)

