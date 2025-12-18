"""
V4.2 (K=15) Test Inference

Generate submission file for V4.2 model
"""

import pandas as pd
import numpy as np
import pickle
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


def main():
    print("=" * 80)
    print("  V4.2 (K=15) - Test Inference")
    print("=" * 80)
    print()

    # Load test data
    print("Loading test data...")
    X_test = pd.read_csv('processed_test_data_v4.2_k15.csv')
    print(f"Test shape: {X_test.shape}\n")

    # Load model
    print("Loading V4.2 model...")
    with open('lightgbm_model_v4.2_k15_5fold.pkl', 'rb') as f:
        model_data = pickle.load(f)

    models_x = model_data['models_x']
    models_y = model_data['models_y']
    val_score = model_data['val_score']

    print(f"Validation: {val_score:.4f}m")
    print(f"K value: {model_data.get('K', 15)}\n")

    # Prepare features
    print("Preparing features...")
    drop_cols = ['game_episode', 'game_id', 'target_x', 'target_y', 'final_team_id']
    test_episodes = X_test['game_episode'].copy()

    X_feat = X_test.drop(columns=[c for c in drop_cols if c in X_test.columns])
    X_feat = X_feat.fillna(0)

    for col in X_feat.columns:
        if X_feat[col].dtype == 'object':
            X_feat[col] = pd.to_numeric(X_feat[col], errors='coerce').fillna(0)

    print(f"Features: {X_feat.shape}\n")

    # Predict
    print("5-Fold ensemble prediction...")

    pred_x_folds = []
    pred_y_folds = []

    for fold, (model_x, model_y) in enumerate(zip(models_x, models_y)):
        print(f"  Fold {fold+1}")
        pred_x = model_x.predict(X_feat, num_iteration=model_x.best_iteration)
        pred_y = model_y.predict(X_feat, num_iteration=model_y.best_iteration)
        pred_x_folds.append(pred_x)
        pred_y_folds.append(pred_y)

    pred_x = np.mean(pred_x_folds, axis=0)
    pred_y = np.mean(pred_y_folds, axis=0)

    pred_x = np.clip(pred_x, 0, 105)
    pred_y = np.clip(pred_y, 0, 68)

    print("Prediction complete!\n")

    # Create submission
    print("Creating submission file...")

    submission = pd.DataFrame({
        'game_episode': test_episodes,
        'end_x': pred_x,
        'end_y': pred_y
    })

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'submission_v4.2_k15_{timestamp}.csv'
    submission.to_csv(filename, index=False)

    print(f"Saved: {filename}\n")

    # Summary
    print("="*80)
    print("  Summary")
    print("="*80)

    print(f"\nSubmission: {filename}")
    print(f"Samples: {len(submission)}")
    print(f"Validation: {val_score:.4f}m")

    print("\nPerformance comparison:")
    print("  V4.1 (K=20): 14.138 (Test)")
    print(f"  V4.2 (K=15): Expected 13.9-14.1 (Test)")

    if val_score < 14.25:
        print("\nExpected improvement: 0.03-0.2 points")

    print("\n" + "="*80)


if __name__ == "__main__":
    main()

