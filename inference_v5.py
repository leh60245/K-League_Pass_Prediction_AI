"""
LightGBM V5 - Test 추론
"""

import pandas as pd
import numpy as np
import pickle
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


def main():
    print("=" * 80)
    print("  LightGBM V5 - Test 추론")
    print("=" * 80)
    print()

    # 1. 데이터 로딩
    print("📊 Test 데이터 로딩...")
    X_test = pd.read_csv('processed_test_data_v5.csv')
    print(f"Test: {X_test.shape}\n")

    # 2. 모델 로딩
    print("🔧 V5 모델 로딩...")
    with open('lightgbm_model_v5_5fold.pkl', 'rb') as f:
        model_data = pickle.load(f)

    models_x = model_data['models_x']
    models_y = model_data['models_y']
    val_score = model_data['val_score']

    print(f"✅ 로딩 완료")
    print(f"   Validation: {val_score:.4f}m\n")

    # 3. 피처 준비
    print("📊 피처 준비...")
    drop_cols = ['game_episode', 'game_id', 'target_x', 'target_y', 'final_team_id']
    test_episodes = X_test['game_episode'].copy()

    X_test_feat = X_test.drop(columns=[c for c in drop_cols if c in X_test.columns])
    X_test_feat = X_test_feat.fillna(0)

    # for col in X_test_feat.columns:
    #     if X_test_feat[col].dtype == 'object':
    #         X_test_feat[col] = pd.to_numeric(X_test_feat[col], errors='coerce').fillna(0)

    print(f"✅ 준비 완료: {X_test_feat.shape}\n")

    # 4. 예측
    print("🔮 5-Fold 앙상블 예측...")

    pred_x_folds = []
    pred_y_folds = []

    for fold, (model_x, model_y) in enumerate(zip(models_x, models_y)):
        print(f"  Fold {fold+1}")
        pred_x = model_x.predict(X_test_feat, num_iteration=model_x.best_iteration)
        pred_y = model_y.predict(X_test_feat, num_iteration=model_y.best_iteration)
        pred_x_folds.append(pred_x)
        pred_y_folds.append(pred_y)

    pred_x = np.mean(pred_x_folds, axis=0)
    pred_y = np.mean(pred_y_folds, axis=0)

    pred_x = np.clip(pred_x, 0, 105)
    pred_y = np.clip(pred_y, 0, 68)

    print(f"✅ 예측 완료\n")

    # 5. 제출 파일
    print("📝 제출 파일 생성...")

    submission = pd.DataFrame({
        'game_episode': test_episodes,
        'end_x': pred_x,
        'end_y': pred_y
    })

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'submission_v5_{timestamp}.csv'
    submission.to_csv(filename, index=False)

    print(f"✅ 저장: {filename}\n")

    # 6. 요약
    print("="*80)
    print("  완료!")
    print("="*80)
    print(f"\n제출 파일: {filename}")
    print(f"Validation: {val_score:.4f}m")
    print(f"예상 Test: 14.0~14.2점")

    print("\n성능 비교:")
    print("  V3:  14.535점")
    print("  V4:  14.308점")
    print("  V4.1: 예상 14.138점")

    print("\n" + "="*80)


if __name__ == "__main__":
    main()

