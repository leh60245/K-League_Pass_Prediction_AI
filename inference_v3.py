"""
Test 추론 - V3 (5-Fold 앙상블)

목표: V3 모델로 Test 데이터 예측 및 제출 파일 생성
"""

import pandas as pd
import numpy as np
import pickle
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


def main():
    print("=" * 80)
    print("  Test 추론 - V3 (5-Fold 앙상블)")
    print("=" * 80)
    print()

    # 1. 모델 로딩
    print("모델 로딩 중...")
    with open('lightgbm_model_v3_5fold.pkl', 'rb') as f:
        saved = pickle.load(f)
        models_x = saved['models_x']
        models_y = saved['models_y']
    print(f"✅ {len(models_x)}개 Fold 모델 로딩 완료\n")

    # 2. Test 데이터 로딩
    print("Test 데이터 로딩 중...")
    X_test = pd.read_csv('processed_test_data_v3.csv')
    print(f"✅ Test 데이터: {len(X_test):,} 에피소드\n")

    # 3. 피처 준비
    print("피처 준비 중...")
    game_episodes = X_test['game_episode'].values

    drop_cols = ['game_episode', 'game_id', 'target_x', 'target_y']
    X_test_feat = X_test.drop(columns=[c for c in drop_cols if c in X_test.columns])
    X_test_feat = X_test_feat.fillna(0)

    # 데이터 타입 변환
    for col in X_test_feat.columns:
        if X_test_feat[col].dtype == 'object':
            X_test_feat[col] = pd.to_numeric(X_test_feat[col], errors='coerce').fillna(0)

    print(f"✅ 피처 수: {X_test_feat.shape[1]}\n")

    # 4. 앙상블 예측
    print("5-Fold 앙상블 예측 중...")
    pred_x_folds = []
    pred_y_folds = []

    for fold, (model_x, model_y) in enumerate(zip(models_x, models_y)):
        print(f"  Fold {fold+1}/{len(models_x)} 예측 중...")
        pred_x_folds.append(model_x.predict(X_test_feat, num_iteration=model_x.best_iteration))
        pred_y_folds.append(model_y.predict(X_test_feat, num_iteration=model_y.best_iteration))

    # 앙상블 평균
    pred_x = np.mean(pred_x_folds, axis=0)
    pred_y = np.mean(pred_y_folds, axis=0)

    # 필드 범위로 클립
    pred_x = np.clip(pred_x, 0, 105)
    pred_y = np.clip(pred_y, 0, 68)

    print("✅ 앙상블 예측 완료\n")

    # 5. 제출 파일 생성
    print("제출 파일 생성 중...")

    sample_sub_path = os.path.join('data', 'sample_submission.csv')
    sample_sub = pd.read_csv(sample_sub_path)

    pred_df = pd.DataFrame({
        'game_episode': game_episodes,
        'end_x': pred_x,
        'end_y': pred_y
    })

    submission = sample_sub.drop(columns=['end_x', 'end_y'], errors='ignore')
    submission = submission.merge(pred_df, on='game_episode', how='left')

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_path = f'submission_v3_5fold_{timestamp}.csv'
    submission.to_csv(output_path, index=False)

    print(f"✅ 제출 파일 저장: {output_path}")
    print()

    # 6. 통계 출력
    print("=" * 80)
    print("  예측 통계")
    print("=" * 80)
    print(f"예측 개수: {len(pred_x):,}")
    print(f"\nend_x 통계:")
    print(f"  평균: {pred_x.mean():.2f}")
    print(f"  표준편차: {pred_x.std():.2f}")
    print(f"  범위: [{pred_x.min():.2f}, {pred_x.max():.2f}]")
    print(f"\nend_y 통계:")
    print(f"  평균: {pred_y.mean():.2f}")
    print(f"  표준편차: {pred_y.std():.2f}")
    print(f"  범위: [{pred_y.min():.2f}, {pred_y.max():.2f}]")
    print("=" * 80)


if __name__ == "__main__":
    main()

