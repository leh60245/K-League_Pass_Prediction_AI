"""
LightGBM V4 - Test 추론 및 제출 파일 생성

V2의 풍부한 피처 + V3의 시퀀스 모델링
5-Fold 앙상블 예측
"""

import pandas as pd
import numpy as np
import pickle
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


def main():
    print("=" * 80)
    print("  LightGBM V4 - Test 추론")
    print("  V2 피처 + V3 시퀀스 모델링")
    print("=" * 80)
    print()

    # 1. Test 데이터 로딩
    print("📊 Test 데이터 로딩...")
    X_test = pd.read_csv('processed_test_data_v4.csv')
    print(f"Test 데이터: {X_test.shape}\n")

    # 2. 모델 로딩
    print("🔧 모델 로딩...")
    with open('lightgbm_model_v4_5fold.pkl', 'rb') as f:
        model_data = pickle.load(f)

    models_x = model_data['models_x']
    models_y = model_data['models_y']
    val_score = model_data['val_score']

    print(f"✅ 5-Fold 모델 로딩 완료")
    print(f"   - Validation 성능: {val_score:.4f}m\n")

    # 3. 피처 준비
    print("📊 피처 준비...")

    # 불필요한 컬럼 제거
    drop_cols = ['game_episode', 'game_id', 'target_x', 'target_y', 'final_team_id']

    # game_episode 저장 (제출 파일용)
    test_episodes = X_test['game_episode'].copy()

    # 피처만 추출
    X_test_feat = X_test.drop(columns=[c for c in drop_cols if c in X_test.columns])

    # NaN 채우기
    X_test_feat = X_test_feat.fillna(0)

    # 데이터 타입 변환
    for col in X_test_feat.columns:
        if X_test_feat[col].dtype == 'object':
            X_test_feat[col] = pd.to_numeric(X_test_feat[col], errors='coerce').fillna(0)

    print(f"✅ 피처 준비 완료: {X_test_feat.shape}\n")

    # 4. 5-Fold 앙상블 예측
    print("🔮 5-Fold 앙상블 예측 중...")

    pred_x_folds = []
    pred_y_folds = []

    for fold, (model_x, model_y) in enumerate(zip(models_x, models_y)):
        print(f"  Fold {fold+1} 예측 중...")
        pred_x = model_x.predict(X_test_feat, num_iteration=model_x.best_iteration)
        pred_y = model_y.predict(X_test_feat, num_iteration=model_y.best_iteration)
        pred_x_folds.append(pred_x)
        pred_y_folds.append(pred_y)

    # 앙상블 평균
    pred_x = np.mean(pred_x_folds, axis=0)
    pred_y = np.mean(pred_y_folds, axis=0)

    # 필드 범위로 클립
    pred_x = np.clip(pred_x, 0, 105)
    pred_y = np.clip(pred_y, 0, 68)

    print(f"✅ 예측 완료\n")

    # 5. 제출 파일 생성
    print("📝 제출 파일 생성 중...")

    submission = pd.DataFrame({
        'game_episode': test_episodes,
        'end_x': pred_x,
        'end_y': pred_y
    })

    # 타임스탬프
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'submission_v4_5fold_{timestamp}.csv'

    submission.to_csv(filename, index=False)
    print(f"✅ 제출 파일 저장: {filename}\n")

    # 6. 예측 통계
    print("=" * 80)
    print("  예측 통계")
    print("=" * 80)

    print(f"\nend_x 통계:")
    print(f"  - 평균: {pred_x.mean():.2f}")
    print(f"  - 표준편차: {pred_x.std():.2f}")
    print(f"  - 범위: [{pred_x.min():.2f}, {pred_x.max():.2f}]")

    print(f"\nend_y 통계:")
    print(f"  - 평균: {pred_y.mean():.2f}")
    print(f"  - 표준편차: {pred_y.std():.2f}")
    print(f"  - 범위: [{pred_y.min():.2f}, {pred_y.max():.2f}]")

    # 7. Fold간 예측 일관성
    print("\n" + "=" * 80)
    print("  Fold간 예측 일관성")
    print("=" * 80)

    # 각 Fold 예측의 표준편차 (불확실성)
    pred_x_std = np.std(pred_x_folds, axis=0)
    pred_y_std = np.std(pred_y_folds, axis=0)

    print(f"\nFold간 예측 표준편차 (불확실성):")
    print(f"  - end_x: {pred_x_std.mean():.4f}m (평균)")
    print(f"  - end_y: {pred_y_std.mean():.4f}m (평균)")
    print(f"  - 총 불확실성: {np.sqrt(pred_x_std**2 + pred_y_std**2).mean():.4f}m")

    if np.sqrt(pred_x_std**2 + pred_y_std**2).mean() < 0.5:
        print("\n✅ Fold간 예측이 매우 일관적입니다 (안정적)")
    elif np.sqrt(pred_x_std**2 + pred_y_std**2).mean() < 1.0:
        print("\n✅ Fold간 예측이 일관적입니다")
    else:
        print("\n⚠️  Fold간 예측 차이가 다소 큽니다")

    # 8. 최종 요약
    print("\n" + "=" * 80)
    print("  최종 요약")
    print("=" * 80)

    print(f"\n✅ 제출 파일: {filename}")
    print(f"✅ 예측 샘플 수: {len(submission)}")
    print(f"✅ Validation 성능: {val_score:.4f}m")
    print(f"✅ 예상 Test 성능: 13~15점대")

    print("\n📊 성능 비교 (기대):")
    print("   - V1: 24점대")
    print("   - V3: 14점대")
    print("   - V4: 13~15점대 (V2 피처 + V3 시퀀스)")

    print("\n" + "=" * 80)
    print("다음 단계:")
    print("   1. 제출 파일을 대회 사이트에 업로드")
    print("   2. Test 점수 확인")
    print("   3. V3와 V4 성능 비교")
    print("   4. 필요시 하이퍼파라미터 튜닝")
    print("=" * 80)

    return submission


if __name__ == "__main__":
    submission = main()

