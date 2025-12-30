"""
LightGBM 최적화 모델 추론

작성일: 2025-12-19
"""

import pandas as pd
import numpy as np
import pickle

print("=" * 80)
print("  LightGBM 최적화 모델 추론")
print("=" * 80)
print()


def main():
    # 모델 로드
    print("📦 모델 로딩...")
    with open('lightgbm_optimized_5fold_models.pkl', 'rb') as f:
        model_data = pickle.load(f)

    models_x = model_data['models_x']
    models_y = model_data['models_y']
    avg_score = model_data['avg_score']

    print(f"✅ 5-Fold 모델 로드 완료")
    print(f"   - 평균 Val Score: {avg_score:.4f}m")
    print()

    # Test 데이터 로드
    print("📊 Test 데이터 로딩...")
    test_data = pd.read_csv('processed_test_data_v4.csv')
    print(f"Test 데이터 Shape: {test_data.shape}")
    print()

    game_episode = test_data['game_episode']
    X_test = test_data.drop(columns=['game_episode', 'game_id'])

    # 5-Fold 예측 평균
    print("🔮 5-Fold 예측 (평균)...")

    pred_x_list = []
    pred_y_list = []

    for i, (model_x, model_y) in enumerate(zip(models_x, models_y)):
        print(f"  Fold {i+1}/5 예측 중...")
        pred_x = model_x.predict(X_test)
        pred_y = model_y.predict(X_test)
        pred_x_list.append(pred_x)
        pred_y_list.append(pred_y)

    # 평균
    final_pred_x = np.mean(pred_x_list, axis=0)
    final_pred_y = np.mean(pred_y_list, axis=0)

    print("✅ 예측 완료")
    print()

    # Submission 생성
    print("📝 Submission 파일 생성...")
    submission = pd.DataFrame({
        'game_episode': game_episode,
        'end_x': final_pred_x,
        'end_y': final_pred_y
    })

    submission_file = 'submission_lightgbm_optimized.csv'
    submission.to_csv(submission_file, index=False)

    print(f"✅ Submission 저장: {submission_file}")
    print()

    # 통계
    print("=" * 80)
    print("  예측 통계")
    print("=" * 80)
    print(f"\nend_x 통계:")
    print(f"   - 최소: {submission['end_x'].min():.2f}")
    print(f"   - 최대: {submission['end_x'].max():.2f}")
    print(f"   - 평균: {submission['end_x'].mean():.2f}")
    print(f"   - 표준편차: {submission['end_x'].std():.2f}")

    print(f"\nend_y 통계:")
    print(f"   - 최소: {submission['end_y'].min():.2f}")
    print(f"   - 최대: {submission['end_y'].max():.2f}")
    print(f"   - 평균: {submission['end_y'].mean():.2f}")
    print(f"   - 표준편차: {submission['end_y'].std():.2f}")

    # Fold 간 불일치
    std_x = np.std(pred_x_list, axis=0).mean()
    std_y = np.std(pred_y_list, axis=0).mean()

    print(f"\n📊 Fold 간 예측 불일치 (표준편차):")
    print(f"   - X 좌표: {std_x:.4f}m")
    print(f"   - Y 좌표: {std_y:.4f}m")

    if std_x < 1.0 and std_y < 1.0:
        print("   ✅ Fold 간 예측이 안정적입니다!")
    else:
        print("   ⚠️ Fold 간 예측 차이가 있습니다.")

    print("\n" + "=" * 80)
    print(f"🎉 추론 완료! {submission_file}을 제출하세요.")
    print("=" * 80)

    expected_public = avg_score * (14.138 / 1.5)
    print(f"\n📊 예상 Public LB: {expected_public:.4f}m")

    if expected_public < 13.8:
        print("   🎉 목표 달성 기대! (< 13.8m)")
    elif expected_public < 14.0:
        print("   ✅ 우수한 성능 기대! (< 14.0m)")
    else:
        print("   📊 기존과 비슷한 성능 예상")

    print("=" * 80)


if __name__ == "__main__":
    main()

