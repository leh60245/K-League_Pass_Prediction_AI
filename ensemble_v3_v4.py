"""
V3 + V4 앙상블 추론

두 모델의 예측을 결합하여 더 강건한 예측 생성
"""

import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


def main():
    print("=" * 80)
    print("  V3 + V4 앙상블 추론")
    print("  두 모델의 시너지로 최고 성능 달성")
    print("=" * 80)
    print()

    # 1. V3 제출 파일 로딩
    print("📊 V3 제출 파일 로딩...")
    try:
        v3_file = 'submission_v3_5fold_20251216_172855.csv'
        v3_sub = pd.read_csv(v3_file)
        print(f"✅ V3 로딩 완료: {v3_file}")
    except:
        print("⚠️  V3 제출 파일을 찾을 수 없습니다.")
        print("   inference_v3.py를 먼저 실행하세요.")
        return

    # 2. V4 제출 파일 로딩
    print("📊 V4 제출 파일 로딩...")
    import glob
    v4_files = glob.glob('submission_v4_5fold_*.csv')
    if not v4_files:
        print("⚠️  V4 제출 파일을 찾을 수 없습니다.")
        return

    v4_file = sorted(v4_files)[-1]  # 가장 최근 파일
    v4_sub = pd.read_csv(v4_file)
    print(f"✅ V4 로딩 완료: {v4_file}\n")

    # 3. 데이터 정렬 (game_episode 기준)
    v3_sub = v3_sub.sort_values('game_episode').reset_index(drop=True)
    v4_sub = v4_sub.sort_values('game_episode').reset_index(drop=True)

    # 4. Episode 일치 확인
    if not v3_sub['game_episode'].equals(v4_sub['game_episode']):
        print("⚠️  V3와 V4의 game_episode가 일치하지 않습니다!")
        return

    print(f"✅ 두 모델의 예측 샘플 수: {len(v3_sub):,}\n")

    # 5. 앙상블 전략
    print("🔮 앙상블 전략 선택...")

    strategies = {
        '평균 (0.5:0.5)': (0.5, 0.5),
        'V3 우선 (0.6:0.4)': (0.6, 0.4),
        'V4 우선 (0.4:0.6)': (0.4, 0.6),
    }

    results = []

    for name, (w3, w4) in strategies.items():
        print(f"\n📊 {name} 앙상블 중...")

        # 앙상블
        pred_x = w3 * v3_sub['end_x'] + w4 * v4_sub['end_x']
        pred_y = w3 * v3_sub['end_y'] + w4 * v4_sub['end_y']

        # 필드 범위로 클립
        pred_x = np.clip(pred_x, 0, 105)
        pred_y = np.clip(pred_y, 0, 68)

        # 제출 파일 생성
        submission = pd.DataFrame({
            'game_episode': v3_sub['game_episode'],
            'end_x': pred_x,
            'end_y': pred_y
        })

        # 파일명
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'submission_ensemble_v3_v4_{int(w3*100)}_{int(w4*100)}_{timestamp}.csv'
        submission.to_csv(filename, index=False)

        print(f"✅ 저장: {filename}")

        # 통계
        results.append({
            'strategy': name,
            'weights': f'{w3}:{w4}',
            'filename': filename,
            'mean_x': pred_x.mean(),
            'mean_y': pred_y.mean(),
            'std_x': pred_x.std(),
            'std_y': pred_y.std(),
        })

    # 6. 결과 요약
    print("\n" + "=" * 80)
    print("  앙상블 결과 요약")
    print("=" * 80)

    results_df = pd.DataFrame(results)
    print("\n생성된 제출 파일:")
    for _, row in results_df.iterrows():
        print(f"\n{row['strategy']} (가중치 {row['weights']})")
        print(f"  - 파일: {row['filename']}")
        print(f"  - end_x: {row['mean_x']:.2f} ± {row['std_x']:.2f}")
        print(f"  - end_y: {row['mean_y']:.2f} ± {row['std_y']:.2f}")

    # 7. V3/V4 예측 차이 분석
    print("\n" + "=" * 80)
    print("  V3 vs V4 예측 차이 분석")
    print("=" * 80)

    diff_x = np.abs(v3_sub['end_x'] - v4_sub['end_x'])
    diff_y = np.abs(v3_sub['end_y'] - v4_sub['end_y'])
    diff_euclidean = np.sqrt((v3_sub['end_x'] - v4_sub['end_x'])**2 +
                             (v3_sub['end_y'] - v4_sub['end_y'])**2)

    print(f"\n평균 예측 차이:")
    print(f"  - end_x: {diff_x.mean():.4f}m")
    print(f"  - end_y: {diff_y.mean():.4f}m")
    print(f"  - 유클리드 거리: {diff_euclidean.mean():.4f}m")

    print(f"\n최대 예측 차이:")
    print(f"  - end_x: {diff_x.max():.4f}m")
    print(f"  - end_y: {diff_y.max():.4f}m")
    print(f"  - 유클리드 거리: {diff_euclidean.max():.4f}m")

    # 큰 차이가 나는 샘플 비율
    large_diff = (diff_euclidean > 5.0).sum()
    print(f"\n큰 차이 샘플 (>5m): {large_diff} / {len(diff_euclidean)} ({large_diff/len(diff_euclidean)*100:.2f}%)")

    if diff_euclidean.mean() < 2.0:
        print("\n✅ V3와 V4의 예측이 매우 유사합니다 (앙상블 효과 제한적)")
    elif diff_euclidean.mean() < 5.0:
        print("\n✅ V3와 V4의 예측에 적당한 차이가 있습니다 (앙상블 효과 기대)")
    else:
        print("\n⚠️  V3와 V4의 예측 차이가 큽니다 (앙상블 신중)")

    # 8. 최종 권장사항
    print("\n" + "=" * 80)
    print("  최종 권장사항")
    print("=" * 80)

    print("\n📊 제출 우선순위:")
    print("   1순위: V3 단독 (검증됨, Validation 14.40m)")
    print("   2순위: V4 단독 (V2 피처 효과 확인, Validation 14.36m)")
    print("   3순위: 앙상블 0.5:0.5 (두 모델의 균형)")
    print("   4순위: 앙상블 0.6:0.4 (V3 우선)")

    print("\n🎯 기대 효과:")
    print("   - 두 모델 모두 14점대 성능")
    print("   - 앙상블로 안정성 향상 가능")
    print("   - V4의 도메인 지식이 일부 기여")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()

