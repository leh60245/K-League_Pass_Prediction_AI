"""
제출 파일 비교 분석

V1 vs V2 모델의 예측 차이 분석
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def compare_submissions(file1, file2, label1='V1', label2='V2'):
    """두 제출 파일 비교"""

    print("=" * 80)
    print("  제출 파일 비교 분석")
    print("=" * 80)
    print()

    # 1. 데이터 로딩
    print(f"📂 로딩: {file1}")
    sub1 = pd.read_csv(file1)
    print(f"📂 로딩: {file2}")
    sub2 = pd.read_csv(file2)

    print(f"✅ {label1}: {len(sub1):,}개")
    print(f"✅ {label2}: {len(sub2):,}개")

    # 2. 기본 통계 비교
    print("\n" + "=" * 80)
    print("  기본 통계 비교")
    print("=" * 80)

    stats_data = []

    for name, df in [(label1, sub1), (label2, sub2)]:
        stats_data.append({
            'Model': name,
            'end_x_mean': df['end_x'].mean(),
            'end_x_std': df['end_x'].std(),
            'end_x_min': df['end_x'].min(),
            'end_x_max': df['end_x'].max(),
            'end_y_mean': df['end_y'].mean(),
            'end_y_std': df['end_y'].std(),
            'end_y_min': df['end_y'].min(),
            'end_y_max': df['end_y'].max(),
        })

    stats_df = pd.DataFrame(stats_data)
    print("\n📊 통계:")
    print(stats_df.to_string(index=False))

    # 3. 예측 차이 분석
    print("\n" + "=" * 80)
    print("  예측 차이 분석")
    print("=" * 80)

    # 병합
    merged = pd.merge(sub1, sub2, on='game_episode', suffixes=('_v1', '_v2'))

    # 유클리드 거리 계산
    merged['diff_distance'] = np.sqrt(
        (merged['end_x_v1'] - merged['end_x_v2'])**2 +
        (merged['end_y_v1'] - merged['end_y_v2'])**2
    )

    # X, Y 개별 차이
    merged['diff_x'] = merged['end_x_v1'] - merged['end_x_v2']
    merged['diff_y'] = merged['end_y_v1'] - merged['end_y_v2']

    print(f"\n📊 예측 차이 (유클리드 거리):")
    print(f"  - 평균: {merged['diff_distance'].mean():.4f}m")
    print(f"  - 중앙값: {merged['diff_distance'].median():.4f}m")
    print(f"  - 최소: {merged['diff_distance'].min():.4f}m")
    print(f"  - 최대: {merged['diff_distance'].max():.4f}m")
    print(f"  - 표준편차: {merged['diff_distance'].std():.4f}m")

    print(f"\n📊 X 좌표 차이:")
    print(f"  - 평균: {merged['diff_x'].mean():.4f}m")
    print(f"  - 표준편차: {merged['diff_x'].std():.4f}m")

    print(f"\n📊 Y 좌표 차이:")
    print(f"  - 평균: {merged['diff_y'].mean():.4f}m")
    print(f"  - 표준편차: {merged['diff_y'].std():.4f}m")

    # 4. 차이 분포
    print("\n📊 차이 범위별 비율:")
    ranges = [(0, 1), (1, 2), (2, 5), (5, 10), (10, 100)]
    for low, high in ranges:
        mask = (merged['diff_distance'] >= low) & (merged['diff_distance'] < high)
        count = mask.sum()
        pct = (count / len(merged)) * 100
        print(f"  - {low:3.0f}m ~ {high:3.0f}m: {count:5,}개 ({pct:5.1f}%)")

    # 5. 가장 차이 나는 케이스
    print("\n📊 가장 차이 나는 케이스 Top 5:")
    top_diff = merged.nlargest(5, 'diff_distance')[['game_episode', 'end_x_v1', 'end_y_v1',
                                                      'end_x_v2', 'end_y_v2', 'diff_distance']]
    print(top_diff.to_string(index=False))

    # 6. 영역별 분포 비교
    print("\n" + "=" * 80)
    print("  영역별 분포 비교")
    print("=" * 80)

    # X축
    print("\n📊 X축 분포:")
    for name, df in [(label1, sub1), (label2, sub2)]:
        x_dist = pd.cut(df['end_x'], bins=[0, 35, 70, 105], labels=['수비진', '중원', '공격진'])
        print(f"\n{name}:")
        print(x_dist.value_counts(normalize=True).sort_index().to_string())

    # Y축
    print("\n📊 Y축 분포:")
    for name, df in [(label1, sub1), (label2, sub2)]:
        y_dist = pd.cut(df['end_y'], bins=[0, 22.67, 45.33, 68], labels=['좌측', '중앙', '우측'])
        print(f"\n{name}:")
        print(y_dist.value_counts(normalize=True).sort_index().to_string())

    # 7. 권장사항
    print("\n" + "=" * 80)
    print("  권장사항")
    print("=" * 80)

    avg_diff = merged['diff_distance'].mean()

    print(f"\n평균 예측 차이: {avg_diff:.4f}m")

    if avg_diff < 2:
        print("✅ 두 모델의 예측이 매우 유사합니다.")
        print("   → Ensemble 시 효과가 제한적일 수 있습니다.")
        print(f"   → Validation 성능이 더 좋은 {label1}을 제출하는 것을 권장합니다.")
    elif avg_diff < 5:
        print("⚠️  두 모델의 예측이 약간 다릅니다.")
        print("   → Ensemble을 고려해볼 수 있습니다.")
        print("   → 가중 평균: (V1 × 0.7 + V2 × 0.3) 추천")
    else:
        print("🔍 두 모델의 예측이 상당히 다릅니다.")
        print("   → 모델이 다른 패턴을 학습했을 가능성")
        print("   → Ensemble 시 다양성 확보 가능")
        print("   → Validation 점수 기반 가중치 설정 필요")

    print("\n" + "=" * 80)
    print("✅ 비교 분석 완료!")
    print("=" * 80)

    return merged


if __name__ == "__main__":
    # V1 vs V2 비교
    merged = compare_submissions(
        'submission_v1_final.csv',
        'submission_v2_20251216_162340.csv',
        label1='V1 (0.93m)',
        label2='V2 (1.06m)'
    )

