"""
V4 vs V5 성능 저하 원인 분석 스크립트

V5가 V4.1보다 16점대로 성능이 나빠진 원인을 파악
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("  V4 vs V5 성능 저하 원인 분석")
print("=" * 80)
print()

# 1. 데이터 로딩
print("📊 데이터 로딩 중...")
try:
    train_v4 = pd.read_csv('processed_train_data_v4.csv')
    print(f"✅ V4 Train: {train_v4.shape}")
except FileNotFoundError:
    print("❌ V4 파일 없음")
    train_v4 = None

try:
    train_v5 = pd.read_csv('processed_train_data_v5.csv')
    print(f"✅ V5 Train: {train_v5.shape}")
except FileNotFoundError:
    print("❌ V5 파일 없음")
    train_v5 = None

print()

if train_v4 is None or train_v5 is None:
    print("데이터 파일이 없어 분석을 중단합니다.")
    exit(1)

# 2. 기본 통계 비교
print("=" * 80)
print("1️⃣ 기본 통계 비교")
print("=" * 80)

print(f"\n샘플 수:")
print(f"  V4: {len(train_v4):,}개")
print(f"  V5: {len(train_v5):,}개")
print(f"  차이: {len(train_v5) - len(train_v4):+,}개")

print(f"\n피처 수:")
print(f"  V4: {train_v4.shape[1]}개")
print(f"  V5: {train_v5.shape[1]}개")
print(f"  차이: {train_v5.shape[1] - train_v4.shape[1]:+}개")

# 3. 타겟 변수 비교
print("\n" + "=" * 80)
print("2️⃣ 타겟 변수 분포 비교")
print("=" * 80)

v4_target_x = train_v4['target_x'].describe()
v5_target_x = train_v5['target_x'].describe()
v4_target_y = train_v4['target_y'].describe()
v5_target_y = train_v5['target_y'].describe()

print("\ntarget_x:")
print(f"  V4 평균: {v4_target_x['mean']:.2f}, 표준편차: {v4_target_x['std']:.2f}")
print(f"  V5 평균: {v5_target_x['mean']:.2f}, 표준편차: {v5_target_x['std']:.2f}")

print("\ntarget_y:")
print(f"  V4 평균: {v4_target_y['mean']:.2f}, 표준편차: {v4_target_y['std']:.2f}")
print(f"  V5 평균: {v5_target_y['mean']:.2f}, 표준편차: {v5_target_y['std']:.2f}")

# 4. 결측치 비교
print("\n" + "=" * 80)
print("3️⃣ 결측치 비교 (치명적 문제 확인)")
print("=" * 80)

drop_cols = ['target_x', 'target_y', 'game_episode', 'game_id']
v4_features = train_v4.drop(columns=[c for c in drop_cols if c in train_v4.columns])
v5_features = train_v5.drop(columns=[c for c in drop_cols if c in train_v5.columns])

v4_nan = v4_features.isna().sum().sum()
v5_nan = v5_features.isna().sum().sum()

print(f"\nV4 결측치: {v4_nan:,}개")
print(f"V5 결측치: {v5_nan:,}개")

if v4_nan > 0:
    print("\n⚠️ V4 결측치 분포:")
    v4_nan_cols = v4_features.isna().sum()
    v4_nan_cols = v4_nan_cols[v4_nan_cols > 0].sort_values(ascending=False)
    for col, count in v4_nan_cols.head(10).items():
        print(f"  {col}: {count:,}개 ({count/len(train_v4)*100:.1f}%)")

if v5_nan > 0:
    print("\n⚠️ V5 결측치 분포:")
    v5_nan_cols = v5_features.isna().sum()
    v5_nan_cols = v5_nan_cols[v5_nan_cols > 0].sort_values(ascending=False)
    for col, count in v5_nan_cols.head(10).items():
        print(f"  {col}: {count:,}개 ({count/len(train_v5)*100:.1f}%)")

# 5. 패딩 값 비교
print("\n" + "=" * 80)
print("4️⃣ 패딩 처리 비교 (핵심 개선사항)")
print("=" * 80)

v4_minus_one = (v4_features == -1).sum().sum()
v5_minus_one = (v5_features == -1).sum().sum()
v4_zero = (v4_features == 0).sum().sum()
v5_zero = (v5_features == 0).sum().sum()

print(f"\n-1 값 개수:")
print(f"  V4: {v4_minus_one:,}개 ({v4_minus_one/(v4_features.shape[0]*v4_features.shape[1])*100:.1f}%)")
print(f"  V5: {v5_minus_one:,}개 ({v5_minus_one/(v5_features.shape[0]*v5_features.shape[1])*100:.1f}%)")

print(f"\n0 값 개수:")
print(f"  V4: {v4_zero:,}개 ({v4_zero/(v4_features.shape[0]*v4_features.shape[1])*100:.1f}%)")
print(f"  V5: {v5_zero:,}개 ({v5_zero/(v5_features.shape[0]*v5_features.shape[1])*100:.1f}%)")

# 6. 신규 피처 확인
print("\n" + "=" * 80)
print("5️⃣ 신규 피처 확인")
print("=" * 80)

v4_cols = set(train_v4.columns)
v5_cols = set(train_v5.columns)

new_in_v5 = v5_cols - v4_cols
removed_in_v5 = v4_cols - v5_cols

print(f"\nV5에서 추가된 컬럼: {len(new_in_v5)}개")
if new_in_v5:
    # 패턴별 그룹화
    norm_x = [c for c in new_in_v5 if 'start_x_norm' in c]
    norm_y = [c for c in new_in_v5 if 'start_y_norm' in c]
    mc = [c for c in new_in_v5 if 'movement_consistency' in c]
    others = [c for c in new_in_v5 if c not in norm_x + norm_y + mc]

    print(f"  start_x_norm_*: {len(norm_x)}개")
    print(f"  start_y_norm_*: {len(norm_y)}개")
    print(f"  movement_consistency_*: {len(mc)}개")
    if others:
        print(f"  기타: {len(others)}개")
        for col in list(others)[:5]:
            print(f"    - {col}")

print(f"\nV5에서 제거된 컬럼: {len(removed_in_v5)}개")
if removed_in_v5:
    for col in list(removed_in_v5)[:10]:
        print(f"  - {col}")

# 7. 신규 피처 통계
print("\n" + "=" * 80)
print("6️⃣ 신규 피처 통계 분석")
print("=" * 80)

# movement_consistency 분석
mc_cols = [c for c in train_v5.columns if 'movement_consistency' in c]
if mc_cols:
    mc_data = train_v5[mc_cols]
    mc_mean = mc_data.mean().mean()
    mc_std = mc_data.std().mean()
    mc_min = mc_data.min().min()
    mc_max = mc_data.max().max()
    mc_nan_ratio = mc_data.isna().sum().sum() / (len(train_v5) * len(mc_cols)) * 100

    print(f"\nmovement_consistency ({len(mc_cols)}개 컬럼):")
    print(f"  평균: {mc_mean:.4f}")
    print(f"  표준편차: {mc_std:.4f}")
    print(f"  범위: [{mc_min:.4f}, {mc_max:.4f}]")
    print(f"  결측치: {mc_nan_ratio:.2f}%")

    # -1 비율 확인 (패딩)
    mc_minus_one = (mc_data == -1).sum().sum()
    mc_minus_one_ratio = mc_minus_one / (len(train_v5) * len(mc_cols)) * 100
    print(f"  패딩(-1): {mc_minus_one_ratio:.2f}%")

# start_x_norm 분석
norm_x_cols = [c for c in train_v5.columns if 'start_x_norm' in c]
if norm_x_cols:
    norm_x_data = train_v5[norm_x_cols]
    # -1 제외하고 통계 계산
    norm_x_valid = norm_x_data[norm_x_data != -1]

    print(f"\nstart_x_norm ({len(norm_x_cols)}개 컬럼):")
    print(f"  평균: {norm_x_valid.mean().mean():.4f}")
    print(f"  표준편차: {norm_x_valid.std().mean():.4f}")
    print(f"  범위: [{norm_x_valid.min().min():.4f}, {norm_x_valid.max().max():.4f}]")

    # -1 비율
    norm_x_minus_one = (norm_x_data == -1).sum().sum()
    norm_x_minus_one_ratio = norm_x_minus_one / (len(train_v5) * len(norm_x_cols)) * 100
    print(f"  패딩(-1): {norm_x_minus_one_ratio:.2f}%")

# 8. Speed 컬럼 비교
print("\n" + "=" * 80)
print("7️⃣ Speed 이상치 제어 확인")
print("=" * 80)

v4_speed_cols = [c for c in train_v4.columns if c.startswith('speed_')]
v5_speed_cols = [c for c in train_v5.columns if c.startswith('speed_')]

if v4_speed_cols and v5_speed_cols:
    v4_speed = train_v4[v4_speed_cols]
    v5_speed = train_v5[v5_speed_cols]

    # NaN이 아닌 값만 필터링
    v4_speed_valid = v4_speed[v4_speed != -1].dropna()
    v5_speed_valid = v5_speed[v5_speed != -1].dropna()

    print(f"\nV4 Speed 통계:")
    print(f"  최대: {v4_speed_valid.max().max():.2f} m/s")
    print(f"  평균: {v4_speed_valid.mean().mean():.2f} m/s")
    print(f"  중앙값: {v4_speed_valid.median().median():.2f} m/s")

    print(f"\nV5 Speed 통계:")
    print(f"  최대: {v5_speed_valid.max().max():.2f} m/s")
    print(f"  평균: {v5_speed_valid.mean().mean():.2f} m/s")
    print(f"  중앙값: {v5_speed_valid.median().median():.2f} m/s")

    # 50 초과 비율
    v4_over_50 = (v4_speed_valid > 50).sum().sum()
    v4_over_50_ratio = v4_over_50 / v4_speed_valid.count().sum() * 100
    v5_over_50 = (v5_speed_valid > 50).sum().sum()
    v5_over_50_ratio = v5_over_50 / v5_speed_valid.count().sum() * 100

    print(f"\n50 m/s 초과:")
    print(f"  V4: {v4_over_50:,}개 ({v4_over_50_ratio:.2f}%)")
    print(f"  V5: {v5_over_50:,}개 ({v5_over_50_ratio:.2f}%)")

# 9. 핵심 피처 분포 비교
print("\n" + "=" * 80)
print("8️⃣ 핵심 피처 분포 비교")
print("=" * 80)

# 공통 컬럼만 비교
common_cols = list(v4_cols & v5_cols)
# 메타 컬럼 제외
meta_cols = ['game_episode', 'game_id', 'target_x', 'target_y', 'final_team_id',
             'is_home', 'period_id', 'game_clock_min']
feature_cols = [c for c in common_cols if c not in meta_cols]

# 주요 피처만 비교
important_features = []
for pattern in ['distance_to_goal_start', 'shooting_angle', 'goal_approach',
                'in_penalty_area', 'in_final_third', 'x_zone', 'lane']:
    important_features.extend([c for c in feature_cols if pattern in c])

if important_features:
    print(f"\n주요 피처 비교 ({len(important_features[:10])}개만 표시):")

    for col in important_features[:10]:
        v4_mean = train_v4[col].replace(-1, np.nan).mean()
        v5_mean = train_v5[col].replace(-1, np.nan).mean()
        v4_std = train_v4[col].replace(-1, np.nan).std()
        v5_std = train_v5[col].replace(-1, np.nan).std()

        diff_mean = v5_mean - v4_mean
        diff_std = v5_std - v4_std

        print(f"\n{col}:")
        print(f"  평균: V4={v4_mean:.4f}, V5={v5_mean:.4f}, 차이={diff_mean:+.4f}")
        print(f"  표준편차: V4={v4_std:.4f}, V5={v5_std:.4f}, 차이={diff_std:+.4f}")

# 10. 최종 진단
print("\n" + "=" * 80)
print("🔍 최종 진단 - 성능 저하 원인 추정")
print("=" * 80)

issues = []

# 1. 결측치 증가
if v5_nan > v4_nan:
    issues.append(f"❌ 결측치 증가: V4={v4_nan:,} → V5={v5_nan:,} (+{v5_nan-v4_nan:,})")

# 2. 패딩 비율 변화
v4_padding_ratio = v4_minus_one / (v4_features.shape[0] * v4_features.shape[1]) * 100
v5_padding_ratio = v5_minus_one / (v5_features.shape[0] * v5_features.shape[1]) * 100

if abs(v5_padding_ratio - v4_padding_ratio) > 5:
    issues.append(f"⚠️ 패딩 비율 큰 변화: V4={v4_padding_ratio:.1f}% → V5={v5_padding_ratio:.1f}%")

# 3. 신규 피처 품질
if mc_cols:
    if mc_nan_ratio > 10:
        issues.append(f"❌ movement_consistency 결측치 과다: {mc_nan_ratio:.1f}%")
    if mc_minus_one_ratio > 40:
        issues.append(f"⚠️ movement_consistency 패딩 과다: {mc_minus_one_ratio:.1f}%")

# 4. Speed 클리핑 영향
if v4_speed_cols and v5_speed_cols:
    if v4_over_50_ratio > 1:
        issues.append(f"⚠️ V4에 이상치 다수 존재 (50 m/s 초과: {v4_over_50_ratio:.2f}%)")

print("\n의심 원인:")
if issues:
    for i, issue in enumerate(issues, 1):
        print(f"{i}. {issue}")
else:
    print("명확한 데이터 품질 문제는 발견되지 않았습니다.")

# 추가 분석 제안
print("\n" + "=" * 80)
print("💡 추가 분석 제안")
print("=" * 80)

print("""
1. 모델 하이퍼파라미터 재튜닝 필요
   - V5 데이터는 피처가 60개 더 많음 (780 → 840)
   - num_leaves, max_depth 증가 필요할 수 있음

2. 신규 피처의 Feature Importance 확인
   - movement_consistency가 노이즈일 가능성
   - 좌표 정규화가 중복 정보일 가능성

3. fillna 전략 재검토
   - V4: NaN 방치 → 트리가 자동 처리
   - V5: -1로 통일 → 명시적 패딩
   - fillna(0) 대신 fillna(-1) 시도 권장

4. 패딩 비율이 높은 에피소드 제거 시도
   - 짧은 에피소드(<10 이벤트)가 노이즈일 수 있음

5. V4.1이 v5 데이터로 14.1점을 달성했다면
   - train_v5.py와 train_v4.1_optuna.py의 차이점 확인 필요
   - fillna 전략, 피처 선택 등 비교

다음 단계:
python compare_training_configs.py  # 학습 설정 비교
python feature_importance_v5.py     # 피처 중요도 분석
""")

print("\n" + "=" * 80)

