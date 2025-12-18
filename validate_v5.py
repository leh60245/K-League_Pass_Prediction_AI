"""
V5 데이터 검증 스크립트

V4 vs V5 비교를 통해 5대 개선사항이 제대로 반영되었는지 검증
"""

import pandas as pd
import numpy as np

print("=" * 80)
print("  V5 데이터 검증 스크립트")
print("=" * 80)
print()

# 데이터 로딩
print("📊 데이터 로딩 중...")
try:
    train_v5 = pd.read_csv('processed_train_data_v5.csv')
    test_v5 = pd.read_csv('processed_test_data_v5.csv')
    print(f"✅ V5 Train: {train_v5.shape}")
    print(f"✅ V5 Test: {test_v5.shape}")
except FileNotFoundError:
    print("❌ V5 파일이 없습니다. preprocessing_v5.py를 먼저 실행하세요.")
    exit(1)

try:
    train_v4 = pd.read_csv('processed_train_data_v4.csv')
    test_v4 = pd.read_csv('processed_test_data_v4.csv')
    print(f"✅ V4 Train: {train_v4.shape}")
    print(f"✅ V4 Test: {test_v4.shape}")
    has_v4 = True
except FileNotFoundError:
    print("⚠️  V4 파일 없음 (비교 생략)")
    has_v4 = False

print()

# 1. 결측치 검증 (치명적 오류 수정 확인)
print("=" * 80)
print("1️⃣  결측치 검증 (패딩 -1 통일 확인)")
print("=" * 80)

target_cols = ['target_x', 'target_y']
v5_train_nan = train_v5.drop(columns=target_cols, errors='ignore').isna().sum().sum()
v5_test_nan = test_v5.isna().sum().sum()

print(f"V5 Train 결측치 (target 제외): {v5_train_nan:,}개")
print(f"V5 Test 결측치: {v5_test_nan:,}개")

if v5_train_nan == 0 and v5_test_nan == 0:
    print("✅ PASS: 결측치가 없습니다 (패딩 -1 처리 완료)")
else:
    print("❌ FAIL: 결측치가 존재합니다!")

if has_v4:
    v4_train_nan = train_v4.drop(columns=target_cols, errors='ignore').isna().sum().sum()
    v4_test_nan = test_v4.isna().sum().sum()
    print(f"\n📊 V4 Train 결측치 (비교): {v4_train_nan:,}개")
    print(f"📊 V4 Test 결측치 (비교): {v4_test_nan:,}개")
    print(f"📈 개선: {v4_train_nan - v5_train_nan:,}개 결측치 제거")

print()

# 2. 속도 이상치 검증
print("=" * 80)
print("2️⃣  속도 이상치 검증 (50 m/s 클리핑 확인)")
print("=" * 80)

speed_cols = [col for col in train_v5.columns if col.startswith('speed_')]
if speed_cols:
    v5_max_speed = train_v5[speed_cols].max().max()
    v5_mean_speed = train_v5[speed_cols].mean().mean()

    print(f"V5 속도 통계:")
    print(f"  - 최대값: {v5_max_speed:.2f} m/s")
    print(f"  - 평균값: {v5_mean_speed:.2f} m/s")

    if v5_max_speed <= 50.0:
        print("✅ PASS: 속도가 50 m/s 이하입니다 (클리핑 정상 작동)")
    else:
        print("❌ FAIL: 속도가 50 m/s를 초과합니다!")

    if has_v4:
        speed_cols_v4 = [col for col in train_v4.columns if col.startswith('speed_')]
        if speed_cols_v4:
            v4_max_speed = train_v4[speed_cols_v4].max().max()
            print(f"\n📊 V4 최대 속도 (비교): {v4_max_speed:.2f} m/s")
            if v4_max_speed > 50:
                print(f"📈 개선: 이상치 {v4_max_speed - 50:.2f} m/s 제거")
else:
    print("⚠️  Speed 컬럼을 찾을 수 없습니다")

print()

# 3. 신규 피처 확인
print("=" * 80)
print("3️⃣  신규 피처 확인")
print("=" * 80)

# start_x_norm, start_y_norm 확인
norm_x_cols = [col for col in train_v5.columns if 'start_x_norm' in col]
norm_y_cols = [col for col in train_v5.columns if 'start_y_norm' in col]

print(f"✅ start_x_norm 피처: {len(norm_x_cols)}개")
print(f"✅ start_y_norm 피처: {len(norm_y_cols)}개")

if norm_x_cols:
    norm_x_min = train_v5[norm_x_cols].min().min()
    norm_x_max = train_v5[norm_x_cols].max().max()
    print(f"   범위: [{norm_x_min:.3f}, {norm_x_max:.3f}] (0~1 예상)")

    if -1 <= norm_x_min < 0 or 1 < norm_x_max <= 1.1:
        print("   ⚠️  주의: 범위가 [0, 1]을 벗어났습니다 (패딩 -1 포함)")
    elif 0 <= norm_x_min and norm_x_max <= 1:
        print("   ✅ 정상: 0~1 범위 내")

# movement_consistency 확인
mc_cols = [col for col in train_v5.columns if 'movement_consistency' in col]
print(f"\n✅ movement_consistency 피처: {len(mc_cols)}개")

if mc_cols:
    mc_min = train_v5[mc_cols].min().min()
    mc_max = train_v5[mc_cols].max().max()
    mc_mean = train_v5[mc_cols].mean().mean()

    print(f"   범위: [{mc_min:.3f}, {mc_max:.3f}] (-1~1 예상)")
    print(f"   평균: {mc_mean:.3f}")

    if -1 <= mc_min and mc_max <= 1:
        print("   ✅ PASS: [-1, 1] 범위 내 (코사인 유사도 정상)")
    else:
        print("   ❌ FAIL: [-1, 1] 범위를 벗어났습니다!")

print()

# 4. 피처 개수 비교
print("=" * 80)
print("4️⃣  피처 개수 비교")
print("=" * 80)

v5_train_cols = train_v5.shape[1]
v5_test_cols = test_v5.shape[1]

print(f"V5 Train 컬럼: {v5_train_cols}개")
print(f"V5 Test 컬럼: {v5_test_cols}개")

if has_v4:
    v4_train_cols = train_v4.shape[1]
    v4_test_cols = test_v4.shape[1]

    print(f"\nV4 Train 컬럼: {v4_train_cols}개")
    print(f"V4 Test 컬럼: {v4_test_cols}개")

    train_diff = v5_train_cols - v4_train_cols
    test_diff = v5_test_cols - v4_test_cols

    print(f"\n📈 Train 증가: +{train_diff}개")
    print(f"📈 Test 증가: +{test_diff}개")

    if train_diff == 60 and test_diff == 60:
        print("✅ PASS: 예상대로 60개 증가 (3개 피처 × K=20)")
    elif train_diff > 0 and test_diff > 0:
        print(f"⚠️  주의: 증가량이 예상({60}개)과 다릅니다")
    else:
        print("❌ FAIL: 피처가 증가하지 않았습니다!")

# 추가된 컬럼 확인
if has_v4:
    new_cols = set(train_v5.columns) - set(train_v4.columns)

    # 신규 피처 패턴 확인
    new_norm_x = [c for c in new_cols if 'start_x_norm' in c]
    new_norm_y = [c for c in new_cols if 'start_y_norm' in c]
    new_mc = [c for c in new_cols if 'movement_consistency' in c]

    print(f"\n신규 컬럼 패턴 분석:")
    print(f"  - start_x_norm_*: {len(new_norm_x)}개")
    print(f"  - start_y_norm_*: {len(new_norm_y)}개")
    print(f"  - movement_consistency_*: {len(new_mc)}개")
    print(f"  - 기타: {len(new_cols) - len(new_norm_x) - len(new_norm_y) - len(new_mc)}개")

print()

# 5. 패딩 값 확인 (-1 통일 검증)
print("=" * 80)
print("5️⃣  패딩 값 확인 (-1 통일 검증)")
print("=" * 80)

# -1 값의 비율 확인 (패딩으로 추정)
v5_minus_one_ratio = (train_v5 == -1).sum().sum() / (train_v5.shape[0] * train_v5.shape[1])
print(f"V5 Train 전체 데이터 중 -1 비율: {v5_minus_one_ratio * 100:.2f}%")

# 범주형 컬럼에서 -1 존재 확인
cat_cols = [col for col in train_v5.columns if any(x in col for x in ['type_id', 'res_id', 'team_id_enc'])]
if cat_cols:
    v5_cat_has_minus_one = (train_v5[cat_cols] == -1).any().any()
    print(f"범주형 컬럼에 -1 존재: {'✅ 예' if v5_cat_has_minus_one else '❌ 아니오'}")

    if v5_cat_has_minus_one:
        print("✅ PASS: 범주형 패딩이 -1로 처리되었습니다")

print()

# 최종 요약
print("=" * 80)
print("📊 최종 검증 요약")
print("=" * 80)

checks = []
checks.append(("결측치 제거", v5_train_nan == 0 and v5_test_nan == 0))
checks.append(("속도 클리핑 (≤50)", speed_cols and v5_max_speed <= 50.0))
checks.append(("좌표 정규화 추가", len(norm_x_cols) > 0 and len(norm_y_cols) > 0))
checks.append(("관성 피처 추가", len(mc_cols) > 0))
if has_v4:
    checks.append(("피처 개수 증가", train_diff > 0))

passed = sum([1 for _, result in checks if result])
total = len(checks)

print(f"\n통과한 검증: {passed}/{total}")
for check_name, result in checks:
    status = "✅" if result else "❌"
    print(f"{status} {check_name}")

if passed == total:
    print("\n" + "=" * 80)
    print("🎉 축하합니다! V5 전처리 파이프라인이 완벽하게 작동합니다!")
    print("=" * 80)
    print("\n다음 단계:")
    print("1. V5 데이터로 모델 재학습")
    print("2. V4 vs V5 성능 비교")
    print("3. 예상 Test RMSPE: 12~14점대")
else:
    print("\n" + "=" * 80)
    print("⚠️  일부 검증이 실패했습니다. 코드를 다시 확인하세요.")
    print("=" * 80)

