"""
LightGBM V5 Optuna - Test 추론
Optuna로 최적화된 모델 사용
"""

import pandas as pd
import numpy as np
import pickle
from datetime import datetime
import warnings
import os
warnings.filterwarnings('ignore')


def main():
    print("=" * 80)
    print("  LightGBM V5 Optuna - Test 추론")
    print("=" * 80)
    print()

    # 1. 데이터 로딩
    print("📊 Test 데이터 로딩...")
    X_test = pd.read_csv('processed_test_data_v5.csv')
    print(f"Test: {X_test.shape}\n")

    # 2. 모델 로딩 (우선순위: final > checkpoint)
    print("🔧 Optuna 최적 모델 로딩...")

    model_file = None
    if os.path.exists('best_model_v5_optuna_final.pkl'):
        model_file = 'best_model_v5_optuna_final.pkl'
        print(f"   📁 최종 모델 발견: {model_file}")
    elif os.path.exists('best_model_v5_optuna_checkpoint.pkl'):
        model_file = 'best_model_v5_optuna_checkpoint.pkl'
        print(f"   📁 체크포인트 모델 발견: {model_file}")
    else:
        print("❌ 모델 파일을 찾을 수 없습니다!")
        print("   다음 중 하나를 먼저 실행하세요:")
        print("   1. python train_lightgbm_v5optuna.py")
        return

    with open(model_file, 'rb') as f:
        model_data = pickle.load(f)

    models_x = model_data['models_x']
    models_y = model_data['models_y']
    val_score = model_data['score']
    params = model_data.get('params', {})

    print(f"✅ 로딩 완료")
    print(f"   Validation CV: {val_score:.4f}m")
    print(f"   5-Fold 앙상블 모델\n")

    # 3. 피처 준비
    print("📊 피처 준비...")
    drop_cols = ['game_episode', 'game_id', 'target_x', 'target_y', 'final_team_id']
    test_episodes = X_test['game_episode'].copy()

    X_test_feat = X_test.drop(columns=[c for c in drop_cols if c in X_test.columns])

    # 🚨 [중요] fillna(0) 제거 - NaN 유지
    # X_test_feat = X_test_feat.fillna(0)  # 제거!

    # 범주형 변수 처리
    cat_keywords = ['type_id', 'res_id', 'team_id_enc', 'is_home', 'period_id', 'is_last']
    cat_features = [c for c in X_test_feat.columns if any(k in c for k in cat_keywords)]

    print(f"   범주형 변수 {len(cat_features)}개 -> category 타입 변환")
    for col in cat_features:
        X_test_feat[col] = X_test_feat[col].astype('category')

    # 나머지 수치형 변수 중 object 타입 변환
    for col in X_test_feat.columns:
        if col not in cat_features and X_test_feat[col].dtype == 'object':
            X_test_feat[col] = pd.to_numeric(X_test_feat[col], errors='coerce')

    print(f"✅ 준비 완료: {X_test_feat.shape}\n")

    # 4. 예측
    print("🔮 5-Fold 앙상블 예측...")

    pred_x_folds = []
    pred_y_folds = []

    for fold, (model_x, model_y) in enumerate(zip(models_x, models_y), 1):
        print(f"  Fold {fold}/5 예측 중...")
        pred_x = model_x.predict(X_test_feat, num_iteration=model_x.best_iteration)
        pred_y = model_y.predict(X_test_feat, num_iteration=model_y.best_iteration)
        pred_x_folds.append(pred_x)
        pred_y_folds.append(pred_y)

    # 5-Fold 평균
    pred_x = np.mean(pred_x_folds, axis=0)
    pred_y = np.mean(pred_y_folds, axis=0)

    # 좌표 클리핑 (경기장 범위)
    pred_x = np.clip(pred_x, 0, 105)
    pred_y = np.clip(pred_y, 0, 68)

    print(f"✅ 예측 완료\n")

    # 예측 통계
    print("📊 예측 통계:")
    print(f"   end_x: 평균={pred_x.mean():.2f}, 범위=[{pred_x.min():.2f}, {pred_x.max():.2f}]")
    print(f"   end_y: 평균={pred_y.mean():.2f}, 범위=[{pred_y.min():.2f}, {pred_y.max():.2f}]\n")

    # 5. 제출 파일 생성
    print("📝 제출 파일 생성...")

    submission = pd.DataFrame({
        'game_episode': test_episodes,
        'end_x': pred_x,
        'end_y': pred_y
    })

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'submission_v5_optuna_{timestamp}.csv'
    submission.to_csv(filename, index=False)

    print(f"✅ 저장: {filename}")
    print(f"   샘플 수: {len(submission):,}개\n")

    # 6. 요약
    print("="*80)
    print("  완료!")
    print("="*80)
    print(f"\n제출 파일: {filename}")
    print(f"Validation CV: {val_score:.4f}m")

    # 예상 Test 점수
    if val_score < 13.9:
        print(f"예상 Test: 13.7~13.9점 🎯 (우수!)")
    elif val_score < 14.0:
        print(f"예상 Test: 13.8~14.0점 ✅ (우수)")
    elif val_score < 14.1:
        print(f"예상 Test: 13.9~14.1점 ✅ (양호)")
    else:
        print(f"예상 Test: 14.0~14.2점")

    print("\n성능 비교:")
    print("  V3:    14.535점")
    print("  V4:    14.308점")
    print("  V4.1:  14.138점 (baseline)")
    print(f"  V5:    {val_score:.3f}m (CV)")

    # 개선도 계산
    baseline = 14.138
    if val_score < 14.0:
        improvement = baseline - val_score
        print(f"\n🎉 V4.1 대비 {improvement:.3f}m 개선!")

    print("\n" + "="*80)


if __name__ == "__main__":
    main()

