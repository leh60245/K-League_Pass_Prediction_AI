"""
LightGBM 모델 학습 - V4 (5-Fold 앙상블)

V2의 풍부한 피처 + V3의 시퀀스 모델링 = 최고 성능
목표: Test 성능 14~16점대
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GroupKFold
import pickle
import warnings
warnings.filterwarnings('ignore')


def euclidean_distance(y_true, y_pred):
    """유클리드 거리 계산"""
    distances = np.sqrt((y_true[:, 0] - y_pred[:, 0])**2 +
                       (y_true[:, 1] - y_pred[:, 1])**2)
    return distances.mean()


def main():
    print("=" * 80)
    print("  LightGBM V4 - 5-Fold 앙상블 학습")
    print("  V2 피처 + V3 시퀀스 모델링")
    print("=" * 80)
    print()

    # 1. 데이터 로딩
    print("📊 데이터 로딩...")
    data = pd.read_csv('processed_train_data_v4.csv')
    print(f"데이터: {data.shape}\n")

    # 2. 피처/타겟 분리
    print("📊 피처/타겟 분리...")

    # 타겟
    y_train_x = data['target_x'].values
    y_train_y = data['target_y'].values

    # game_id 추출 (GroupKFold용)
    game_ids = data['game_id'].values

    # 피처 (불필요한 컬럼 제거)
    drop_cols = ['game_episode', 'game_id', 'target_x', 'target_y', 'final_team_id']
    X_train = data.drop(columns=[c for c in drop_cols if c in data.columns])

    # NaN 채우기
    X_train = X_train.fillna(0)

    # 데이터 타입 변환
    for col in X_train.columns:
        if X_train[col].dtype == 'object':
            X_train[col] = pd.to_numeric(X_train[col], errors='coerce').fillna(0)

    print(f"피처 수: {X_train.shape[1]}")
    print(f"샘플 수: {len(X_train):,}\n")

    # 3. 하이퍼파라미터
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'learning_rate': 0.05,
        'num_leaves': 127,
        'min_data_in_leaf': 80,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 1,
        'verbose': -1,
    }

    # 4. 5-Fold GroupKFold 학습
    print("🔧 5-Fold GroupKFold 학습 시작...\n")

    gkf = GroupKFold(n_splits=5)

    models_x = []
    models_y = []
    fold_scores = []

    for fold, (train_idx, val_idx) in enumerate(gkf.split(X_train, groups=game_ids)):
        print(f"{'='*60}")
        print(f"  Fold {fold+1}/5")
        print(f"{'='*60}")

        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr_x, y_val_x = y_train_x[train_idx], y_train_x[val_idx]
        y_tr_y, y_val_y = y_train_y[train_idx], y_train_y[val_idx]

        print(f"Train: {len(X_tr):,}, Val: {len(X_val):,}")

        # X 좌표 모델
        print("end_x 모델 학습 중...")
        dtrain_x = lgb.Dataset(X_tr, label=y_tr_x)
        dvalid_x = lgb.Dataset(X_val, label=y_val_x, reference=dtrain_x)

        model_x = lgb.train(
            params,
            dtrain_x,
            num_boost_round=3000,
            valid_sets=[dtrain_x, dvalid_x],
            valid_names=['train', 'valid'],
            callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False)]
        )
        models_x.append(model_x)
        print(f"  -> 최종 {model_x.best_iteration} rounds")

        # Y 좌표 모델
        print("end_y 모델 학습 중...")
        dtrain_y = lgb.Dataset(X_tr, label=y_tr_y)
        dvalid_y = lgb.Dataset(X_val, label=y_val_y, reference=dtrain_y)

        model_y = lgb.train(
            params,
            dtrain_y,
            num_boost_round=3000,
            valid_sets=[dtrain_y, dvalid_y],
            valid_names=['train', 'valid'],
            callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False)]
        )
        models_y.append(model_y)
        print(f"  -> 최종 {model_y.best_iteration} rounds")

        # 검증
        pred_x = model_x.predict(X_val, num_iteration=model_x.best_iteration)
        pred_y = model_y.predict(X_val, num_iteration=model_y.best_iteration)
        y_pred = np.column_stack([pred_x, pred_y])
        y_val = np.column_stack([y_val_x, y_val_y])

        eucl_dist = euclidean_distance(y_val, y_pred)
        mse_x = mean_squared_error(y_val_x, pred_x)
        mse_y = mean_squared_error(y_val_y, pred_y)

        print(f"\nFold {fold+1} 결과:")
        print(f"  - 유클리드 거리: {eucl_dist:.4f}m")
        print(f"  - MSE X: {mse_x:.4f}")
        print(f"  - MSE Y: {mse_y:.4f}\n")

        fold_scores.append({
            'fold': fold + 1,
            'euclidean': eucl_dist,
            'mse_x': mse_x,
            'mse_y': mse_y
        })

        # Feature Importance (첫 번째 fold만)
        if fold == 0:
            print("\n📊 Feature Importance Top 20 (X 좌표):")
            importance_x = model_x.feature_importance()
            importance_df = pd.DataFrame({
                'feature': X_train.columns,
                'importance': importance_x
            }).sort_values('importance', ascending=False)

            for idx, row in importance_df.head(20).iterrows():
                print(f"  {importance_df.index.get_loc(idx)+1:2d}. {row['feature']:40s}: {row['importance']:8.1f}")

    # 5. 전체 결과 요약
    print("\n" + "="*80)
    print("  전체 결과 요약")
    print("="*80)

    scores_df = pd.DataFrame(fold_scores)
    mean_eucl = scores_df['euclidean'].mean()
    std_eucl = scores_df['euclidean'].std()

    print(f"\n평균 유클리드 거리: {mean_eucl:.4f}m ± {std_eucl:.4f}m")
    print(f"평균 MSE X: {scores_df['mse_x'].mean():.4f}")
    print(f"평균 MSE Y: {scores_df['mse_y'].mean():.4f}")

    print("\nFold별 상세:")
    for _, row in scores_df.iterrows():
        print(f"  Fold {int(row['fold'])}: {row['euclidean']:.4f}m")

    # 6. 모델 저장
    print("\n💾 모델 저장 중...")
    with open('lightgbm_model_v4_5fold.pkl', 'wb') as f:
        pickle.dump({
            'models_x': models_x,
            'models_y': models_y,
            'val_score': mean_eucl,
            'fold_scores': fold_scores
        }, f)
    print("✅ 모델 저장: lightgbm_model_v4_5fold.pkl")

    # 7. 성능 비교
    print("\n" + "="*80)
    print("  성능 비교")
    print("="*80)

    print("\n📊 버전별 성능 비교:")
    print("V1 (Baseline):")
    print("  - Validation: 0.93m (Data Leakage)")
    print("  - Test: 24점대")

    print("\nV2 (도메인 지식 피처):")
    print("  - 풍부한 피처 엔지니어링")
    print("  - Data Leakage 존재")

    print("\nV3 (시퀀스 모델링):")
    print("  - Validation: ~1.5m")
    print("  - Test: 14점대 (30% 개선)")

    print(f"\nV4 (V2 + V3 통합):")
    print(f"  - Validation: {mean_eucl:.4f}m")
    print(f"  - 예상 Test: 13~15점대 (최고 성능 기대)")

    if mean_eucl < 1.5:
        print("\n🎉 매우 우수한 성능! Test에서 좋은 결과 기대")
    elif mean_eucl < 2.0:
        print("\n✅ 좋은 성능! V3와 유사하거나 더 나은 결과 예상")
    else:
        print("\n📈 추가 튜닝으로 개선 가능")

    print("\n" + "="*80)
    print("다음 단계:")
    print("   1. Test 추론 (inference_v4.py)")
    print("   2. 제출 및 점수 확인")
    print("   3. V3/V4 성능 비교 분석")
    print("="*80)


if __name__ == "__main__":
    main()

