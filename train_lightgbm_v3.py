"""
LightGBM 모델 학습 - V3 (5-Fold 앙상블)

목표: Data Leakage 제거 + 시퀀스 모델링으로 Test 성능 대폭 개선
예상: 15~18점대 (V1 24점 대비 30% 개선)
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
    print("  LightGBM V3 - 5-Fold 앙상블 학습")
    print("  Data Leakage 제거 + 시퀀스 모델링")
    print("=" * 80)
    print()

    # 1. 데이터 로딩
    print("데이터 로딩...")
    data = pd.read_csv('processed_train_data_v3.csv')
    print(f"데이터: {data.shape}\n")

    # 2. 피처/타겟 분리
    print("피처/타겟 분리...")

    # 타겟
    y_train_x = data['target_x'].values
    y_train_y = data['target_y'].values

    # game_id 추출 (GroupKFold용)
    game_ids = data['game_id'].values

    # 피처 (불필요한 컬럼 제거)
    drop_cols = ['game_episode', 'game_id', 'target_x', 'target_y']
    X_train = data.drop(columns=[c for c in drop_cols if c in data.columns])

    # NaN 채우기
    X_train = X_train.fillna(0)

    # 데이터 타입 변환 (object → numeric)
    for col in X_train.columns:
        if X_train[col].dtype == 'object':
            X_train[col] = pd.to_numeric(X_train[col], errors='coerce').fillna(0)

    print(f"피처 수: {X_train.shape[1]}")
    print(f"샘플 수: {len(X_train):,}\n")

    # 3. 하이퍼파라미터 (참고 코드와 동일)
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
    print("5-Fold GroupKFold 학습 시작...\n")

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
    print("\n모델 저장 중...")
    with open('lightgbm_model_v3_5fold.pkl', 'wb') as f:
        pickle.dump({
            'models_x': models_x,
            'models_y': models_y,
            'val_score': mean_eucl,
            'fold_scores': fold_scores
        }, f)
    print("모델 저장: lightgbm_model_v3_5fold.pkl")

    # 7. 성능 비교
    print("\n" + "="*80)
    print("  성능 비교")
    print("="*80)
    print("\nV1 (Data Leakage 있음):")
    print("  - Validation: 0.93m (부정확)")
    print("  - Test: 24점대")

    print("\nV3 (Data Leakage 제거 + 시퀀스):")
    print(f"  - Validation: {mean_eucl:.4f}m (정확)")
    print("  - 예상 Test: 15~18점대 (30% 개선)")

    if mean_eucl < 2.0:
        print("\n매우 우수한 성능! Test에서도 좋은 결과 기대")
    elif mean_eucl < 3.0:
        print("\n좋은 성능! 하이퍼파라미터 튜닝으로 추가 개선 가능")
    else:
        print("\n추가 튜닝 필요")

    print("\n" + "="*80)
    print("다음 단계:")
    print("   1. Test 추론 (inference_v3.py)")
    print("   2. 제출 및 점수 확인")
    print("   3. 하이퍼파라미터 튜닝 (필요시)")
    print("="*80)

    return models_x, models_y, scores_df


if __name__ == "__main__":
    models_x, models_y, scores_df = main()

