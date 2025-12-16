"""
LightGBM 모델 학습 - V2 데이터 사용

목표: 개선된 전처리 데이터로 성능 향상 확인
"""

import pandas as pd
import numpy as np
import pickle
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')

def euclidean_distance(y_true, y_pred):
    """유클리드 거리 계산"""
    distances = np.sqrt((y_true[:, 0] - y_pred[:, 0])**2 +
                       (y_true[:, 1] - y_pred[:, 1])**2)
    return distances.mean()

def main():
    print("=" * 80)
    print("  LightGBM V2 - 개선된 피처로 학습")
    print("=" * 80)
    print()

    # 1. 데이터 로딩
    print("📊 V2 데이터 로딩...")
    data = pd.read_csv('processed_train_data_v2.csv')
    print(f"✅ 데이터: {data.shape}")

    # 2. Preprocessor 로딩
    print("\n📦 Preprocessor V2 로딩...")
    with open('preprocessor_v2.pkl', 'rb') as f:
        preprocessor_data = pickle.load(f)
    print("✅ Preprocessor 로딩 완료")

    # 3. 피처/타겟 분리
    from preprocessing_v2 import DataPreprocessorV2
    preprocessor = DataPreprocessorV2()
    preprocessor.type_encoder = preprocessor_data['type_encoder']
    preprocessor.result_encoder = preprocessor_data['result_encoder']

    feature_cols = preprocessor.get_feature_columns()

    # 실제 존재하는 피처만 사용
    available_features = [col for col in feature_cols if col in data.columns]
    print(f"\n📊 사용 가능한 피처: {len(available_features)}개")

    X = data[available_features]
    y = data[['end_x', 'end_y']].values

    # 4. Train/Val Split (Game-based)
    print("\n📊 Train/Val Split...")
    from sklearn.model_selection import GroupKFold

    gkf = GroupKFold(n_splits=5)
    fold_scores = []

    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, groups=data['game_id'])):
        print(f"\n{'='*60}")
        print(f"  Fold {fold+1}/5")
        print(f"{'='*60}")

        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        print(f"Train: {X_train.shape}, Val: {X_val.shape}")

        # 5. 모델 학습 (X 좌표)
        print("\n🔧 X 좌표 모델 학습...")
        model_x = LGBMRegressor(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=8,
            num_leaves=63,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbose=-1
        )

        model_x.fit(
            X_train, y_train[:, 0],
            eval_set=[(X_val, y_val[:, 0])],
            eval_metric='rmse',
            callbacks=[
                # early_stopping(50, verbose=False)
            ]
        )

        # 6. 모델 학습 (Y 좌표)
        print("🔧 Y 좌표 모델 학습...")
        model_y = LGBMRegressor(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=8,
            num_leaves=63,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbose=-1
        )

        model_y.fit(
            X_train, y_train[:, 1],
            eval_set=[(X_val, y_val[:, 1])],
            eval_metric='rmse',
            callbacks=[
                # early_stopping(50, verbose=False)
            ]
        )

        # 7. 예측 및 평가
        print("\n📊 평가 중...")
        y_pred_x = model_x.predict(X_val)
        y_pred_y = model_y.predict(X_val)
        y_pred = np.column_stack([y_pred_x, y_pred_y])

        # 유클리드 거리
        eucl_dist = euclidean_distance(y_val, y_pred)

        # MSE
        mse_x = mean_squared_error(y_val[:, 0], y_pred[:, 0])
        mse_y = mean_squared_error(y_val[:, 1], y_pred[:, 1])

        print(f"\n✅ Fold {fold+1} 결과:")
        print(f"  - 유클리드 거리: {eucl_dist:.4f}m")
        print(f"  - MSE X: {mse_x:.4f}")
        print(f"  - MSE Y: {mse_y:.4f}")

        fold_scores.append({
            'fold': fold + 1,
            'euclidean': eucl_dist,
            'mse_x': mse_x,
            'mse_y': mse_y
        })

        # Feature Importance (첫 번째 fold만)
        if fold == 0:
            print("\n📊 Feature Importance Top 20 (X 좌표):")
            importance_x = model_x.feature_importances_
            importance_df = pd.DataFrame({
                'feature': available_features,
                'importance': importance_x
            }).sort_values('importance', ascending=False)

            for i, row in importance_df.head(20).iterrows():
                print(f"  {row.name+1:2d}. {row['feature']:35s}: {row['importance']:8.1f}")

    # 8. 전체 결과 요약
    print("\n" + "="*80)
    print("  전체 결과 요약")
    print("="*80)

    scores_df = pd.DataFrame(fold_scores)
    print(f"\n평균 유클리드 거리: {scores_df['euclidean'].mean():.4f}m ± {scores_df['euclidean'].std():.4f}m")
    print(f"평균 MSE X: {scores_df['mse_x'].mean():.4f} ± {scores_df['mse_x'].std():.4f}")
    print(f"평균 MSE Y: {scores_df['mse_y'].mean():.4f} ± {scores_df['mse_y'].std():.4f}")

    print("\nFold별 상세:")
    for _, row in scores_df.iterrows():
        print(f"  Fold {int(row['fold'])}: {row['euclidean']:.4f}m")

    # 9. 최종 모델 학습 (전체 데이터)
    print("\n" + "="*80)
    print("  최종 모델 학습 (전체 데이터)")
    print("="*80)

    model_x_final = LGBMRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=8,
        num_leaves=63,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbose=-1
    )

    model_y_final = LGBMRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=8,
        num_leaves=63,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbose=-1
    )

    print("🔧 X 좌표 최종 모델 학습...")
    model_x_final.fit(X, y[:, 0])

    print("🔧 Y 좌표 최종 모델 학습...")
    model_y_final.fit(X, y[:, 1])

    # 10. 모델 저장
    print("\n💾 모델 저장...")
    with open('lightgbm_model_v2.pkl', 'wb') as f:
        pickle.dump({
            'model_x': model_x_final,
            'model_y': model_y_final,
            'feature_cols': available_features
        }, f)
    print("✅ 모델 저장 완료: lightgbm_model_v2.pkl")

    # 11. 성능 비교
    print("\n" + "="*80)
    print("  성능 비교 (V1 vs V2)")
    print("="*80)
    print("\n기존 LightGBM (V1):")
    print("  - Validation 평균: 0.93m")
    print(f"\n개선된 LightGBM (V2):")
    print(f"  - Validation 평균: {scores_df['euclidean'].mean():.4f}m")

    improvement = (0.93 - scores_df['euclidean'].mean()) / 0.93 * 100
    if improvement > 0:
        print(f"\n✅ 성능 개선: {improvement:.2f}% 향상!")
    else:
        print(f"\n⚠️  성능 변화: {improvement:.2f}%")

    print("\n" + "="*80)
    print("✅ V2 모델 학습 완료!")
    print("="*80)

if __name__ == "__main__":
    main()

