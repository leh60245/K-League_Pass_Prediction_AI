"""
K-League Pass Prediction - 고급 전술 피처를 활용한 모델 학습

새로운 피처:
- 압박 강도 (Pressure Intensity)
- 공간 창출 (Space Creation)
- 진행 방향성 (Directional Consistency)
- 경로 효율성 (Path Efficiency)
- 골 각도 (Shooting Angle)
- 템포 분석 (Tempo Analysis)
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error
import pickle
from preprocessing import DataPreprocessor
import os
from datetime import datetime

def train_model_with_tactical_features():
    """고급 전술 피처를 활용한 모델 학습"""

    print("=" * 80)
    print("  K-League Pass Prediction - 고급 전술 피처 모델 학습")
    print("=" * 80)
    print()

    # 1. 데이터 로딩
    print("📊 전처리된 데이터 로딩 중...")
    if os.path.exists('processed_train_data.csv'):
        processed_data = pd.read_csv('processed_train_data.csv')
        print(f"✅ 데이터 로딩 완료: {processed_data.shape}")
    else:
        print("❌ processed_train_data.csv 파일이 없습니다. preprocessing.py를 먼저 실행하세요.")
        return

    # 2. Preprocessor 로딩
    print("\n🔧 Preprocessor 로딩 중...")
    preprocessor = DataPreprocessor(data_dir='./data')
    if os.path.exists('preprocessor.pkl'):
        preprocessor.load_preprocessor('preprocessor.pkl')
    else:
        print("❌ preprocessor.pkl 파일이 없습니다.")
        return

    # 3. 피처 컬럼 가져오기
    feature_cols = preprocessor.get_feature_columns()

    # 실제 존재하는 피처만 선택
    available_features = [col for col in feature_cols if col in processed_data.columns]
    print(f"\n✅ 사용 가능한 피처: {len(available_features)}개")

    # 4. 타겟 변수
    target_cols = ['end_x', 'end_y']

    # 5. Train/Val Split 생성
    print("\n📊 Train/Val Split 생성 중...")
    splits = preprocessor.create_train_val_split(processed_data, n_splits=5, verbose=False)

    # Fold 0만 사용 (빠른 테스트)
    fold = 0
    train_idx = splits[fold]['train_idx']
    val_idx = splits[fold]['val_idx']

    X_train = processed_data.loc[train_idx, available_features]
    y_train = processed_data.loc[train_idx, target_cols]
    X_val = processed_data.loc[val_idx, available_features]
    y_val = processed_data.loc[val_idx, target_cols]

    print(f"✅ Train: {X_train.shape}, Val: {X_val.shape}")

    # 6. 결측치 확인 및 처리
    print("\n🔧 결측치 처리 중...")
    X_train = X_train.fillna(0)
    X_val = X_val.fillna(0)
    print(f"✅ Train NaN: {X_train.isna().sum().sum()}, Val NaN: {X_val.isna().sum().sum()}")

    # 7. 모델 학습 (X, Y 좌표 각각)
    print("\n" + "=" * 80)
    print("  모델 학습 시작")
    print("=" * 80)

    models = {}
    results = {}

    for target in ['end_x', 'end_y']:
        print(f"\n🎯 타겟: {target}")
        print("-" * 40)

        # XGBoost 파라미터 (고급 설정)
        params = {
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',  # 여기로 이동
            'n_estimators': 500,
            'max_depth': 8,
            'learning_rate': 0.05,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_weight': 3,
            'gamma': 0.1,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0,
            'random_state': 42,
            'tree_method': 'hist',
            'n_jobs': -1
        }

        # 모델 초기화
        model = xgb.XGBRegressor(**params)

        # 학습
        print("🔄 학습 중...")
        model.fit(
            X_train, y_train[target],
            eval_set=[(X_train, y_train[target]), (X_val, y_val[target])],
            verbose=50
        )

        # 예측
        train_pred = model.predict(X_train)
        val_pred = model.predict(X_val)

        # 평가
        train_rmse = np.sqrt(mean_squared_error(y_train[target], train_pred))
        val_rmse = np.sqrt(mean_squared_error(y_val[target], val_pred))

        print(f"\n📊 성능:")
        print(f"  Train RMSE: {train_rmse:.4f}")
        print(f"  Val RMSE:   {val_rmse:.4f}")

        models[target] = model
        results[target] = {
            'train_rmse': train_rmse,
            'val_rmse': val_rmse
        }

    # 8. 피처 중요도 분석
    print("\n" + "=" * 80)
    print("  피처 중요도 분석 (Top 20)")
    print("=" * 80)

    for target in ['end_x', 'end_y']:
        print(f"\n🎯 {target}:")
        print("-" * 40)

        model = models[target]
        importance = model.feature_importances_

        # 중요도 정렬
        feature_importance = pd.DataFrame({
            'feature': available_features,
            'importance': importance
        }).sort_values('importance', ascending=False)

        # Top 20 출력
        for i, row in feature_importance.head(20).iterrows():
            print(f"  {row['feature']:30s}: {row['importance']:.4f}")

        # 전술 피처 중요도 따로 분석
        tactical_features = [
            'shooting_angle', 'goal_approach',
            'local_pressure', 'weighted_pressure', 'event_density',
            'distance_change_rate', 'vertical_spread', 'attack_width',
            'forward_momentum', 'pass_angle_change',
            'direction_consistency', 'path_efficiency',
            'dist_from_team_center', 'match_phase',
            'velocity', 'acceleration', 'tempo_change'
        ]

        tactical_importance = feature_importance[
            feature_importance['feature'].isin(tactical_features)
        ]

        if len(tactical_importance) > 0:
            print(f"\n⚽ 전술 피처 중요도:")
            for i, row in tactical_importance.head(10).iterrows():
                print(f"  {row['feature']:30s}: {row['importance']:.4f}")

    # 9. 모델 저장
    print("\n" + "=" * 80)
    print("  모델 저장")
    print("=" * 80)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_dir = f"models/tactical_features_{timestamp}"
    os.makedirs(model_dir, exist_ok=True)

    # 각 모델 저장
    for target, model in models.items():
        model_path = os.path.join(model_dir, f'{target}_model.pkl')
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        print(f"✅ {target} 모델 저장: {model_path}")

    # 결과 저장
    results_path = os.path.join(model_dir, 'performance.txt')
    with open(results_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("  K-League Pass Prediction - 고급 전술 피처 모델 성능\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"학습 시간: {timestamp}\n")
        f.write(f"피처 개수: {len(available_features)}\n")
        f.write(f"Train 샘플: {len(X_train)}\n")
        f.write(f"Val 샘플: {len(X_val)}\n\n")

        for target in ['end_x', 'end_y']:
            f.write(f"\n{target}:\n")
            f.write(f"  Train RMSE: {results[target]['train_rmse']:.4f}\n")
            f.write(f"  Val RMSE:   {results[target]['val_rmse']:.4f}\n")

        f.write("\n\n사용된 전술 피처:\n")
        for feature in tactical_features:
            if feature in available_features:
                f.write(f"  - {feature}\n")

    print(f"✅ 성능 결과 저장: {results_path}")

    print("\n" + "=" * 80)
    print("✅ 학습 완료!")
    print("=" * 80)

    return models, results, available_features

if __name__ == "__main__":
    models, results, features = train_model_with_tactical_features()

    print("\n📊 최종 요약:")
    print("-" * 80)
    print(f"총 피처 수: {len(features)}")
    print(f"end_x Val RMSE: {results['end_x']['val_rmse']:.4f}")
    print(f"end_y Val RMSE: {results['end_y']['val_rmse']:.4f}")
    print(f"평균 RMSE: {(results['end_x']['val_rmse'] + results['end_y']['val_rmse']) / 2:.4f}")

