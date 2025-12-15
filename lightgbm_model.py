"""
K-League Pass Prediction - LightGBM Model

목표: XGBoost 대비 성능 비교 및 앙상블 준비
예상 성능: 1.0 ~ 1.3m (단독), 0.9 ~ 1.0m (앙상블)
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
import pickle
import warnings
warnings.filterwarnings('ignore')

def euclidean_distance(y_true, y_pred):
    """유클리드 거리 계산"""
    true_x, true_y = y_true[:, 0], y_true[:, 1]
    pred_x, pred_y = y_pred[:, 0], y_pred[:, 1]
    return np.mean(np.sqrt((true_x - pred_x)**2 + (true_y - pred_y)**2))

class LightGBMModel:
    def __init__(self):
        self.model_x = None
        self.model_y = None

    def train(self, X_train, y_train, X_val, y_val, params=None, verbose=True):
        """LightGBM 모델 학습"""

        if params is None:
            params = {
                'objective': 'regression',
                'metric': 'rmse',
                'boosting_type': 'gbdt',
                'num_leaves': 127,  # 참고 코드처럼 증가
                'learning_rate': 0.05,  # 더 작은 learning rate
                'min_data_in_leaf': 80,
                'feature_fraction': 0.8,
                'bagging_fraction': 0.8,
                'bagging_freq': 1,
                'verbose': -1,
                'random_state': 42,
                'n_estimators': 3000  # 참고 코드처럼 크게 증가
            }

        if verbose:
            print("=" * 80)
            print("  LightGBM 모델 학습")
            print("=" * 80)
            print(f"\n📊 학습 데이터: {X_train.shape}")
            print(f"📊 검증 데이터: {X_val.shape}\n")

        # end_x 예측 모델
        if verbose:
            print("🔵 end_x 모델 학습 중... (최대 3000 rounds)")

        self.model_x = lgb.LGBMRegressor(**params)
        self.model_x.fit(
            X_train, y_train[:, 0],
            eval_set=[(X_val, y_val[:, 0])],
            callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False)]  # 더 여유있게
        )

        if verbose:
            print(f"  → 최종 {self.model_x.best_iteration_} rounds 학습 완료")

        # end_y 예측 모델
        if verbose:
            print("🔴 end_y 모델 학습 중... (최대 3000 rounds)")

        self.model_y = lgb.LGBMRegressor(**params)
        self.model_y.fit(
            X_train, y_train[:, 1],
            eval_set=[(X_val, y_val[:, 1])],
            callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False)]
        )

        if verbose:
            print(f"  → 최종 {self.model_y.best_iteration_} rounds 학습 완료")
            print("✅ 학습 완료!\n")

    def predict(self, X):
        """예측"""
        pred_x = self.model_x.predict(X)
        pred_y = self.model_y.predict(X)
        return np.column_stack([pred_x, pred_y])

    def evaluate(self, X, y_true, verbose=True):
        """평가"""
        y_pred = self.predict(X)

        # 유클리드 거리
        eucl_dist = euclidean_distance(y_true, y_pred)

        # MSE (개별)
        mse_x = mean_squared_error(y_true[:, 0], y_pred[:, 0])
        mse_y = mean_squared_error(y_true[:, 1], y_pred[:, 1])

        if verbose:
            print(f"📊 평가 결과:")
            print(f"  - 유클리드 거리: {eucl_dist:.2f}m")
            print(f"  - MSE X: {mse_x:.2f}")
            print(f"  - MSE Y: {mse_y:.2f}")

        return eucl_dist, mse_x, mse_y

    def get_feature_importance(self, feature_names, top_n=20):
        """피처 중요도"""
        importance_x = self.model_x.feature_importances_
        importance_y = self.model_y.feature_importances_

        # 평균 중요도
        importance_avg = (importance_x + importance_y) / 2

        # DataFrame으로 정리
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance_x': importance_x,
            'importance_y': importance_y,
            'importance_avg': importance_avg
        }).sort_values('importance_avg', ascending=False)

        return importance_df.head(top_n)

    def save_model(self, filename='lightgbm_model.pkl'):
        """모델 저장"""
        with open(filename, 'wb') as f:
            pickle.dump({
                'model_x': self.model_x,
                'model_y': self.model_y
            }, f)
        print(f"✅ 모델 저장: {filename}")

    def load_model(self, filename='lightgbm_model.pkl'):
        """모델 로딩"""
        with open(filename, 'rb') as f:
            saved = pickle.load(f)
            self.model_x = saved['model_x']
            self.model_y = saved['model_y']
        print(f"✅ 모델 로딩: {filename}")

def main():
    """메인 실행 함수"""
    print("=" * 80)
    print("  K-League Pass Prediction - LightGBM Model")
    print("  목표: XGBoost 대비 성능 비교")
    print("=" * 80)
    print()

    # 🔥 실무 패턴: 유틸리티 함수 사용으로 코드 간소화
    from train_utils import (
        load_data_and_features,
        prepare_train_val_split,
        euclidean_distance,
        print_performance_summary,
        get_feature_group_importance,
        print_feature_group_importance
    )

    # 1. 데이터 및 피처 설정 로딩 (JSON 자동 사용)
    data, feature_cols, target_cols, config = load_data_and_features()

    # 2. Train/Val Split (게임 기반)
    X_train, y_train, X_val, y_val = prepare_train_val_split(
        data, feature_cols, target_cols, val_ratio=0.2
    )

    # 3. 모델 학습
    model = LightGBMModel()
    model.train(X_train, y_train, X_val, y_val, verbose=True)

    # 4. 평가 (유틸리티 함수 사용)
    print("\n" + "=" * 80)
    print("  모델 평가")
    print("=" * 80)

    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)

    train_eucl = euclidean_distance(y_train, y_train_pred)
    val_eucl = euclidean_distance(y_val, y_val_pred)

    print(f"\n[Train Set]")
    print(f"  - 유클리드 거리: {train_eucl:.2f}m")

    print(f"\n[Validation Set]")
    print(f"  - 유클리드 거리: {val_eucl:.2f}m")

    # 5. 성능 요약 (유틸리티 함수 사용)
    print_performance_summary(train_eucl, val_eucl)


    # 6. XGBoost와 비교
    print("\n" + "=" * 80)
    print("  XGBoost vs LightGBM 비교")
    print("=" * 80)

    xgb_val_eucl = 1.24  # XGBoost Validation 성능 (최신)

    print(f"\n📊 XGBoost:  {xgb_val_eucl:.2f}m")
    print(f"📊 LightGBM: {val_eucl:.2f}m")

    diff = xgb_val_eucl - val_eucl
    if diff > 0:
        print(f"✅ LightGBM이 {diff:.2f}m 더 좋습니다!")
    elif diff < 0:
        print(f"⚠️  XGBoost가 {-diff:.2f}m 더 좋습니다.")
    else:
        print(f"📊 동일한 성능입니다.")

    # 7. 피처 그룹별 중요도 분석
    group_importance = get_feature_group_importance(
        model.model_x, model.model_y, feature_cols, config
    )
    print_feature_group_importance(group_importance)

    # 8. 모델 저장
    print("\n" + "=" * 80)
    model.save_model('lightgbm_model.pkl')

    # 9. 최종 요약
    print("\n" + "=" * 80)
    print("  실행 완료!")
    print("=" * 80)

    ensemble_expected = (xgb_val_eucl + val_eucl) / 2 * 0.90  # 앙상블 효과 10%

    print(f"\n✅ LightGBM 모델 개발 완료!")
    print(f"   - Val 성능: {val_eucl:.2f}m")
    print(f"   - 피처 개수: {len(feature_cols)}")
    print(f"   - 모델 저장: lightgbm_model.pkl")
    print(f"\n📊 앙상블 예상 성능:")
    print(f"   - XGBoost + LightGBM: {ensemble_expected:.2f}m")
    if ensemble_expected < 1.0:
        print(f"   🎯 1.0m 이하 달성 가능!")

    return model, val_eucl

if __name__ == "__main__":
    model, val_eucl = main()

