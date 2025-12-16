"""
K-League Pass Prediction - CatBoost Model

목표: 3종 GBM 앙상블을 위한 CatBoost 모델 추가
예상 성능: 0.9 ~ 1.2m (단독), 0.85 ~ 0.90m (3종 앙상블)
"""

import pandas as pd
import numpy as np
from catboost import CatBoostRegressor, Pool
from sklearn.metrics import mean_squared_error
import pickle
import warnings
warnings.filterwarnings('ignore')

def euclidean_distance(y_true, y_pred):
    """유클리드 거리 계산"""
    true_x, true_y = y_true[:, 0], y_true[:, 1]
    pred_x, pred_y = y_pred[:, 0], y_pred[:, 1]
    return np.mean(np.sqrt((true_x - pred_x)**2 + (true_y - pred_y)**2))

class CatBoostModel:
    def __init__(self):
        self.model_x = None
        self.model_y = None

    def train(self, X_train, y_train, X_val, y_val,
              categorical_features=None, params=None, verbose=True):
        """CatBoost 모델 학습"""

        if params is None:
            params = {
                'iterations': 1000,  # 3000에서 1000으로 줄임 (빠른 학습)
                'learning_rate': 0.1,  # 0.05에서 0.1로 증가 (빠른 수렴)
                'depth': 6,  # 8에서 6으로 줄임 (빠른 학습)
                'l2_leaf_reg': 3,
                'min_data_in_leaf': 80,
                'random_strength': 1,
                'bagging_temperature': 1,
                'border_count': 128,  # 254에서 128로 줄임
                'loss_function': 'RMSE',
                'eval_metric': 'RMSE',
                'random_seed': 42,
                'verbose': 50,  # False에서 50으로 변경 (진행 상황 표시)
                'early_stopping_rounds': 50,  # 100에서 50으로 줄임 (빠른 조기 종료)
                'task_type': 'CPU',
                'thread_count': -1
            }

        if verbose:
            print("=" * 80)
            print("  CatBoost 모델 학습")
            print("=" * 80)
            print(f"\n📊 학습 데이터: {X_train.shape}")
            print(f"📊 검증 데이터: {X_val.shape}")
            if categorical_features:
                print(f"📊 범주형 피처: {len(categorical_features)}개\n")

        # 범주형 피처 인덱스 찾기
        cat_features_idx = None
        if categorical_features:
            cat_features_idx = [i for i, col in enumerate(X_train.columns)
                              if col in categorical_features]

        # end_x 예측 모델
        if verbose:
            print("🔵 end_x 모델 학습 중... (최대 3000 rounds)")

        train_pool_x = Pool(X_train, y_train[:, 0], cat_features=cat_features_idx)
        val_pool_x = Pool(X_val, y_val[:, 0], cat_features=cat_features_idx)

        self.model_x = CatBoostRegressor(**params)
        self.model_x.fit(
            train_pool_x,
            eval_set=val_pool_x,
            use_best_model=True,
            plot=False
        )

        if verbose:
            print(f"  → 최종 {self.model_x.best_iteration_} rounds 학습 완료")

        # end_y 예측 모델
        if verbose:
            print("🔴 end_y 모델 학습 중... (최대 3000 rounds)")

        train_pool_y = Pool(X_train, y_train[:, 1], cat_features=cat_features_idx)
        val_pool_y = Pool(X_val, y_val[:, 1], cat_features=cat_features_idx)

        self.model_y = CatBoostRegressor(**params)
        self.model_y.fit(
            train_pool_y,
            eval_set=val_pool_y,
            use_best_model=True,
            plot=False
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
        importance_x = self.model_x.get_feature_importance()
        importance_y = self.model_y.get_feature_importance()

        # 평균 중요도
        avg_importance = (importance_x + importance_y) / 2

        # 정규화
        avg_importance = avg_importance / avg_importance.sum()

        # DataFrame 생성
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': avg_importance,
            'importance_x': importance_x / importance_x.sum(),
            'importance_y': importance_y / importance_y.sum()
        })

        importance_df = importance_df.sort_values('importance', ascending=False)

        if top_n:
            importance_df = importance_df.head(top_n)

        return importance_df

    def save(self, filename='catboost_model.pkl'):
        """모델 저장"""
        with open(filename, 'wb') as f:
            pickle.dump({
                'model_x': self.model_x,
                'model_y': self.model_y
            }, f)
        print(f"✅ 모델 저장: {filename}")

    @staticmethod
    def load(filename='catboost_model.pkl'):
        """모델 로딩"""
        with open(filename, 'rb') as f:
            saved = pickle.load(f)

        model = CatBoostModel()
        model.model_x = saved['model_x']
        model.model_y = saved['model_y']

        print(f"✅ 모델 로딩: {filename}")
        return model


def main():
    """메인 실행 함수"""
    print("=" * 80)
    print("  K-League Pass Prediction - CatBoost Model")
    print("=" * 80)
    print("목표: 3종 GBM 앙상블 준비")
    print("=" * 80 + "\n")

    # train_utils 사용하여 데이터 로딩
    from train_utils import load_data_and_features, prepare_train_val_split
    from feature_config import FeatureConfig

    # 1. 데이터 및 피처 설정 로딩
    data, feature_cols, target_cols, config = load_data_and_features()

    print(f"\n📊 피처 정보:")
    print(f"  - 설정 파일 피처: {len(feature_cols)}개")
    print(f"  - 사용 가능 피처: {len([c for c in feature_cols if c in data.columns])}개")
    print(f"  - 타겟: {', '.join(target_cols)}")

    # 2. Train/Val Split
    # 게임 기반 분할
    games = data['game_id'].unique()
    np.random.seed(42)
    np.random.shuffle(games)

    n_val_games = int(len(games) * 0.2)
    val_games = games[:n_val_games]

    val_mask = data['game_id'].isin(val_games)
    train_mask = ~val_mask

    # DataFrame 형태로 유지
    X_train = data.loc[train_mask, feature_cols].fillna(0)
    y_train = data.loc[train_mask, target_cols].values
    X_val = data.loc[val_mask, feature_cols].fillna(0)
    y_val = data.loc[val_mask, target_cols].values

    print(f"\n📊 Train/Val Split (게임 기반)...")
    print(f"  - Train: {len(games) - n_val_games} 게임, {len(X_train):,} 에피소드")
    print(f"  - Val: {n_val_games} 게임, {len(X_val):,} 에피소드")
    print(f"  - 피처: {len(feature_cols)}개\n")

    # 3. 범주형 피처 추출 및 타입 변환
    categorical_features = config.get_categorical_features()
    categorical_features = [f for f in categorical_features if f in feature_cols]

    # 범주형 피처를 integer로 변환 (CatBoost 요구사항)
    for col in categorical_features:
        if col in X_train.columns:
            X_train[col] = X_train[col].astype(int)
            X_val[col] = X_val[col].astype(int)

    # 4. CatBoost 모델 학습
    model = CatBoostModel()
    model.train(
        X_train, y_train, X_val, y_val,
        categorical_features=categorical_features,
        verbose=True
    )

    # 5. 평가
    print("\n" + "=" * 80)
    print("  모델 평가")
    print("=" * 80 + "\n")

    print("[Train Set]")
    train_eucl, train_mse_x, train_mse_y = model.evaluate(X_train, y_train, verbose=False)
    print(f"  - 유클리드 거리: {train_eucl:.2f}m\n")

    print("[Validation Set]")
    val_eucl, val_mse_x, val_mse_y = model.evaluate(X_val, y_val)

    # 6. 성능 요약
    print("\n" + "=" * 80)
    print("  성능 요약")
    print("=" * 80 + "\n")

    print(f"📊 유클리드 거리:")
    print(f"  - Train: {train_eucl:.2f}m")
    print(f"  - Val: {val_eucl:.2f}m")

    baseline = 20.37
    improvement = baseline - val_eucl
    improvement_pct = (improvement / baseline) * 100

    print(f"\n📊 베이스라인 대비:")
    print(f"  - 베이스라인: {baseline}m")
    print(f"  - 개선: {improvement:.2f}m (+{improvement_pct:.1f}%)")

    if val_eucl < baseline:
        print(f"  ✅ 베이스라인보다 {improvement:.2f}m 개선!")

    target = 18.0
    print(f"\n📊 목표 달성:")
    print(f"  - 목표: < {target:.2f}m")
    print(f"  - 현재: {val_eucl:.2f}m")

    if val_eucl < target:
        print(f"  🎯 목표 달성! ({val_eucl:.2f}m < {target:.2f}m)")

    print("=" * 80)

    # 7. 다른 모델들과 비교
    print("\n" + "=" * 80)
    print("  기존 모델과 비교")
    print("=" * 80 + "\n")

    xgb_score = 1.24  # 이전 결과
    lgb_score = 0.93  # 이전 결과

    print(f"📊 XGBoost:  {xgb_score:.2f}m")
    print(f"📊 LightGBM: {lgb_score:.2f}m")
    print(f"📊 CatBoost: {val_eucl:.2f}m")

    best_single = min(xgb_score, lgb_score, val_eucl)
    if val_eucl == best_single:
        print(f"✅ CatBoost가 최고 성능!")
    elif val_eucl < xgb_score:
        print(f"✅ CatBoost가 XGBoost보다 {xgb_score - val_eucl:.2f}m 더 좋습니다!")

    # 3종 앙상블 예상 성능
    expected_ensemble = best_single * 0.95  # 보통 5% 정도 개선
    print(f"\n📊 3종 앙상블 예상 성능:")
    print(f"   - XGBoost + LightGBM + CatBoost: ~{expected_ensemble:.2f}m")
    if expected_ensemble < 0.9:
        print(f"   🎯 0.9m 이하 달성 가능!")

    # 8. 피처 중요도 (Top 20)
    print("\n" + "=" * 80)
    print("  피처 중요도 (Top 20)")
    print("=" * 80 + "\n")

    importance_df = model.get_feature_importance(feature_cols, top_n=20)

    for idx, row in importance_df.iterrows():
        print(f"{row['feature']:30s}: {row['importance']:.4f}")

    # 9. 모델 저장
    print("\n" + "=" * 80)
    model.save('catboost_model.pkl')

    # 10. 최종 요약
    print("\n" + "=" * 80)
    print("  실행 완료!")
    print("=" * 80 + "\n")

    print(f"✅ CatBoost 모델 개발 완료!")
    print(f"   - Val 성능: {val_eucl:.2f}m")
    print(f"   - 피처 개수: {len(feature_cols)}")
    print(f"   - 모델 저장: catboost_model.pkl")

    print(f"\n📊 다음 단계:")
    print(f"   1. 3종 앙상블 구성 (XGBoost + LightGBM + CatBoost)")
    print(f"   2. 최적 가중치 탐색")
    print(f"   3. Test 데이터 예측")
    print(f"   4. 최종 제출")

    return model, val_eucl


if __name__ == "__main__":
    model, final_score = main()

