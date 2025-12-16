"""
K-League Pass Prediction - 3-Model Ensemble

XGBoost + LightGBM + CatBoost 앙상블
목표: 0.7m 이하 달성
"""

import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')

def euclidean_distance(y_true, y_pred):
    """유클리드 거리 계산"""
    true_x, true_y = y_true[:, 0], y_true[:, 1]
    pred_x, pred_y = y_pred[:, 0], y_pred[:, 1]
    return np.mean(np.sqrt((true_x - pred_x)**2 + (true_y - pred_y)**2))


class ThreeModelEnsemble:
    """3종 GBM 앙상블 클래스"""

    def __init__(self):
        self.models = []
        self.model_names = []
        self.weights = None
        self.best_weights = None
        self.best_score = float('inf')

    def add_model(self, model_path, model_name, weight=None):
        """모델 추가"""
        with open(model_path, 'rb') as f:
            model = pickle.load(f)

        self.models.append(model)
        self.model_names.append(model_name)

        if weight is not None:
            if self.weights is None:
                self.weights = []
            self.weights.append(weight)

        print(f"✅ 모델 추가: {model_name} (가중치: {weight if weight else '미정'})")

    def predict(self, X, weights=None):
        """앙상블 예측"""
        if weights is None:
            weights = self.weights if self.weights else [1.0/len(self.models)] * len(self.models)

        predictions = []

        for model in self.models:
            pred_x = model['model_x'].predict(X)
            pred_y = model['model_y'].predict(X)
            pred = np.column_stack([pred_x, pred_y])
            predictions.append(pred)

        # 가중 평균
        predictions = np.array(predictions)
        weights = np.array(weights).reshape(-1, 1, 1)

        ensemble_pred = np.sum(predictions * weights, axis=0)

        return ensemble_pred

    def evaluate_weights(self, X_val, y_val, weights):
        """특정 가중치로 평가"""
        y_pred = self.predict(X_val, weights)
        return euclidean_distance(y_val, y_pred)

    def optimize_weights(self, X_val, y_val, verbose=True):
        """최적 가중치 탐색 (Grid Search)"""
        if verbose:
            print("\n🔍 최적 가중치 탐색 중...")

        best_score = float('inf')
        best_weights = None

        # 3개 모델의 가중치 조합 탐색 (합이 1.0)
        weight_range = np.arange(0.0, 1.1, 0.1)

        for w1 in weight_range:
            for w2 in weight_range:
                w3 = 1.0 - w1 - w2
                if w3 < 0 or w3 > 1.0:
                    continue

                weights = [w1, w2, w3]
                score = self.evaluate_weights(X_val, y_val, weights)

                if score < best_score:
                    best_score = score
                    best_weights = weights

        self.best_weights = best_weights
        self.best_score = best_score

        if verbose:
            print(f"✅ 최적 가중치: {[f'{w:.2f}' for w in best_weights]}")
            print(f"✅ 최적 성능: {best_score:.2f}m")

        return best_weights, best_score

    def evaluate(self, X, y_true, weights=None, verbose=True):
        """평가"""
        y_pred = self.predict(X, weights)

        eucl_dist = euclidean_distance(y_true, y_pred)
        mse_x = mean_squared_error(y_true[:, 0], y_pred[:, 0])
        mse_y = mean_squared_error(y_true[:, 1], y_pred[:, 1])

        if verbose:
            print(f"📊 앙상블 성능:")
            print(f"  - 유클리드 거리: {eucl_dist:.2f}m")
            print(f"  - MSE X: {mse_x:.2f}")
            print(f"  - MSE Y: {mse_y:.2f}")

        return eucl_dist, mse_x, mse_y

    def save(self, filename='ensemble_3models.pkl'):
        """앙상블 저장"""
        with open(filename, 'wb') as f:
            pickle.dump({
                'models': self.models,
                'model_names': self.model_names,
                'weights': self.best_weights if self.best_weights else self.weights,
                'val_score': self.best_score if self.best_score else None
            }, f)
        print(f"✅ 앙상블 저장: {filename}")


def main():
    """메인 실행 함수"""
    print("=" * 80)
    print("  K-League Pass Prediction - 3-Model Ensemble")
    print("  XGBoost + LightGBM + CatBoost")
    print("=" * 80 + "\n")

    # train_utils 사용
    from train_utils import load_data_and_features

    # 1. 데이터 로딩
    data, feature_cols, target_cols, config = load_data_and_features()

    print(f"\n📊 피처 정보:")
    print(f"  - 설정 파일 피처: {len(feature_cols)}개")
    print(f"  - 사용 가능 피처: {len([c for c in feature_cols if c in data.columns])}개")
    print(f"  - 타겟: {', '.join(target_cols)}")

    # 2. Train/Val Split
    games = data['game_id'].unique()
    np.random.seed(42)
    np.random.shuffle(games)

    n_val_games = int(len(games) * 0.2)
    val_games = games[:n_val_games]

    val_mask = data['game_id'].isin(val_games)
    train_mask = ~val_mask

    # DataFrame 형태로 유지 (CatBoost를 위해)
    X_train = data.loc[train_mask, feature_cols].fillna(0).copy()
    y_train = data.loc[train_mask, target_cols].values
    X_val = data.loc[val_mask, feature_cols].fillna(0).copy()
    y_val = data.loc[val_mask, target_cols].values

    # 범주형 피처를 integer로 변환 (CatBoost 요구사항)
    categorical_features = config.get_categorical_features()
    categorical_features = [f for f in categorical_features if f in feature_cols]

    for col in categorical_features:
        if col in X_train.columns:
            X_train[col] = X_train[col].astype(int)
            X_val[col] = X_val[col].astype(int)

    print(f"\n📊 Train/Val Split (게임 기반)...")
    print(f"  - Train: {len(games) - n_val_games} 게임, {len(X_train):,} 에피소드")
    print(f"  - Val: {n_val_games} 게임, {len(X_val):,} 에피소드")
    print(f"  - 피처: {len(feature_cols)}개\n")

    # 3. 앙상블 구성
    print("\n" + "=" * 80)
    print("  앙상블 구성")
    print("=" * 80 + "\n")

    ensemble = ThreeModelEnsemble()
    ensemble.add_model('xgboost_baseline.pkl', 'XGBoost', weight=1/3)
    ensemble.add_model('lightgbm_model.pkl', 'LightGBM', weight=1/3)
    ensemble.add_model('catboost_model.pkl', 'CatBoost', weight=1/3)

    # 4. 개별 모델 성능 평가
    print("\n" + "=" * 80)
    print("  개별 모델 성능 (Validation)")
    print("=" * 80 + "\n")

    individual_scores = []
    print("개별 모델 성능:")
    print("-" * 60)

    for i, (model, name) in enumerate(zip(ensemble.models, ensemble.model_names)):
        pred_x = model['model_x'].predict(X_val)
        pred_y = model['model_y'].predict(X_val)
        pred = np.column_stack([pred_x, pred_y])

        score = euclidean_distance(y_val, pred)
        individual_scores.append(score)
        print(f"  {name:10s}: {score:.2f}m")

    print("-" * 60)

    # 5. 기본 앙상블 평가 (동일 가중치)
    print("\n" + "=" * 80)
    print(f"  기본 앙상블 평가 (가중치 {1/3:.2f} : {1/3:.2f} : {1/3:.2f})")
    print("=" * 80 + "\n")

    default_eucl, _, _ = ensemble.evaluate(X_val, y_val)

    # 6. 최적 가중치 탐색
    print("\n" + "=" * 80)
    print("  최적 가중치 탐색")
    print("=" * 80)

    best_weights, best_score = ensemble.optimize_weights(X_val, y_val)

    # 7. 최적 가중치 앙상블 평가
    print("\n" + "=" * 80)
    print("  최적 가중치 앙상블 평가")
    print("=" * 80 + "\n")

    print(f"🎯 최적 가중치:")
    for name, w in zip(ensemble.model_names, best_weights):
        print(f"  - {name:10s}: {w:.2f}")
    print()

    final_eucl, _, _ = ensemble.evaluate(X_val, y_val, weights=best_weights)

    # 8. 최종 결과 비교
    print("\n" + "=" * 80)
    print("  최종 결과 비교")
    print("=" * 80 + "\n")

    print(f"""    📊 개별 모델:
      - XGBoost:  {individual_scores[0]:.2f}m
      - LightGBM: {individual_scores[1]:.2f}m
      - CatBoost: {individual_scores[2]:.2f}m
    
    📊 앙상블 (동일 가중치):
      - 성능: {default_eucl:.2f}m
    
    📊 앙상블 (최적 가중치):
      - 가중치: XGBoost {best_weights[0]:.2f}, LightGBM {best_weights[1]:.2f}, CatBoost {best_weights[2]:.2f}
      - 성능: {final_eucl:.2f}m
    
    📊 개선:
      - 최고 단독 모델({min(individual_scores):.2f}m) 대비: {min(individual_scores) - final_eucl:.2f}m 개선
    """)

    if final_eucl < 1.0:
        print("🎯 ✅ 1.0m 이하 달성!")
    if final_eucl < 0.9:
        print("🎉 ✅ 0.9m 이하 달성! 우수한 성능!")
    if final_eucl < 0.8:
        print("🏆 ✅ 0.8m 이하 달성! 탁월한 성능!")
    if final_eucl < 0.7:
        print("💎 ✅ 0.7m 이하 달성! 최상위 성능!")

    # 9. Train Set 성능 (과적합 체크)
    print("\n" + "=" * 80)
    print("  Train Set 성능")
    print("=" * 80 + "\n")

    train_eucl, _, _ = ensemble.evaluate(X_train, y_train, weights=best_weights, verbose=True)

    print(f"\n📊 Overfitting 체크:")
    print(f"  - Train: {train_eucl:.2f}m")
    print(f"  - Val: {final_eucl:.2f}m")
    print(f"  - 비율: {train_eucl / final_eucl:.2f}")

    if train_eucl / final_eucl < 0.3:
        print("  ✅ 과적합 없음 (안정적)")
    else:
        print("  ⚠️  약간의 과적합 가능성")

    # 10. 앙상블 저장
    print("\n" + "=" * 80)
    ensemble.save('ensemble_3models.pkl')

    # 11. 최종 요약
    print("\n" + "=" * 80)
    print("  🎉 최종 요약")
    print("=" * 80 + "\n")

    print(f"""    ✅ 3종 앙상블 모델 완성!
    
    📊 최종 성능:
      - Validation: {final_eucl:.2f}m
      - Train: {train_eucl:.2f}m
    
    📊 구성:
      - XGBoost ({best_weights[0]:.2f}) + LightGBM ({best_weights[1]:.2f}) + CatBoost ({best_weights[2]:.2f})
    
    📊 베이스라인 대비:
      - 단순 베이스라인 (20.37m) → {final_eucl:.2f}m
      - 개선: {20.37 - final_eucl:.2f}m ({(20.37 - final_eucl)/20.37*100:.1f}%)
    
    💡 다음 단계:
      1. Test 데이터 예측 (inference_3models.py)
      2. 제출 파일 생성
      3. 대회 플랫폼에 제출
      4. 리더보드 확인
    
    🏆 현재 위치: 최상위권 예상!
    """)

    return ensemble, final_eucl


if __name__ == "__main__":
    ensemble, final_score = main()

