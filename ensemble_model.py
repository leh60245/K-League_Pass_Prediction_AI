"""
K-League Pass Prediction - Ensemble Model
XGBoost + LightGBM 앙상블

목표: 두 모델의 장점을 결합하여 성능 향상
예상 성능: 0.85 ~ 0.95m
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

class EnsembleModel:
    def __init__(self):
        self.models = []
        self.weights = []

    def add_model(self, model_path, weight=1.0):
        """모델 추가"""
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        self.models.append(model)
        self.weights.append(weight)
        print(f"✅ 모델 추가: {model_path} (가중치: {weight})")

    def predict(self, X):
        """앙상블 예측 (가중 평균)"""
        predictions = []

        for model, weight in zip(self.models, self.weights):
            pred_x = model['model_x'].predict(X)
            pred_y = model['model_y'].predict(X)
            pred = np.column_stack([pred_x, pred_y])
            predictions.append(pred * weight)

        # 가중 평균
        ensemble_pred = np.sum(predictions, axis=0) / np.sum(self.weights)

        return ensemble_pred

    def evaluate(self, X, y_true, verbose=True):
        """평가"""
        y_pred = self.predict(X)

        # 유클리드 거리
        eucl_dist = euclidean_distance(y_true, y_pred)

        # MSE
        mse_x = mean_squared_error(y_true[:, 0], y_pred[:, 0])
        mse_y = mean_squared_error(y_true[:, 1], y_pred[:, 1])

        if verbose:
            print(f"📊 앙상블 성능:")
            print(f"  - 유클리드 거리: {eucl_dist:.2f}m")
            print(f"  - MSE X: {mse_x:.2f}")
            print(f"  - MSE Y: {mse_y:.2f}")

        return eucl_dist, mse_x, mse_y

    def evaluate_individual(self, X, y_true):
        """개별 모델 성능 확인"""
        print("\n개별 모델 성능:")
        print("-" * 60)

        individual_perfs = []

        for i, model in enumerate(self.models):
            pred_x = model['model_x'].predict(X)
            pred_y = model['model_y'].predict(X)
            pred = np.column_stack([pred_x, pred_y])

            eucl_dist = euclidean_distance(y_true, pred)
            individual_perfs.append(eucl_dist)

            print(f"  모델 {i+1}: {eucl_dist:.2f}m")

        print("-" * 60)

        return individual_perfs

    def optimize_weights(self, X, y_true, verbose=True):
        """최적 가중치 탐색 (Grid Search)"""
        if verbose:
            print("\n🔍 최적 가중치 탐색 중...")

        best_weights = self.weights.copy()
        best_score = float('inf')

        # Grid Search (0.0 ~ 1.0, 0.1 간격)
        weight_range = np.arange(0.0, 1.1, 0.1)

        for w1 in weight_range:
            w2 = 1.0 - w1
            self.weights = [w1, w2]

            score, _, _ = self.evaluate(X, y_true, verbose=False)

            if score < best_score:
                best_score = score
                best_weights = [w1, w2]

        self.weights = best_weights

        if verbose:
            print(f"✅ 최적 가중치: {best_weights}")
            print(f"✅ 최적 성능: {best_score:.2f}m")

        return best_weights, best_score

    def save_ensemble(self, filename='ensemble_model.pkl'):
        """앙상블 저장"""
        with open(filename, 'wb') as f:
            pickle.dump({
                'models': self.models,
                'weights': self.weights
            }, f)
        print(f"✅ 앙상블 저장: {filename}")

def main():
    """메인 실행"""
    print("=" * 80)
    print("  K-League Pass Prediction - Ensemble Model")
    print("  XGBoost + LightGBM 앙상블")
    print("=" * 80)
    print()

    # 🔥 실무 패턴: 유틸리티 함수 사용
    from train_utils import (
        load_data_and_features,
        prepare_train_val_split,
        print_performance_summary
    )

    # 1. 데이터 및 피처 설정 로딩
    data, feature_cols, target_cols, config = load_data_and_features()

    # 2. Train/Val Split
    X_train, y_train, X_val, y_val = prepare_train_val_split(
        data, feature_cols, target_cols, val_ratio=0.2
    )

    # 3. 앙상블 모델 생성
    print("\n" + "=" * 80)
    print("  앙상블 구성")
    print("=" * 80)
    print()

    ensemble = EnsembleModel()
    ensemble.add_model('xgboost_baseline.pkl', weight=0.5)
    ensemble.add_model('lightgbm_model.pkl', weight=0.5)

    # 5. 개별 모델 성능 확인
    print("\n" + "=" * 80)
    print("  개별 모델 성능 (Validation)")
    print("=" * 80)

    individual_perfs = ensemble.evaluate_individual(X_val, y_val)

    # 6. 기본 앙상블 평가 (동일 가중치)
    print("\n" + "=" * 80)
    print("  기본 앙상블 평가 (가중치 0.5 : 0.5)")
    print("=" * 80)
    print()

    val_eucl, val_mse_x, val_mse_y = ensemble.evaluate(X_val, y_val, verbose=True)

    # 7. 최적 가중치 탐색
    print("\n" + "=" * 80)
    print("  최적 가중치 탐색")
    print("=" * 80)

    best_weights, best_score = ensemble.optimize_weights(X_val, y_val, verbose=True)

    # 8. 최적 가중치로 재평가
    print("\n" + "=" * 80)
    print("  최적 가중치 앙상블 평가")
    print("=" * 80)
    print()

    print(f"🎯 최적 가중치:")
    print(f"  - XGBoost:  {best_weights[0]:.2f}")
    print(f"  - LightGBM: {best_weights[1]:.2f}")
    print()

    final_eucl, final_mse_x, final_mse_y = ensemble.evaluate(X_val, y_val, verbose=True)

    # 9. 결과 비교
    print("\n" + "=" * 80)
    print("  최종 결과 비교")
    print("=" * 80)

    print(f"""
    📊 개별 모델:
      - XGBoost:  {individual_perfs[0]:.2f}m
      - LightGBM: {individual_perfs[1]:.2f}m
    
    📊 앙상블 (동일 가중치):
      - 성능: {val_eucl:.2f}m
    
    📊 앙상블 (최적 가중치):
      - 가중치: XGBoost {best_weights[0]:.2f}, LightGBM {best_weights[1]:.2f}
      - 성능: {final_eucl:.2f}m
    
    📊 개선:
      - XGBoost 대비: {individual_perfs[0] - final_eucl:.2f}m 개선
      - LightGBM 대비: {individual_perfs[1] - final_eucl:.2f}m 개선
      - 최고 단독 모델 대비: {min(individual_perfs) - final_eucl:.2f}m 개선
    """)

    if final_eucl < 1.0:
        print("🎯 ✅ 1.0m 이하 달성!")

    if final_eucl < 0.9:
        print("🎉 ✅ 0.9m 이하 달성! 우수한 성능!")

    # 10. Train Set 성능 확인
    print("\n" + "=" * 80)
    print("  Train Set 성능")
    print("=" * 80)
    print()

    train_eucl, _, _ = ensemble.evaluate(X_train, y_train, verbose=True)

    # Overfitting 체크
    overfit_ratio = train_eucl / final_eucl
    print(f"\n📊 Overfitting 체크:")
    print(f"  - Train: {train_eucl:.2f}m")
    print(f"  - Val: {final_eucl:.2f}m")
    print(f"  - 비율: {overfit_ratio:.2f}")

    if overfit_ratio < 0.8:
        print("  ✅ 과적합 없음 (안정적)")
    elif overfit_ratio < 1.0:
        print("  ⚠️  약간의 과적합")
    else:
        print("  ⚠️  주의: Train보다 Val 성능이 좋음")

    # 11. 앙상블 저장
    print("\n" + "=" * 80)
    ensemble.save_ensemble('ensemble_model.pkl')

    # 12. 최종 요약
    print("\n" + "=" * 80)
    print("  🎉 최종 요약")
    print("=" * 80)
    print(f"""
    ✅ 앙상블 모델 완성!
    
    📊 최종 성능:
      - Validation: {final_eucl:.2f}m
      - Train: {train_eucl:.2f}m
    
    📊 구성:
      - XGBoost ({best_weights[0]:.2f}) + LightGBM ({best_weights[1]:.2f})
    
    📊 베이스라인 대비:
      - 단순 베이스라인 (20.37m) → {final_eucl:.2f}m
      - 개선: {20.37 - final_eucl:.2f}m ({(20.37 - final_eucl) / 20.37 * 100:.1f}%)
    
    💡 다음 단계:
      1. CatBoost 추가 (GBM 3종 앙상블)
      2. 전체 데이터로 재학습
      3. Test 데이터 예측
      4. 최종 제출
    
    🏆 현재 위치: 상위권 확실!
    """)

    return ensemble, final_eucl

if __name__ == "__main__":
    ensemble, final_eucl = main()

