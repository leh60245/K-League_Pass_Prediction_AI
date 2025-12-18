"""
LightGBM V5 - Optuna 하이퍼파라미터 최적화

개선사항:
- fillna(0) 완전 제거 (NaN 유지로 LightGBM 최적화)
- 최적 모델 발견 시 즉시 저장 (Ctrl+C 대비)
- Optuna DB 기반 중단 후 재개 기능
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GroupKFold
import optuna
from optuna.samplers import TPESampler
import pickle
import warnings
import os
from datetime import datetime
warnings.filterwarnings('ignore')


def euclidean_distance(y_true, y_pred):
    """유클리드 거리 계산"""
    distances = np.sqrt((y_true[:, 0] - y_pred[:, 0])**2 +
                       (y_true[:, 1] - y_pred[:, 1])**2)
    return distances.mean()


class LightGBMOptimizer:
    def __init__(self, X_train, y_train_x, y_train_y, game_ids, cat_features):
        self.X_train = X_train
        self.y_train_x = y_train_x
        self.y_train_y = y_train_y
        self.game_ids = game_ids
        self.cat_features = cat_features  # 범주형 변수 리스트 받기
        self.best_score = float('inf')
        self.best_models_x = []  # [추가] 최적 모델 저장
        self.best_models_y = []
        self.best_params = None
        self.best_fold_scores = []

    def objective(self, trial):
        """Optuna objective function"""

        # 하이퍼파라미터 탐색 공간
        params = {
            'objective': 'regression',
            'metric': 'rmse',
            'verbosity': -1,
            'boosting_type': 'gbdt',
            'n_jobs': -1,  # CPU 풀가동

            # 학습률 (정밀 탐색)
            'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.1, log=True),

            # 트리 구조 (과적합 방지 포함)
            'num_leaves': trial.suggest_int('num_leaves', 31, 127),  # 255는 너무 큼 -> 127로 축소
            'max_depth': trial.suggest_int('max_depth', 7, 15),
            'min_child_samples': trial.suggest_int('min_child_samples', 10, 100),  # 중요 파라미터 추가

            # 정규화
            'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 10.0, log=True),
            'lambda_l2': trial.suggest_float('lambda_l2', 1e-8, 10.0, log=True),

            # 샘플링
            'feature_fraction': trial.suggest_float('feature_fraction', 0.6, 0.95),
            'bagging_fraction': trial.suggest_float('bagging_fraction', 0.6, 0.95),
            'bagging_freq': trial.suggest_int('bagging_freq', 1, 5),

            # 범주형 변수 처리 방식 (기본값 활용 권장)
            # 'cat_smooth': trial.suggest_float('cat_smooth', 1.0, 50.0)
        }

        # 5-Fold Cross Validation
        gkf = GroupKFold(n_splits=5)
        fold_scores = []
        models_x = []  # [추가] 각 fold의 모델 저장
        models_y = []

        for fold, (train_idx, val_idx) in enumerate(gkf.split(self.X_train, groups=self.game_ids)):
            X_tr, X_val = self.X_train.iloc[train_idx], self.X_train.iloc[val_idx]
            y_tr_x, y_val_x = self.y_train_x[train_idx], self.y_train_x[val_idx]
            y_tr_y, y_val_y = self.y_train_y[train_idx], self.y_train_y[val_idx]

            # 🚨 [중요] categorical_feature 명시
            dtrain_x = lgb.Dataset(X_tr, label=y_tr_x, categorical_feature=self.cat_features)
            dvalid_x = lgb.Dataset(X_val, label=y_val_x, reference=dtrain_x, categorical_feature=self.cat_features)

            model_x = lgb.train(
                params,
                dtrain_x,
                num_boost_round=3000,
                valid_sets=[dvalid_x],
                callbacks=[
                    lgb.early_stopping(stopping_rounds=50, verbose=False),  # 빠른 탐색 위해 50으로 단축
                    lgb.log_evaluation(0)
                ]
            )
            models_x.append(model_x)  # [추가] 모델 저장

            dtrain_y = lgb.Dataset(X_tr, label=y_tr_y, categorical_feature=self.cat_features)
            dvalid_y = lgb.Dataset(X_val, label=y_val_y, reference=dtrain_y, categorical_feature=self.cat_features)

            model_y = lgb.train(
                params,
                dtrain_y,
                num_boost_round=3000,
                valid_sets=[dvalid_y],
                callbacks=[
                    lgb.early_stopping(stopping_rounds=50, verbose=False),
                    lgb.log_evaluation(0)
                ]
            )
            models_y.append(model_y)  # [추가] 모델 저장

            # 검증
            pred_x = model_x.predict(X_val, num_iteration=model_x.best_iteration)
            pred_y = model_y.predict(X_val, num_iteration=model_y.best_iteration)

            # 좌표 클리핑 (경기장 밖 예측 방지)
            pred_x = np.clip(pred_x, 0, 105)
            pred_y = np.clip(pred_y, 0, 68)

            y_pred = np.column_stack([pred_x, pred_y])
            y_val = np.column_stack([y_val_x, y_val_y])

            eucl_dist = euclidean_distance(y_val, y_pred)
            fold_scores.append(eucl_dist)

        mean_score = np.mean(fold_scores)

        # [추가] 최적 모델 발견 시 즉시 저장
        if mean_score < self.best_score:
            self.best_score = mean_score
            self.best_models_x = models_x
            self.best_models_y = models_y
            self.best_params = params
            self.best_fold_scores = fold_scores

            print(f"\n🎯 New Best Score: {mean_score:.4f}m")

            # [추가] 즉시 파일로 저장 (Ctrl+C 대비)
            try:
                with open('best_model_v5_optuna_checkpoint.pkl', 'wb') as f:
                    pickle.dump({
                        'models_x': models_x,
                        'models_y': models_y,
                        'params': params,
                        'score': mean_score,
                        'fold_scores': fold_scores
                    }, f)
                print(f"   💾 모델 저장 완료: best_model_v5_optuna_checkpoint.pkl")
            except Exception as e:
                print(f"   ⚠️  모델 저장 실패: {e}")

        return mean_score


def main():
    print("=" * 80)
    print("  LightGBM V5 - Optuna 하이퍼파라미터 최적화")
    print("  목표: 0.2-0.5점 추가 개선")
    print("=" * 80)
    print()

    # 1. 데이터 로딩
    print("📊 데이터 로딩...")
    # 🚨 [주의] 반드시 V5.1 (NaN이 포함된 데이터)를 로드해야 함
    data = pd.read_csv('processed_train_data_v5.csv')
    print(f"데이터: {data.shape}\n")

    # 2. 피처/타겟 분리
    print("📊 피처/타겟 분리 및 전처리 수정...")

    y_train_x = data['target_x'].values
    y_train_y = data['target_y'].values
    game_ids = data['game_id'].values

    drop_cols = ['game_episode', 'game_id', 'target_x', 'target_y', 'final_team_id']
    X_train = data.drop(columns=[c for c in drop_cols if c in data.columns])

    # 🚨 [CRITICAL FIX] fillna(0) 삭제!! 🚨
    # X_train = X_train.fillna(0)  <-- 절대 금지 (LightGBM이 NaN을 스스로 처리하게 둠)

    # 🚨 [CRITICAL FIX] object -> numeric 변환 시에도 fillna(0) 제거
    # 대신 범주형 변수를 찾아서 리스트로 만듦
    cat_keywords = ['type_id', 'res_id', 'team_id_enc', 'is_home', 'period_id', 'is_last']
    cat_features = [c for c in X_train.columns if any(k in c for k in cat_keywords)]

    print(f"📌 범주형 변수 {len(cat_features)}개 감지됨 -> category 타입 변환")

    # 2. [핵심 수정] 범주형 변수들을 'category' 타입으로 강제 변환
    # LightGBM은 object 타입을 싫어하지만, category 타입은 아주 좋아합니다.
    for col in cat_features:
        X_train[col] = X_train[col].astype('category')

    # 3. 나머지 수치형 변수 중 object로 잡힌 것들 숫자 변환
    for col in X_train.columns:
        if col not in cat_features and X_train[col].dtype == 'object':
            X_train[col] = pd.to_numeric(X_train[col], errors='coerce')

    print(f"피처 수: {X_train.shape[1]}")
    print(f"샘플 수: {len(X_train):,}\n")

    # 3. Optuna 최적화
    n_trials = 50  # 초기 테스트로 50회 추천

    # [추가] Optuna DB 파일 경로 (중단 후 재개 가능)
    study_name = 'lightgbm_v5_optimization'
    storage_name = f'sqlite:///optuna_v5_study.db'

    print("🔧 Optuna 하이퍼파라미터 최적화 시작...")

    # [추가] 기존 study가 있는지 확인
    if os.path.exists('optuna_v5_study.db'):
        print(f"📂 기존 study 발견! 중단된 지점부터 재개합니다.")
        try:
            study = optuna.load_study(
                study_name=study_name,
                storage=storage_name,
                sampler=TPESampler(seed=42)
            )
            print(f"   이미 완료된 trial: {len(study.trials)}개")
            print(f"   현재 최고 점수: {study.best_value:.4f}m")
        except:
            print(f"   study 로딩 실패, 새로 시작합니다.")
            study = optuna.create_study(
                direction='minimize',
                sampler=TPESampler(seed=42),
                study_name=study_name,
                storage=storage_name,
                load_if_exists=True
            )
    else:
        print(f"📁 새로운 study 생성")
        study = optuna.create_study(
            direction='minimize',
            sampler=TPESampler(seed=42),
            study_name=study_name,
            storage=storage_name,
            load_if_exists=True
        )

    print(f"💾 진행상황 DB 저장: optuna_v5_study.db")
    print(f"   (Ctrl+C로 중단해도 재실행 시 이어서 진행됩니다)\n")

    # Optimizer 생성 시 cat_features 전달
    optimizer = LightGBMOptimizer(X_train, y_train_x, y_train_y, game_ids, cat_features)

    # [추가] KeyboardInterrupt 처리
    try:
        study.optimize(
            optimizer.objective,
            n_trials=n_trials,
            timeout=None,  # 시간 제한 없음
            show_progress_bar=True
        )
    except KeyboardInterrupt:
        print("\n\n⚠️  사용자 중단 (Ctrl+C)")
        print(f"💾 현재까지 진행: {len(study.trials)}개 trial 완료")
        print(f"🏆 현재 최고 점수: {study.best_value:.4f}m")
        print(f"\n재실행 시 자동으로 이어서 진행됩니다.")
        print(f"완료된 결과는 'best_model_v5_optuna_checkpoint.pkl'에 저장되어 있습니다.\n")

    # 4. 최적 파라미터 출력
    print("\n" + "=" * 80)
    print("  최적화 완료!")
    print("=" * 80)

    best_params = study.best_params
    best_score = study.best_value

    print(f"\n🏆 Best Score: {best_score:.4f}m")
    print(f"\n📊 Best Parameters:")
    for key, value in best_params.items():
        print(f"   {key:20s}: {value}")

    # 5. 개선도 분석
    baseline_score = 14.01  # V4.1 baseline
    improvement = baseline_score - best_score
    improvement_pct = (improvement / baseline_score) * 100

    print(f"\n📈 개선도 분석:")
    print(f"   Baseline (V4.1):     {baseline_score:.4f}m")
    print(f"   Optimized (V5):  {best_score:.4f}m")
    print(f"   Improvement:       {improvement:.4f}m ({improvement_pct:.2f}%)")

    if improvement > 0.2:
        print("\n🎉 우수한 개선! 즉시 V5로 재학습 권장")
    elif improvement > 0.1:
        print("\n✅ 의미 있는 개선! V5 재학습 고려")
    else:
        print("\n📊 소폭 개선. 다른 전략 병행 권장")

    # 6. 최적 파라미터 및 모델 저장
    best_params_full = {
        'objective': 'regression',
        'metric': 'rmse',
        'verbosity': -1,
        **best_params
    }

    # [추가] 최적 모델 전체 저장 (파라미터 + 모델 객체)
    final_save_path = f'best_model_v5_optuna_final.pkl'
    with open(final_save_path, 'wb') as f:
        pickle.dump({
            'models_x': optimizer.best_models_x,
            'models_y': optimizer.best_models_y,
            'params': best_params_full,
            'score': best_score,
            'fold_scores': optimizer.best_fold_scores,
            'improvement': improvement,
            'study': study
        }, f)

    print(f"\n💾 최적 모델 최종 저장: {final_save_path}")

    # 파라미터만 따로 저장
    with open('best_params_v5_optuna.pkl', 'wb') as f:
        pickle.dump({
            'params': best_params_full,
            'score': best_score,
            'improvement': improvement,
            'study': study
        }, f)

    print(f"💾 최적 파라미터 저장: best_params_v5_optuna.pkl")

    # 7. Optuna 시각화 정보
    print("\n" + "=" * 80)
    print("  Optuna 분석")
    print("=" * 80)

    print(f"\n총 시도 횟수: {len(study.trials)}")
    print(f"완료된 시도: {len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])}")
    print(f"실패한 시도: {len([t for t in study.trials if t.state == optuna.trial.TrialState.FAIL])}")

    # 중요 파라미터 분석
    print("\n📊 파라미터 중요도 (추정):")
    try:
        importances = optuna.importance.get_param_importances(study)
        for i, (param, importance) in enumerate(sorted(importances.items(), key=lambda x: x[1], reverse=True)[:10], 1):
            print(f"   {i:2d}. {param:20s}: {importance:.4f}")
    except:
        print("   파라미터 중요도 계산 실패 (trials 부족)")

    # 8. 다음 단계 안내
    print("\n" + "=" * 80)
    print("  다음 단계")
    print("=" * 80)

    print("\n1. 최적 파라미터로 V4.1 재학습:")
    print("   python train_lightgbm_v5_with_best_params.py")

    print("\n2. Test 추론 및 제출:")
    print("   python inference_v5.py")

    print("\n3. 다른 최적화 전략:")
    print("   - K 값 최적화 (K=15,25,30)")
    print("   - Feature Selection")
    print("   - XGBoost/CatBoost 실험")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()

