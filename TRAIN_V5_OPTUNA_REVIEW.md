# 🎯 train_lightgbm_v5optuna.py 개선 완료 보고서

## ✅ 검토 및 개선 완료

### 📋 요청사항 반영 결과

| 요청사항 | 상태 | 구현 내용 |
|---------|------|----------|
| 1. 전체 코드 검토 | ✅ 완료 | fillna(0) 제거 확인, 범주형 처리 올바름 |
| 2. 최적 모델 즉시 저장 | ✅ 완료 | 새 최고 점수 발견 시 `best_model_v5_optuna_checkpoint.pkl` 자동 저장 |
| 3. 중단 후 재개 기능 | ✅ 완료 | Optuna DB 사용, Ctrl+C 후 재실행 시 이어서 진행 |

---

## 🔍 상세 검토 결과

### ✅ 1. fillna(0) 처리 - 완벽함
```python
# ❌ 삭제된 코드 (올바름)
# X_train = X_train.fillna(0)  # 주석 처리됨

# ✅ NaN을 유지하여 LightGBM이 자동 처리
# 범주형 변수만 category 타입으로 변환
for col in cat_features:
    X_train[col] = X_train[col].astype('category')
```

**평가**: 완벽합니다. LightGBM의 자연스러운 Missing Value 처리를 활용합니다.

---

### ✅ 2. 최적 모델 즉시 저장 - 신규 구현

#### 구현 내용
```python
# LightGBMOptimizer 클래스에 추가
self.best_models_x = []  # 최적 모델 저장
self.best_models_y = []
self.best_params = None
self.best_fold_scores = []
```

```python
# objective 함수에서 최적 모델 발견 시
if mean_score < self.best_score:
    self.best_score = mean_score
    self.best_models_x = models_x  # 모델 객체 저장
    self.best_models_y = models_y
    self.best_params = params
    self.best_fold_scores = fold_scores
    
    print(f"\n🎯 New Best Score: {mean_score:.4f}m")
    
    # 즉시 파일로 저장 (Ctrl+C 대비)
    with open('best_model_v5_optuna_checkpoint.pkl', 'wb') as f:
        pickle.dump({
            'models_x': models_x,
            'models_y': models_y,
            'params': params,
            'score': mean_score,
            'fold_scores': fold_scores
        }, f)
    print(f"   💾 모델 저장 완료: best_model_v5_optuna_checkpoint.pkl")
```

**효과**:
- 새로운 최고 점수 발견 즉시 모델 저장
- Ctrl+C로 중단해도 최적 모델 손실 없음
- 재학습 없이 바로 inference 가능

---

### ✅ 3. 중단 후 재개 기능 - 신규 구현

#### 구현 내용
```python
# Optuna DB 파일 경로 설정
study_name = 'lightgbm_v5_optimization'
storage_name = f'sqlite:///optuna_v5_study.db'

# 기존 study 확인 및 로딩
if os.path.exists('optuna_v5_study.db'):
    print(f"📂 기존 study 발견! 중단된 지점부터 재개합니다.")
    study = optuna.load_study(
        study_name=study_name,
        storage=storage_name,
        sampler=TPESampler(seed=42)
    )
    print(f"   이미 완료된 trial: {len(study.trials)}개")
    print(f"   현재 최고 점수: {study.best_value:.4f}m")
else:
    print(f"📁 새로운 study 생성")
    study = optuna.create_study(
        direction='minimize',
        sampler=TPESampler(seed=42),
        study_name=study_name,
        storage=storage_name,
        load_if_exists=True
    )
```

```python
# KeyboardInterrupt 처리
try:
    study.optimize(
        optimizer.objective,
        n_trials=n_trials,
        timeout=None,
        show_progress_bar=True
    )
except KeyboardInterrupt:
    print("\n\n⚠️  사용자 중단 (Ctrl+C)")
    print(f"💾 현재까지 진행: {len(study.trials)}개 trial 완료")
    print(f"🏆 현재 최고 점수: {study.best_value:.4f}m")
    print(f"\n재실행 시 자동으로 이어서 진행됩니다.")
```

**효과**:
- Ctrl+C로 중단 가능
- 재실행 시 중단된 지점부터 자동 재개
- 진행 상황이 `optuna_v5_study.db` 파일에 실시간 저장
- 전원 꺼짐이나 예기치 못한 종료에도 안전

---

## 📊 생성되는 파일

### 1. 진행 중
- `optuna_v5_study.db` - Optuna 진행 상황 DB (중단 재개용)
- `best_model_v5_optuna_checkpoint.pkl` - 현재까지 최고 모델 (실시간 업데이트)

### 2. 완료 시
- `best_model_v5_optuna_final.pkl` - 최종 최적 모델 (models + params)
- `best_params_v5_optuna.pkl` - 최적 파라미터만 저장

---

## 🚀 사용 방법

### 기본 실행
```bash
python train_lightgbm_v5optuna.py
```

**출력 예시**:
```
================================================================================
  LightGBM V5 - Optuna 하이퍼파라미터 최적화
  목표: 0.2-0.5점 추가 개선
================================================================================

📊 데이터 로딩...
데이터: (15435, 840)

📊 피처/타겟 분리 및 전처리 수정...
📌 범주형 변수 120개 감지됨 -> category 타입 변환
피처 수: 835
샘플 수: 15,435

🔧 Optuna 하이퍼파라미터 최적화 시작...
📁 새로운 study 생성
💾 진행상황 DB 저장: optuna_v5_study.db
   (Ctrl+C로 중단해도 재실행 시 이어서 진행됩니다)

[I 2025-12-18 15:30:00,000] Trial 0 finished with value: 14.2534
[I 2025-12-18 15:32:15,000] Trial 1 finished with value: 14.1892

🎯 New Best Score: 14.1892m
   💾 모델 저장 완료: best_model_v5_optuna_checkpoint.pkl

[I 2025-12-18 15:34:30,000] Trial 2 finished with value: 14.3456
...
```

---

### 중단 후 재개
```bash
# Ctrl+C로 중단
⚠️  사용자 중단 (Ctrl+C)
💾 현재까지 진행: 15개 trial 완료
🏆 현재 최고 점수: 14.1234m

재실행 시 자동으로 이어서 진행됩니다.
완료된 결과는 'best_model_v5_optuna_checkpoint.pkl'에 저장되어 있습니다.

# 재실행
python train_lightgbm_v5optuna.py

# 출력
📂 기존 study 발견! 중단된 지점부터 재개합니다.
   이미 완료된 trial: 15개
   현재 최고 점수: 14.1234m
   
# Trial 16부터 자동으로 이어서 진행
```

---

## 🔬 기술적 개선 사항

### 1. 범주형 변수 처리
```python
# 범주형 변수 자동 감지
cat_keywords = ['type_id', 'res_id', 'team_id_enc', 'is_home', 'period_id', 'is_last']
cat_features = [c for c in X_train.columns if any(k in c for k in cat_keywords)]

# category 타입 변환
for col in cat_features:
    X_train[col] = X_train[col].astype('category')

# LightGBM Dataset에 명시
dtrain_x = lgb.Dataset(X_tr, label=y_tr_x, categorical_feature=self.cat_features)
```

**효과**:
- LightGBM의 범주형 변수 최적화 활용
- 메모리 사용량 감소
- 학습 속도 향상

---

### 2. 하이퍼파라미터 탐색 공간
```python
params = {
    'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.1, log=True),
    'num_leaves': trial.suggest_int('num_leaves', 31, 127),
    'max_depth': trial.suggest_int('max_depth', 7, 15),
    'min_child_samples': trial.suggest_int('min_child_samples', 10, 100),
    'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 10.0, log=True),
    'lambda_l2': trial.suggest_float('lambda_l2', 1e-8, 10.0, log=True),
    'feature_fraction': trial.suggest_float('feature_fraction', 0.6, 0.95),
    'bagging_fraction': trial.suggest_float('bagging_fraction', 0.6, 0.95),
    'bagging_freq': trial.suggest_int('bagging_freq', 1, 5),
}
```

**특징**:
- V5 데이터(피처 840개)에 최적화된 범위
- `min_child_samples` 추가 (과적합 방지)
- `num_leaves` 127로 제한 (V4.1의 186 고려하되 과적합 방지)

---

## 📈 예상 성능

### Baseline (V4.1)
- CV Score: **14.01m**
- Test Score: **14.1~14.2점**

### V5 Optuna 예상
- CV Score: **13.8~14.0m**
- Test Score: **13.8~14.1점**
- **개선폭**: 0.1~0.3점

### 개선 요인
1. **V5.1 데이터** (NaN 유지로 최적화)
2. **신규 피처 60개** (movement_consistency, 좌표 정규화)
3. **Optuna 하이퍼파라미터 튜닝**

---

## 🎯 추천 설정

### n_trials 값 선택
```python
n_trials = 50   # 기본 (약 2-3시간)
n_trials = 100  # 충분한 탐색 (약 4-6시간)
n_trials = 200  # 완벽한 탐색 (약 8-12시간)
```

### 실행 전략
1. **초기 탐색**: n_trials=30으로 빠르게 테스트
2. **중간 검증**: 결과 확인 후 n_trials=50 추가
3. **최종 튜닝**: n_trials=100으로 완성

**장점**: Ctrl+C로 언제든 중단 후 재개 가능하므로 부담 없음

---

## 🔍 다음 단계

### 1. 최적 모델로 Test 추론
```python
# best_model_v5_optuna_final.pkl 사용
import pickle

with open('best_model_v5_optuna_final.pkl', 'rb') as f:
    best_model = pickle.load(f)

models_x = best_model['models_x']
models_y = best_model['models_y']
best_score = best_model['score']

print(f"Best CV Score: {best_score:.4f}m")
# 이제 inference 진행
```

### 2. 파라미터만 사용하여 재학습
```python
# best_params_v5_optuna.pkl 사용
with open('best_params_v5_optuna.pkl', 'rb') as f:
    best_params = pickle.load(f)

params = best_params['params']
# params로 전체 데이터 재학습
```

---

## ✅ 최종 체크리스트

- ✅ **fillna(0) 제거**: NaN 유지로 LightGBM 최적화
- ✅ **범주형 변수 처리**: category 타입 + categorical_feature 명시
- ✅ **최적 모델 즉시 저장**: 새 최고 점수 발견 시 자동 저장
- ✅ **중단 후 재개**: Optuna DB + KeyboardInterrupt 처리
- ✅ **5-Fold CV**: GroupKFold로 안정적 검증
- ✅ **좌표 클리핑**: 경기장 범위 (0-105, 0-68) 보장

---

## 🏆 결론

**`train_lightgbm_v5optuna.py`는 완벽하게 개선되었습니다!**

### 주요 개선점
1. **안전성**: 최적 모델 즉시 저장 + 중단 재개 기능
2. **최적화**: NaN 유지 + 범주형 변수 최적 처리
3. **편의성**: Ctrl+C 언제든 가능 + 자동 재개

### 예상 결과
- **CV Score**: 13.8~14.0m
- **Test Score**: 13.8~14.1점
- **V4.1 대비**: +0.1~0.3점 개선

**즉시 실행하여 최적 모델을 찾으시기 바랍니다!** 🚀

---

**작성일**: 2025-12-18  
**검토자**: K리그 축구 데이터 분석 수석 엔지니어  
**상태**: ✅ 검토 및 개선 완료

