# 딥러닝 고급 전략: 전문가 분석 및 해결 방안

## 🔍 현재 상황 심층 분석

### 실험 결과
```
LightGBM V4:    1.50m  ✅
LSTM Original:  15.98m ❌
LSTM Fixed:     14.78m ❌ (1.2m 개선, 여전히 10배 차이)
```

### 예측값 분석
```python
# LSTM 예측 통계
end_x: 평균 65.38m, 표준편차 18.73
end_y: 평균 35.46m, 표준편차 19.99

# 문제점
→ 거의 비슷한 값(중앙값)만 예측
→ 실제 패턴 학습 실패
→ "안전한 평균값으로 회귀" 현상
```

---

## 🚨 근본 원인 (Root Cause)

### 1. **Wide Format의 치명적 한계** ⭐⭐⭐⭐⭐
```python
# 현재 데이터 구조
[start_x_0, start_x_1, ..., start_x_19,  # 20개 컬럼
 start_y_0, start_y_1, ..., start_y_19,  # 20개 컬럼
 ...]

# 문제
→ LSTM은 (batch, seq_len, features)를 원함
→ 시간 축이 feature 축에 펼쳐져 있음
→ LSTM의 recurrent 특성을 전혀 활용 못함!
```

**비유:**
- LightGBM: 평면 지도를 보고 길 찾기 (가능)
- LSTM: 평면 지도를 "시간 순서"로 학습하려 함 (불가능)

### 2. **데이터 부족 (Sample Efficiency)**
```
Train Samples: ~12,000
LSTM Parameters: 300,000+
→ 심각한 Underfitting
```

LightGBM은 15,000 샘플로 충분하지만, 딥러닝은 최소 100k+ 필요!

### 3. **정보 손실**
- NaN → 0: 패딩과 실제 0 구분 불가
- 정규화: 상대적 관계 손실
- Wide format: 시간적 의존성 완전 손실

---

## 💡 고급 해결 전략

### 🥇 전략 A: Long Format 완전 재설계 ⭐⭐⭐⭐⭐

**핵심:** Wide → True Sequence

#### Before (Wide Format)
```python
# Episode 1: [x0, x1, x2, ..., x19, y0, y1, ...]
# → 1개 샘플, 600+ features
```

#### After (Long Format)
```python
# Episode 1: 
# [(x0, y0, type0, ...),    # 시점 0
#  (x1, y1, type1, ...),    # 시점 1
#  ...
#  (x19, y19, type19, ...)] # 시점 19
# → (seq_len=20, features=15)
```

**파일:** `preprocessing_long_format.py` (이미 생성됨)

**예상 개선:**
- 14.78m → **3~5m** (3~5배 개선)
- LSTM이 진짜 시퀀스를 학습!

#### 실행 방법:
```bash
# 1. Long format 전처리
python preprocessing_long_format.py

# 2. Long format 학습 (별도 작성 필요)
python train_lstm_long_format.py
```

---

### 🥈 전략 B: Transformer + Attention ⭐⭐⭐⭐⭐

**핵심:** Self-Attention으로 중요한 이벤트 자동 선택

#### Why Transformer?
1. **Parallel Processing**: LSTM보다 빠름
2. **Long-range Dependencies**: 멀리 떨어진 이벤트도 연결
3. **Attention Weights**: 어떤 이벤트가 중요한지 시각화 가능

**파일:** `model_transformer.py` (이미 생성됨)

**주요 특징:**
```python
class TransformerPassPredictor:
    - Positional Encoding (시간 정보 명시)
    - Multi-Head Attention (다양한 패턴)
    - Attention Pooling (중요 이벤트 가중치)
```

**예상 개선:**
- 14.78m → **2~4m** (3~7배 개선)
- LightGBM과 경쟁 가능!

---

### 🥉 전략 C: Knowledge Distillation ⭐⭐⭐⭐

**핵심:** LightGBM(Teacher)의 지식을 LSTM(Student)에 전달

#### 아이디어
```python
# Step 1: Teacher(LightGBM) 예측
lgbm_pred = lightgbm_model.predict(X)  # 1.5m 성능

# Step 2: Student(LSTM) 학습
loss = alpha * MSE(lstm_pred, true_target) + \
       (1-alpha) * MSE(lstm_pred, lgbm_pred)
       
# alpha=0.7: 30%는 LightGBM에서 배움
```

**장점:**
- LightGBM의 좋은 예측을 "힌트"로 활용
- Soft targets로 학습 안정화

**예상 개선:**
- 14.78m → **5~8m** (2~3배 개선)

**구현:**
```python
class DistillationLoss(nn.Module):
    def __init__(self, alpha=0.7, temperature=2.0):
        super().__init__()
        self.alpha = alpha
        self.temperature = temperature
    
    def forward(self, student_pred, true_target, teacher_pred):
        # Hard target loss
        hard_loss = F.mse_loss(student_pred, true_target)
        
        # Soft target loss (with temperature)
        soft_loss = F.mse_loss(
            student_pred / self.temperature,
            teacher_pred / self.temperature
        )
        
        return self.alpha * hard_loss + (1 - self.alpha) * soft_loss
```

---

### 🏆 전략 D: 앙상블의 최종 진화 ⭐⭐⭐⭐⭐

**핵심:** LightGBM + (Long LSTM) + (Transformer)

#### Level 1: 단순 평균
```python
final = 0.7 * lgbm + 0.2 * lstm + 0.1 * transformer
```

#### Level 2: Stacking (Meta-Learner)
```python
# Base models
lgbm_pred = lgbm.predict(X)
lstm_pred = lstm.predict(X)
trans_pred = transformer.predict(X)

# Meta features
meta_X = np.column_stack([lgbm_pred, lstm_pred, trans_pred])

# Meta model (간단한 Ridge)
meta_model = Ridge(alpha=1.0)
meta_model.fit(meta_X, y_true)

# Final prediction
final = meta_model.predict(meta_X)
```

**예상 최종 성능:**
- **1.2~1.4m** (LightGBM 1.5m보다 좋음!)
- Test 점수: **12~13점대**

---

## 📊 전략별 비교표

| 전략 | 구현 난이도 | 시간 | 예상 개선 | 성공 확률 | 우선순위 |
|-----|-----------|------|----------|----------|---------|
| **A. Long Format** | 중 | 2~3시간 | 14.78m → 3~5m | 80% | 🥇 1순위 |
| **B. Transformer** | 중상 | 3~4시간 | 14.78m → 2~4m | 70% | 🥈 2순위 |
| **C. Distillation** | 하 | 1~2시간 | 14.78m → 5~8m | 60% | 🥉 3순위 |
| **D. Stacking** | 하 | 30분 | 1.5m → 1.2m | 90% | 🏆 최종 |

---

## 🚀 실행 로드맵 (권장)

### Phase 1: Long Format (최우선) ⏱️ 3시간
```bash
# Step 1: 전처리
python preprocessing_long_format.py

# Step 2: 학습 (작성 필요)
# → train_lstm_long_format.py 작성
python train_lstm_long_format.py

# 예상: 3~5m 달성
```

### Phase 2: Transformer (병행) ⏱️ 4시간
```bash
# model_transformer.py 기반 학습
python train_transformer.py

# 예상: 2~4m 달성
```

### Phase 3: Stacking (최종) ⏱️ 30분
```bash
# LightGBM + Long LSTM + Transformer 앙상블
python create_stacking_ensemble.py

# 예상: 1.2~1.4m 달성
```

---

## 💻 즉시 실행 가능한 코드

### Quick Win 1: Knowledge Distillation (가장 빠름)

```python
# train_lstm_v4_distillation.py

# 1. LightGBM 예측 로딩
with open('lightgbm_model_v4_5fold.pkl', 'rb') as f:
    lgbm_models = pickle.load(f)

# 2. LightGBM 예측 생성
lgbm_pred_x = np.mean([m.predict(X_val) for m in lgbm_models['models_x']], axis=0)
lgbm_pred_y = np.mean([m.predict(X_val) for m in lgbm_models['models_y']], axis=0)
lgbm_pred = np.column_stack([lgbm_pred_x, lgbm_pred_y])

# 3. Distillation Loss
class DistillationLoss(nn.Module):
    def __init__(self, alpha=0.7):
        super().__init__()
        self.alpha = alpha
        self.mse = nn.MSELoss()
    
    def forward(self, student_pred, true_target, teacher_pred):
        hard_loss = self.mse(student_pred, true_target)
        soft_loss = self.mse(student_pred, teacher_pred)
        return self.alpha * hard_loss + (1 - self.alpha) * soft_loss

# 4. 학습
criterion = DistillationLoss(alpha=0.7)
for epoch in range(epochs):
    output = model(X)
    loss = criterion(output, y_true, torch.from_numpy(lgbm_pred).to(device))
    ...
```

### Quick Win 2: Stacking Ensemble

```python
# create_stacking_simple.py

import numpy as np
from sklearn.linear_model import Ridge

# 1. Base model 예측 로딩
lgbm_val_pred = np.load('lgbm_val_pred.npy')  # (N, 2)
lstm_val_pred = np.load('lstm_val_pred.npy')  # (N, 2)

# 2. Meta features
meta_X = np.column_stack([
    lgbm_val_pred.flatten(),  # (N*2,)
    lstm_val_pred.flatten()
])  # (N, 4)

# 3. Meta model
meta_model = Ridge(alpha=1.0)
meta_model.fit(meta_X, y_val.flatten())

# 4. Test 예측
lgbm_test_pred = ...
lstm_test_pred = ...
meta_X_test = np.column_stack([lgbm_test_pred.flatten(), lstm_test_pred.flatten()])
final_pred = meta_model.predict(meta_X_test).reshape(-1, 2)

# 예상: 1.2~1.4m
```

---

## 🎯 최종 권장사항

### 현실적인 접근 (시간 대비 효과)

1. **즉시 (30분):** Stacking Ensemble
   - LightGBM + LSTM Fixed
   - 예상: 1.4m → Test 13점대

2. **단기 (3시간):** Long Format
   - 진짜 시퀀스 학습
   - 예상: 3~5m → Stacking 시 1.2m

3. **중기 (1주):** Transformer
   - SOTA 아키텍처
   - 예상: 2~4m → 최종 1.0~1.2m

### 이상적인 접근 (최고 성능)

```
LightGBM (1.5m) + Long LSTM (3m) + Transformer (2m)
→ Stacking → 1.0~1.2m
→ Test: 11~12점대 (Top 10%)
```

---

## 🔬 추가 실험 아이디어

### 1. Graph Neural Network (GNN)
- 패스를 Graph로 모델링
- 선수 간 관계를 Edge로 표현
- 예상: 2~3m

### 2. Temporal Convolutional Network (TCN)
- 1D Conv로 시퀀스 학습
- LSTM보다 빠르고 효과적
- 예상: 3~5m

### 3. Multi-Task Learning
```python
# 동시에 여러 task 학습
outputs = model(X)
end_x_pred = outputs[:, 0]
end_y_pred = outputs[:, 1]
pass_success_pred = outputs[:, 2]  # 추가 task

loss = mse_loss(end_pred, target) + bce_loss(success_pred, success_label)
```

---

## 📝 결론

### 현재 LSTM이 안 되는 이유
1. **Wide Format**: LSTM의 장점을 전혀 활용 못함 (치명적!)
2. **데이터 부족**: 12k 샘플은 딥러닝에 턱없이 부족
3. **평균값 회귀**: 모델이 "안전한 중앙값"만 학습

### 해결 방법
1. **Long Format**: Wide 버리고 진짜 시퀀스 사용 (필수!)
2. **Transformer**: Attention으로 중요 이벤트 선택
3. **Distillation**: LightGBM 지식 활용
4. **Stacking**: 최종 앙상블

### 현실적 목표
- **Short-term**: Stacking으로 1.4m → Test 13점대
- **Mid-term**: Long Format으로 3m → Stacking 1.2m → Test 12점대
- **Long-term**: Transformer 추가 → 1.0m → Test 11점대

### 최우선 Action
```bash
# 1. Long Format 전처리 (가장 중요!)
python preprocessing_long_format.py

# 2. Stacking Ensemble (가장 빠름)
python create_stacking_simple.py
```

---

**작성일**: 2025-12-18  
**전문가 판단**: Wide Format 버리고 Long Format 재설계 필수!  
**예상 최종 성능**: 1.0~1.2m (LightGBM보다 좋음)

