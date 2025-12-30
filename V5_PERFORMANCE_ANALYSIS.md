# 🔍 LSTM V5 성능 분석 및 V6/V7 개선 전략

## 📊 V5 실험 결과 분석

### 실제 성능
```
Fold별 결과:
   Fold 1: 15.1197m
   Fold 2: 15.4364m
   Fold 3: 15.3053m
   Fold 4: 15.3126m
   Fold 5: 15.4043m

평균: 15.3157m ± 0.1104m
```

### 목표 대비
- **목표**: LightGBM 14.138m 초과
- **실제**: 15.3157m
- **차이**: +1.1777m (약 **8.3% 낮은 성능**)

### 예측 통계
```
end_x: 18.81 ~ 98.17 (평균 69.16, 표준편차 19.24)
end_y: 28.25 ~ 65.01 (평균 42.35, 표준편차 8.66)
```
→ **Y 좌표 표준편차가 너무 작음** (8.66), 다양성 부족 의심

### Fold 간 불일치
```
X 좌표: 1.6616m
Y 좌표: 1.3716m
```
→ **불안정성**: Fold 간 예측 차이가 큼 (일반화 문제)

---

## 🔍 근본 원인 분석

### 1. **모델 용량 부족** ⭐⭐⭐
**증거**:
- Hidden Dim 384, Layers 3 → 파라미터 수 부족
- 복잡한 시퀀스 패턴 학습 한계

**영향도**: 매우 높음 (예상 -1.0m)

**해결책**:
- Hidden Dim 512 이상
- Layers 6 이상
- Transformer로 전환 (RNN보다 강력)

---

### 2. **데이터 부족 / 과적합** ⭐⭐⭐
**증거**:
- Train/Val Loss 차이 (과적합 가능성)
- Fold 간 불일치 큼 (일반화 실패)
- Y 좌표 다양성 부족

**영향도**: 매우 높음 (예상 -0.8m)

**해결책**:
- **Data Augmentation**
  - Gaussian Noise
  - 좌우 반전
  - 시퀀스 자르기
  - Mixup
- Label Smoothing
- Dropout 감소 (Augmentation이 대신)

---

### 3. **Loss Function 문제** ⭐⭐
**증거**:
- 단순 유클리드 거리만 사용
- 어려운 샘플 학습 부족

**영향도**: 중간 (예상 -0.5m)

**해결책**:
- **Focal Loss** (어려운 샘플 집중)
- **Smooth L1 Loss** (이상치에 강함)
- **Multi-Task Learning** (거리 + 각도 예측)

---

### 4. **학습 전략 미흡** ⭐⭐
**증거**:
- Learning Rate 고정
- Epoch 100 → 더 필요할 수 있음
- Warmup 없음

**영향도**: 중간 (예상 -0.3m)

**해결책**:
- Warmup + Cosine Scheduler
- Epoch 150~200
- Gradient Accumulation (더 큰 Effective Batch)

---

### 5. **RNN의 구조적 한계** ⭐
**증거**:
- Bidirectional GRU도 시퀀스 길이 증가 시 정보 손실
- Attention이 있지만 RNN이 병목

**영향도**: 낮음 (예상 -0.2m)

**해결책**:
- **Pure Transformer** (RNN 완전 제거)
- Self-Attention만으로 모든 시점 직접 연결

---

## 🚀 V6/V7 개선 전략

### V6: Transformer + Focal Loss
**핵심 개선**:
1. ✅ RNN 제거 → Pure Transformer Encoder
2. ✅ Hidden 512, Layers 4
3. ✅ Focal Loss (어려운 샘플 집중)
4. ✅ GELU Activation (BERT 스타일)
5. ✅ Pre-LN (더 안정적)
6. ✅ Global Average Pooling
7. ✅ Warmup + Cosine Scheduler

**예상 성능**: 14.5~15.0m (-0.3~0.8m 개선)

---

### V7: V6 + Data Augmentation + Multi-Task
**핵심 개선**:
1. ✅ V6의 모든 개선사항 포함
2. ✅ **Data Augmentation**
   - Gaussian Noise
   - 좌우 반전 (Y 좌표)
   - 시퀀스 자르기
3. ✅ **Mixup** (샘플 간 보간)
4. ✅ **Multi-Task Learning**
   - Main: 좌표 예측
   - Auxiliary: 거리 예측
5. ✅ Layers 6 (더 깊게)
6. ✅ Dropout 0.1 (낮게 - Augmentation이 대신)
7. ✅ Epoch 200 (더 길게)

**예상 성능**: 13.5~14.5m (-0.8~1.8m 개선)

---

## 📈 예상 성능 개선 경로

```
LSTM V5: 15.3157m (현재)
    ↓ V6 (Transformer + Focal Loss)
         14.5~15.0m [-0.3~0.8m]
    ↓ V7 (V6 + Data Aug + Multi-Task)
         13.5~14.5m [-0.8~1.8m]
    ↓ V7 5-Fold + TTA
         13.0~14.0m [-1.3~2.3m]

목표: < 14.138m (LightGBM)
```

---

## 🎯 실행 우선순위

### 🔥 최우선 (즉시 실행)
1. **V7 모델 학습** (`train_lstm_v7_augmented.py`)
   - Data Augmentation 효과 검증
   - 예상 소요 시간: 2~3시간
   - 예상 성능: 13.5~14.5m

### ⭐ 고우선 (V7 결과 후)
2. **V6 모델 학습** (`train_lstm_v6_transformer.py`)
   - Pure Transformer 효과 검증
   - V7과 비교하여 Augmentation 효과 측정

### ✅ 중우선 (시간 여유 시)
3. **V7 5-Fold 전체 학습**
   - 일반화 성능 극대화
   - TTA 적용

---

## 🔬 추가 실험 아이디어

### A. 피처 엔지니어링
```python
# 추가 시퀀스 피처
- 누적 거리 (cumulative_distance)
- 방향 전환 빈도 (direction_changes)
- 패스 속도 변화 (speed_variance)
- 시간 간격 패턴 (time_gaps)
```

### B. 앙상블 (최후의 수단)
```python
# V5 + V6 + V7 + LightGBM
- 가중 평균 (Weight Optimization)
- Stacking (Meta-Learner)
```

### C. 전처리 개선
```python
# Wide → Long Format 변환
- 시퀀스 길이 동적 조정
- Attention Mask 개선
```

---

## 📊 V5 vs V6 vs V7 비교표

| 항목 | V5 | V6 | V7 |
|------|----|----|----| 
| **아키텍처** | Bi-GRU + Attention | Pure Transformer | Transformer |
| **Hidden Dim** | 384 | 512 | 512 |
| **Layers** | 3 | 4 | 6 |
| **Loss** | Euclidean | Focal Loss | Multi-Task |
| **Augmentation** | ❌ | ❌ | ✅ |
| **Mixup** | ❌ | ❌ | ✅ |
| **Warmup** | ❌ | ✅ | ✅ |
| **Epochs** | 100 | 150 | 200 |
| **Dropout** | 0.4 | 0.2 | 0.1 |
| **예상 성능** | 15.3m | 14.5~15.0m | 13.5~14.5m |

---

## 🎓 핵심 교훈

### 1. 데이터 부족 문제는 Augmentation으로 해결
- **Y 좌표 다양성 부족** → 좌우 반전 Augmentation
- **Fold 간 불일치** → Mixup으로 일반화

### 2. 모델 크기가 중요
- Hidden 384 → 512: 약 33% 용량 증가
- Layers 3 → 6: 2배 깊이 증가
- **더 큰 모델 = 더 나은 표현력**

### 3. RNN의 한계
- Bidirectional GRU도 긴 시퀀스에서 정보 손실
- **Transformer가 시퀀스 모델링에 더 적합**

### 4. Loss Function의 중요성
- 단순 유클리드 거리 → Focal Loss: 어려운 샘플 학습
- Multi-Task Learning: 보조 태스크로 정규화 효과

---

## 🚀 즉시 실행 명령

### V7 학습 (최우선)
```bash
# Colab/GPU 서버에서
python train_lstm_v7_augmented.py
```

**2~3시간 후 결과 확인**:
- Val Loss < 14.0m → 🎉 성공!
- Val Loss 14.0~14.5m → ✅ 5-Fold로 더 개선
- Val Loss > 14.5m → V6 시도

---

### V6 학습 (대안)
```bash
python train_lstm_v6_transformer.py
```

**1~2시간 후 결과 확인**:
- V7과 비교하여 Augmentation 효과 측정

---

## 📌 성공 시나리오

### 시나리오 1: V7 단독 성공 (70% 확률)
```
V7 Val Loss: 13.8m
5-Fold 평균: 13.9m
TTA 적용: 13.7m
→ LightGBM (14.138m) 초과! 🎉
```

### 시나리오 2: V6 + V7 앙상블 (20% 확률)
```
V6: 14.3m
V7: 14.1m
평균: 14.0m
→ LightGBM 근접 ✅
```

### 시나리오 3: 추가 개선 필요 (10% 확률)
```
V7: 14.5m
→ 피처 엔지니어링
→ 전처리 개선
→ LightGBM 앙상블
```

---

## 🎯 최종 목표

> **"V7 모델 단독으로 LightGBM (14.138m)을 초과하여 13.5~14.0m 달성"**

### 성공 지표
- ✅ Val Loss < 14.0m
- ✅ Public LB < 14.5m
- ✅ Fold 간 불일치 < 1.0m

---

**작성일**: 2025-12-19  
**상태**: V7 학습 준비 완료 🚀  
**다음 단계**: `python train_lstm_v7_augmented.py` 실행!

