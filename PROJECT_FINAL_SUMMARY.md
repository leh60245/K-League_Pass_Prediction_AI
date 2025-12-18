# 📊 K-League 패스 예측 프로젝트 - 최종 상태

## 🎯 프로젝트 목표
**순수 LSTM 모델만으로 LightGBM (14.138m) 성능 초과**

---

## 📈 성능 히스토리

| 버전 | 모델 | 주요 기법 | Val Loss | Public LB | 비고 |
|------|------|-----------|----------|-----------|------|
| V1-V3 | LightGBM | 기본 트리 모델 | ~1.5m | - | 초기 버전 |
| **V4** | LightGBM | 최적화된 피처 | **1.5m** | **14.138m** | 🏆 최고 성능 |
| V4 | LSTM (Baseline) | 기본 GRU | 14.7m | 15.649m | 딥러닝 첫 시도 |
| **V5** | LSTM + Attention | Multi-Head Attention, Bidirectional, Padding Mask | **TBD** | **TBD** | 🚀 현재 버전 |

---

## 🗂️ 프로젝트 구조

### ✅ 활성 파일 (V5 - LSTM 전용)

#### 📊 데이터
- `processed_train_data_v4.csv` - Wide Format 학습 데이터
- `processed_test_data_v4.csv` - Wide Format 테스트 데이터

#### 🧠 학습 스크립트
- `train_lstm_v5_attention.py` - **단일 Fold 프로토타이핑** (빠른 검증용)
- `train_lstm_v5_5fold.py` - **5-Fold 전체 학습** (최종 성능용)

#### 🔮 추론 스크립트
- `inference_lstm_v5_attention.py` - 단일 Fold 모델 추론
- `inference_lstm_v5_5fold.py` - **5-Fold 앙상블 + TTA 추론** (최종 제출용)

#### 💾 모델 파일 (생성 예정)
- `lstm_model_v5_attention_best.pth` - 단일 Fold 모델
- `lstm_model_v5_fold0.pth` ~ `lstm_model_v5_fold4.pth` - 5-Fold 모델

#### 📄 문서
- `LSTM_PERFORMANCE_IMPROVEMENT_STRATEGY.md` - 성능 향상 전략 상세 설명
- `LSTM_V5_EXECUTION_GUIDE.md` - 실행 가이드 및 문제 해결
- `PROJECT_FINAL_SUMMARY.md` - 이 문서

### ❌ 제거된 파일 (앙상블 관련)
- `ensemble_model.py`
- `ensemble_3models.py`
- `ensemble_v3_v4.py`
- `inference_ensemble.py`
- `create_ensemble.py`
- `inference_3models.py`
- `create_stacking_quickwin.py`
- `optimize_weights.py`
- `ensemble_model.pkl`
- `ensemble_3models.pkl`

### 🗄️ 레거시 파일 (참고용)
- `lightgbm_model_v4.1_5fold.pkl` - LightGBM 최고 성능 모델
- `preprocessing_v4.py` - V4 전처리 스크립트
- `train_lstm_v4.py` - LSTM Baseline (V4)
- 기타 이전 버전 파일들

---

## 🔬 LSTM V5 핵심 개선사항

### 1. **Multi-Head Attention** ⭐⭐⭐
```python
self.attention = nn.MultiheadAttention(
    embed_dim=rnn_output_dim,
    num_heads=8,
    dropout=dropout,
    batch_first=True
)
```
- **효과**: 시퀀스에서 중요한 시점에 집중
- **예상 개선폭**: 10~20%

### 2. **Padding Mask** ⭐⭐⭐
```python
def _create_padding_mask(self):
    mask = (self.numerical_tensor.sum(dim=-1) == 0)
    return mask
```
- **효과**: 실제 데이터와 패딩 구분하여 학습 품질 향상
- **예상 개선폭**: 5~10%

### 3. **Bidirectional RNN** ⭐⭐
```python
self.rnn = nn.GRU(
    hidden_dim, hidden_dim,
    bidirectional=True
)
```
- **효과**: 양방향 시퀀스 정보 활용
- **예상 개선폭**: 5~10%

### 4. **전체 피처 정규화** ⭐⭐
```python
# X 좌표: /105, Y 좌표: /68
# 속도: /30, 각도: /π, 시간: 동적 정규화
```
- **효과**: 학습 안정성 및 수렴 속도 향상
- **예상 개선폭**: 5~10%

### 5. **깊은 Output Head** ⭐
```python
self.fc = nn.Sequential(
    nn.Linear(rnn_output_dim, hidden_dim),
    nn.ReLU(),
    nn.LayerNorm(hidden_dim),
    nn.Dropout(dropout),
    nn.Linear(hidden_dim, hidden_dim // 2),
    nn.ReLU(),
    nn.LayerNorm(hidden_dim // 2),
    nn.Dropout(dropout),
    nn.Linear(hidden_dim // 2, 2)
)
```
- **효과**: 복잡한 패턴 학습 능력 향상
- **예상 개선폭**: 3~5%

### 6. **Residual Connection** ⭐
```python
attn_out = self.attention_norm(attn_out + rnn_out)
```
- **효과**: 그래디언트 흐름 개선
- **예상 개선폭**: 2~5%

### 7. **5-Fold CV + TTA**
- **5-Fold**: 일반화 성능 극대화
- **TTA** (Test Time Augmentation): 좌우 반전 예측 평균
- **예상 개선폭**: 5~10%

---

## 📊 하이퍼파라미터

### 최적화된 설정 (V5)
```python
K = 20                    # 시퀀스 길이
BATCH_SIZE = 64          # 배치 크기
HIDDEN_DIM = 384         # Hidden 차원
NUM_LAYERS = 3           # RNN 레이어 수
DROPOUT = 0.4            # Dropout 비율
LEARNING_RATE = 5e-4     # 학습률
NUM_EPOCHS = 100         # 최대 에포크
EARLY_STOPPING_PATIENCE = 20
USE_LSTM = False         # GRU 사용
BIDIRECTIONAL = True     # 양방향
NUM_HEADS = 8            # Attention Head 수
```

### Baseline (V4) 대비 변경
- `HIDDEN_DIM`: 256 → **384** (↑50%)
- `NUM_LAYERS`: 2 → **3** (↑1)
- `DROPOUT`: 0.3 → **0.4** (↑0.1)
- `LEARNING_RATE`: 1e-3 → **5e-4** (↓50%)
- **Bidirectional**: 추가
- **Attention**: 추가

---

## 🚀 실행 순서

### Phase 1: 빠른 검증 (30분~1시간)
```bash
python train_lstm_v5_attention.py
```
→ `lstm_model_v5_attention_best.pth` 생성

### Phase 2: 전체 학습 (3~5시간)
```bash
python train_lstm_v5_5fold.py
```
→ `lstm_model_v5_fold{0-4}.pth` 생성 (5개)

### Phase 3: 최종 추론 (10~20분)
```bash
python inference_lstm_v5_5fold.py
```
→ `submission_lstm_v5_5fold_tta.csv` 생성

---

## 🎯 예상 성능

### 시나리오 분석

#### 🎉 낙관적 (모든 개선사항이 효과적)
- **Val Loss**: 12.5m ~ 13.0m
- **Public LB**: 13.0m ~ 13.5m
- **결과**: ✅ **LightGBM 초과 성공!**

#### ✅ 현실적 (대부분 개선사항 효과적)
- **Val Loss**: 13.5m ~ 14.0m
- **Public LB**: 14.0m ~ 14.5m
- **결과**: ✅ LightGBM 근접 또는 동등

#### 📈 보수적 (일부 개선사항만 효과적)
- **Val Loss**: 14.0m ~ 14.5m
- **Public LB**: 14.5m ~ 15.0m
- **결과**: 📊 추가 튜닝 필요

---

## 🛠️ 다음 단계 (성능 미달 시)

### 1단계: 하이퍼파라미터 재튜닝
- Learning Rate, Dropout, Hidden Dim 조정
- 더 긴 학습 (Epoch 증가)

### 2단계: Data Augmentation
- 시퀀스 역순
- 좌우 대칭
- Gaussian Noise
- Mixup

### 3단계: 고급 모델 시도
- **Transformer** (Self-Attention only)
- **TCN** (Temporal Convolutional Network)
- **CNN-LSTM Hybrid**

### 4단계: 피처 엔지니어링
- 추가 시퀀스 피처 (누적 거리, 방향 전환 등)
- 게임 상황 컨텍스트 (점수, 시간대 등)

---

## 📚 주요 문서

### 1. `LSTM_PERFORMANCE_IMPROVEMENT_STRATEGY.md`
- 문제 원인 분석
- 단계별 성능 향상 전략
- 실험 우선순위
- 예상 성능 개선 경로

### 2. `LSTM_V5_EXECUTION_GUIDE.md`
- 실행 순서 상세 가이드
- 문제 해결 (메모리 부족, 수렴 실패 등)
- 하이퍼파라미터 튜닝 가이드
- 체크리스트

### 3. `PROJECT_FINAL_SUMMARY.md` (이 문서)
- 프로젝트 전체 개요
- 성능 히스토리
- 파일 구조
- 핵심 개선사항

---

## 🔍 기술적 세부사항

### 데이터 흐름

```
CSV (Wide Format)
    ↓
SoccerDatasetV5
    ↓ Reshape + Normalize
3D Tensor (Batch, 20, Features)
    ↓
Embedding (Categorical)
    ↓
Input Projection (Linear)
    ↓
Bidirectional RNN
    ↓
Multi-Head Attention (with Padding Mask)
    ↓
Residual + LayerNorm
    ↓
Deep Output Head (3-Layer MLP)
    ↓
(target_x, target_y)
```

### 손실 함수

```python
class EuclideanDistanceLoss(nn.Module):
    def forward(self, pred, target):
        # 정규화된 좌표 → 실제 좌표
        pred_real = pred * [105, 68]
        target_real = target * [105, 68]
        
        # 유클리드 거리
        distances = √(Σ(pred - target)²)
        return mean(distances)
```

### 학습 안정화 기법
- **Gradient Clipping**: `max_norm=1.0`
- **Weight Decay**: `1e-3`
- **LayerNorm**: 각 레이어 출력 정규화
- **CosineAnnealingWarmRestarts**: 주기적 LR 조정
- **Early Stopping**: Patience=20

---

## 📊 성능 지표

### 평가 지표
- **Primary**: 유클리드 거리 (미터)
- **Formula**: `√((pred_x - true_x)² + (pred_y - true_y)²)`

### 목표
- **최소 목표**: LightGBM (14.138m)와 동등
- **이상적 목표**: 13.0m 이하 (약 8% 개선)
- **도전적 목표**: 12.5m 이하 (약 12% 개선)

---

## 🎓 학습된 교훈

### 1. 딥러닝 vs 트리 모델
- **트리 모델 강점**: 빠른 학습, 안정적 성능
- **딥러닝 강점**: 시퀀스 패턴 학습, 확장성
- **결론**: 시퀀스 데이터에서는 딥러닝이 유리할 수 있음

### 2. Attention 메커니즘의 중요성
- 단순 RNN보다 Attention이 중요한 시점을 학습
- Multi-Head로 다양한 패턴 포착

### 3. 데이터 전처리의 중요성
- Padding Mask로 실제 데이터 구분
- 전체 피처 정규화로 학습 안정성 향상

### 4. 5-Fold CV + TTA의 효과
- 단일 모델보다 일반화 성능 향상
- TTA로 예측 안정성 증가

---

## 🔮 향후 발전 방향

### 단기 (1주일)
1. V5 모델 학습 및 성능 평가
2. 하이퍼파라미터 파인튜닝
3. Data Augmentation 실험

### 중기 (1개월)
1. Transformer 모델 시도
2. Hybrid 모델 (CNN + RNN) 실험
3. 피처 엔지니어링 고도화

### 장기 (2개월+)
1. Graph Neural Network (선수 간 관계)
2. 강화학습 기반 패스 전략 학습
3. 실시간 예측 시스템 구축

---

## 📞 문의 및 기여

### 문제 보고
- 에러 발생 시: 에러 메시지 + 환경 정보 + 실행 로그
- 성능 문제: Val Loss + Hyperparameters + 학습 곡선

### 기여 방법
1. 새로운 피처 추가
2. 모델 아키텍처 개선
3. 하이퍼파라미터 최적화
4. 문서 개선

---

## 📅 타임라인

- **2025-12-18**: LSTM V4 Baseline 구현 (15.649m)
- **2025-12-19**: LSTM V5 Attention 모델 설계 및 코드 작성
- **2025-12-19 (예정)**: V5 단일 Fold 학습
- **2025-12-19~20 (예정)**: V5 5-Fold 전체 학습
- **2025-12-20 (예정)**: 최종 제출 및 성능 평가

---

## 🏆 목표 달성 기준

### ✅ 최소 목표
- [ ] LSTM V5 Val Loss < 14.5m
- [ ] Public LB < 15.0m

### ✅ 핵심 목표
- [ ] LSTM V5 Val Loss < 14.0m
- [ ] Public LB < 14.5m
- [ ] LightGBM 근접 또는 동등

### 🎉 이상적 목표
- [ ] LSTM V5 Val Loss < 13.0m
- [ ] Public LB < 13.5m
- [ ] **LightGBM 초과 성공!**

---

**프로젝트 상태**: 🚀 **진행 중 (V5 학습 대기)**  
**마지막 업데이트**: 2025-12-19  
**다음 단계**: `train_lstm_v5_attention.py` 실행 (Colab/GPU)

---

## 🎯 핵심 메시지

> **"순수 LSTM 모델의 구조적 개선만으로 LightGBM 성능을 초과할 수 있다"**

이를 위해:
1. ✅ **Attention 메커니즘** - 중요 시점 학습
2. ✅ **Padding Mask** - 데이터 품질 향상
3. ✅ **Bidirectional RNN** - 양방향 정보 활용
4. ✅ **5-Fold + TTA** - 일반화 성능 극대화

**지금 바로 `train_lstm_v5_attention.py`를 실행하여 새로운 성능을 확인하세요!** 🚀

