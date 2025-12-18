# PyTorch LSTM/GRU 기반 K리그 패스 예측 - V4

V4 Wide Format 데이터를 사용한 딥러닝(LSTM/GRU) 기반 패스 도착 위치 예측 모델

## 📌 주요 특징

### 1. **데이터 정규화 (Normalization)**
- **X 좌표 관련 컬럼**: `start_x`, `end_x`, `dx` → `/105` (필드 X축 길이)
- **Y 좌표 관련 컬럼**: `start_y`, `end_y`, `dy` → `/68` (필드 Y축 길이)
- **타겟 정규화**: `target_x`, `target_y`도 동일하게 정규화 후 학습
- **효과**: 딥러닝 학습 안정성 대폭 향상, 수렴 속도 개선

### 2. **Input Projection Layer**
- 수치형 피처 + Embedding을 concatenate한 후, `nn.Linear(input_dim, hidden_dim)`을 통해 차원 변환
- LSTM/GRU 입력 전에 차원을 통일하여 표현력 향상
- **Architecture**: `[Numerical + Embedding] → Projection → LSTM/GRU → Output Head`

### 3. **NaN 처리**
- V4 데이터의 앞부분 패딩(NaN)을 `torch.nan_to_num(x, 0.0)`으로 0 변환
- 모델이 패딩을 자연스럽게 학습

### 4. **Categorical Embedding**
- `type_id`, `res_id`, `team_id_enc`, `is_home`, `is_last`, `period_id` → Embedding 레이어 사용
- 어휘 크기 자동 계산, Embedding 차원 휴리스틱 적용

### 5. **Euclidean Distance Loss**
- 평가지표(유클리드 거리)와 동일한 손실 함수 사용
- MSE 대신 직접적인 거리 최적화

---

## 🚀 실행 방법

### 1. PyTorch 설치
```bash
# CPU 버전 (Windows)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# GPU 버전 (CUDA 11.8)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 2. 학습 실행
```bash
python train_lstm_v4.py
```

**출력:**
- 모델 저장: `lstm_model_v4_best.pth`
- Best Validation Loss (유클리드 거리)
- 학습 로그 (Epoch별 Train/Val Loss)

### 3. 추론 실행
```bash
python inference_lstm_v4.py
```

**출력:**
- 제출 파일: `submission_lstm_v4_YYYYMMDD_HHMMSS.csv`
- 예측 통계 (end_x, end_y 분포)

---

## 📊 하이퍼파라미터

| Parameter | Value | 설명 |
|-----------|-------|------|
| `K` | 20 | 시퀀스 길이 (마지막 20개 이벤트) |
| `BATCH_SIZE` | 128 | 배치 크기 |
| `HIDDEN_DIM` | 256 | LSTM/GRU hidden dimension |
| `NUM_LAYERS` | 2 | LSTM/GRU 레이어 수 |
| `DROPOUT` | 0.3 | Dropout 비율 |
| `LEARNING_RATE` | 1e-3 | 초기 학습률 |
| `NUM_EPOCHS` | 50 | 최대 에포크 |
| `EARLY_STOPPING_PATIENCE` | 10 | Early stopping patience |
| `USE_LSTM` | False | False: GRU, True: LSTM |

---

## 🏗️ 모델 아키텍처

```
Input: (Batch, SeqLen=20, Features)
  ↓
[Numerical Features]  [Categorical Features]
  ↓                         ↓
  ↓                    Embeddings (type_id, res_id, team_id_enc, ...)
  ↓                         ↓
  └─────── Concatenate ─────┘
              ↓
     Input Projection (Linear)
              ↓
       GRU/LSTM (2 layers)
              ↓
    Last Hidden State
              ↓
      Output Head (FC)
              ↓
     (target_x, target_y)
```

---

## 📈 성능 비교

| 모델 | Validation Loss | 예상 Test 점수 | 특징 |
|------|----------------|---------------|------|
| LightGBM V3 | ~1.5m | 14점대 | 시퀀스 모델링, 안정적 |
| LightGBM V4 | ~1.5m | 13~15점대 | V2 피처 + V3 시퀀스 |
| **LSTM/GRU V4** | **?** | **13~16점대** | **딥러닝 시퀀스 학습** |

---

## 🔧 튜닝 포인트

### 1. Hidden Dimension
- 현재: 256
- 시도: 128, 512 (모델 복잡도 조절)

### 2. Learning Rate
- 현재: 1e-3
- 시도: 5e-4, 1e-4 (더 안정적인 학습)

### 3. Dropout
- 현재: 0.3
- 시도: 0.2, 0.5 (과적합 조절)

### 4. RNN Type
- 현재: GRU
- 시도: LSTM (더 긴 메모리 필요 시)

### 5. Num Layers
- 현재: 2
- 시도: 3, 4 (더 깊은 모델)

---

## 📁 파일 구조

```
train_lstm_v4.py          # 학습 스크립트
inference_lstm_v4.py      # 추론 스크립트
lstm_model_v4_best.pth    # 학습된 모델 체크포인트
processed_train_data_v4.csv  # V4 전처리 학습 데이터
processed_test_data_v4.csv   # V4 전처리 테스트 데이터
submission_lstm_v4_*.csv     # 제출 파일
```

---

## 🎯 다음 단계

### 1. 5-Fold 앙상블 학습
- 현재: Fold 1만 사용 (프로토타이핑)
- 개선: 전체 5-Fold 학습 → 앙상블 예측
- 예상 개선: 0.2~0.5m

### 2. LightGBM + LSTM 앙상블
```python
# Weighted Average
final_pred = 0.5 * lgbm_pred + 0.5 * lstm_pred
```

### 3. Attention Mechanism 추가
- Self-Attention으로 중요 이벤트 자동 가중치 부여
- Transformer 기반 모델 시도

### 4. Data Augmentation
- 시퀀스 길이 변화 (K=15, 25)
- 노이즈 추가

---

## 💡 핵심 개선사항

### LightGBM 대비 LSTM/GRU의 장점
1. **시계열 패턴 학습**: RNN의 hidden state가 시간에 따른 패턴을 더 잘 포착
2. **비선형 변환**: 딥러닝의 강력한 표현력
3. **End-to-End 학습**: Embedding까지 함께 최적화

### 주의사항
1. **학습 시간**: LightGBM보다 느림 (GPU 권장)
2. **데이터 부족**: 15,435 샘플로는 과적합 가능 → Regularization 중요
3. **하이퍼파라미터 민감도**: 튜닝 필요

---

## 📞 문의

학습 중 오류 발생 시:
1. PyTorch 설치 확인
2. CUDA 버전 확인 (GPU 사용 시)
3. 메모리 부족 → `BATCH_SIZE` 줄이기
4. `processed_train_data_v4.csv` 경로 확인

---

**작성일**: 2025-12-18  
**버전**: V4 (Wide Format + Deep Learning)  
**목표**: LightGBM 대비 0.3~0.8m 개선 (Test 13~15점대)

