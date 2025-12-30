# 🚀 LSTM 모델 성능 향상 전략

## 📊 현재 상황 분석
- **LightGBM 최고 성능**: 14.138m
- **LSTM 현재 성능**: 15.649m (약 **10.6% 성능 차이**)
- **목표**: LSTM 단독으로 LightGBM 성능을 **초과**하거나 동등 수준 달성

---

## 🔍 문제 원인 분석

### 1. **데이터 표현력 부족**
- **Wide Format의 한계**: 시퀀스 정보가 펼쳐져 있어 시간적 패턴 학습이 어려움
- **NaN 처리 방식**: 단순히 0으로 채우면 실제 0값과 구분 불가
- **정규화 범위**: 좌표만 정규화했지만, 속도/각도 등 다른 피처들은 스케일 불균형

### 2. **모델 구조적 한계**
- **Attention 메커니즘 부재**: 마지막 Hidden State만 사용 → 중요한 시점 정보 손실
- **단순한 Output Head**: 2-Layer MLP는 복잡한 패턴 학습에 부족
- **시퀀스 길이 고정**: K=20이지만 실제 유효 시퀀스는 더 짧을 수 있음

### 3. **하이퍼파라미터 미최적화**
- Learning Rate, Hidden Dim, Batch Size 등이 경험적 값
- Dropout, Weight Decay 등 regularization 미세조정 필요

---

## 🎯 단계별 성능 향상 전략

### **Phase 1: 데이터 전처리 개선** (예상 개선폭: 5~10%)

#### 1.1 Padding Mask 활용
```python
# Dataset에서 padding mask 생성
def create_padding_mask(self, numerical_tensor):
    # 모든 feature가 0인 시점 = Padding
    mask = (numerical_tensor.sum(dim=-1) == 0).float()  # (batch, seq_len)
    return mask
```
- RNN에서 실제 데이터가 있는 시점만 집중하도록 유도

#### 1.2 전체 피처 정규화/표준화
```python
# StandardScaler 또는 MinMaxScaler 적용
# - 속도: 0~max_speed
# - 각도: -π ~ π → -1 ~ 1
# - 시간차: 0~max_time
```

#### 1.3 추가 시퀀스 피처 생성
```python
# - 누적 거리 (cumulative distance)
# - 방향 전환 빈도 (direction change)
# - 패스 리듬 (time between passes)
```

---

### **Phase 2: 모델 아키텍처 고도화** (예상 개선폭: 10~20%)

#### 2.1 Multi-Head Attention 추가 ⭐⭐⭐
```python
class SoccerRNNWithAttention(nn.Module):
    def __init__(self, ...):
        # ...existing layers...
        
        # Multi-Head Attention
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )
        
    def forward(self, num_feat, cat_feat, padding_mask):
        # RNN 출력
        rnn_out, _ = self.rnn(x)  # (batch, seq_len, hidden_dim)
        
        # Attention (중요 시점 가중치 학습)
        attn_out, attn_weights = self.attention(
            rnn_out, rnn_out, rnn_out,
            key_padding_mask=padding_mask  # 패딩 무시
        )
        
        # Attention 출력 사용
        last_hidden = attn_out[:, -1, :]
        output = self.fc(last_hidden)
        return output
```

#### 2.2 Bidirectional RNN
```python
self.rnn = nn.GRU(
    hidden_dim,
    hidden_dim,
    num_layers=num_layers,
    dropout=dropout,
    batch_first=True,
    bidirectional=True  # ✅ 양방향
)

# Output Dim이 2배가 되므로 조정 필요
self.fc = nn.Linear(hidden_dim * 2, ...)
```

#### 2.3 깊은 Output Head
```python
self.fc = nn.Sequential(
    nn.Linear(hidden_dim, hidden_dim),
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

#### 2.4 Residual Connection
```python
# Input Projection 후 Residual 추가
x_proj = self.input_projection(x)
rnn_out, _ = self.rnn(x_proj)
rnn_out = rnn_out + x_proj  # Residual
```

---

### **Phase 3: 학습 전략 개선** (예상 개선폭: 5~10%)

#### 3.1 Loss Function 개선
```python
class WeightedEuclideanLoss(nn.Module):
    def forward(self, pred, target):
        # X축 오차에 더 큰 가중치 (골대 방향이 중요)
        diff = pred - target
        diff[:, 0] *= 1.5  # X축 가중치
        diff[:, 1] *= 1.0  # Y축 가중치
        
        distances = torch.sqrt(torch.sum(diff ** 2, dim=1))
        return distances.mean()
```

#### 3.2 Curriculum Learning
```python
# 쉬운 샘플부터 학습 (짧은 시퀀스 → 긴 시퀀스)
# Epoch 1~10: 시퀀스 길이 5~10
# Epoch 11~20: 시퀀스 길이 10~15
# Epoch 21+: 전체 시퀀스
```

#### 3.3 Data Augmentation
```python
# 1. 시퀀스 역순 (패스 방향 반대로)
# 2. 좌우 대칭 (필드 중심 기준)
# 3. 노이즈 추가 (Gaussian noise)
```

#### 3.4 학습률 스케줄러 개선
```python
# CosineAnnealingWarmRestarts
scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer, T_0=10, T_mult=2, eta_min=1e-6
)
```

---

### **Phase 4: 앙상블 없이 단일 모델 극대화** (예상 개선폭: 5~10%)

#### 4.1 5-Fold Cross Validation
- 5개 Fold 모두 학습 후, **Test 시점에 5개 모델의 평균** 사용
- 단일 모델이지만 일반화 성능 향상

#### 4.2 Test Time Augmentation (TTA)
```python
# Test 시 원본 + 좌우 반전 + 노이즈 추가 등 여러 버전 예측 후 평균
predictions = []
for augmentation in [original, flip_lr, add_noise]:
    pred = model(augmentation)
    predictions.append(pred)

final_pred = torch.mean(torch.stack(predictions), dim=0)
```

#### 4.3 Pseudo Labeling (준지도 학습)
```python
# 1. Train 데이터로 모델 학습
# 2. Test 데이터 예측 (Pseudo Label 생성)
# 3. 높은 신뢰도 예측만 Train 데이터에 추가
# 4. 재학습
```

---

## 🛠️ 구현 우선순위

### **최우선 (즉시 구현)** 🔥
1. **Multi-Head Attention 추가** → 가장 큰 성능 향상 예상
2. **Padding Mask 활용** → 데이터 품질 개선
3. **전체 피처 정규화** → 학습 안정성 향상

### **고우선 (1주일 내)**
4. **Bidirectional RNN** → 시퀀스 양방향 정보 활용
5. **깊은 Output Head** → 복잡한 패턴 학습
6. **5-Fold CV** → 일반화 성능 향상

### **중우선 (시간 여유 있을 때)**
7. **Data Augmentation** → 데이터 부족 문제 완화
8. **Curriculum Learning** → 학습 효율성 향상
9. **Test Time Augmentation** → 추론 안정성 향상

---

## 📈 예상 성능 개선 경로

```
현재 LSTM: 15.649m
    ↓
Phase 1 (데이터 개선): 14.5m ~ 15.0m  [-1.1m]
    ↓
Phase 2 (Attention): 13.5m ~ 14.0m  [-1.0m]
    ↓
Phase 3 (학습 전략): 13.0m ~ 13.5m  [-0.5m]
    ↓
Phase 4 (5-Fold + TTA): 12.5m ~ 13.0m  [-0.5m]
    ↓
최종 목표: 12.5m ~ 13.0m (LightGBM 14.138m 초과!)
```

---

## 🔬 실험 로그 템플릿

### 실험 1: Baseline (현재)
- **모델**: GRU (Hidden=256, Layers=2)
- **Loss**: EuclideanDistance
- **Val Loss**: 15.649m
- **비고**: Wide Format, 단순 NaN→0 처리

### 실험 2: Attention + Padding Mask
- **변경사항**: Multi-Head Attention (8 heads) 추가
- **Val Loss**: [TBD]
- **개선폭**: [TBD]

### 실험 3: Bidirectional RNN
- **변경사항**: Bidirectional GRU
- **Val Loss**: [TBD]
- **개선폭**: [TBD]

(이후 실험 결과 누적 기록)

---

## 🎯 최종 목표

> **"순수 LSTM 단일 모델로 LightGBM (14.138m)을 능가하여 12.5m~13.0m 달성"**

### 성공 기준
✅ Val Loss < 13.0m  
✅ Public Leaderboard < 13.5m  
✅ Private Leaderboard < 14.0m  

---

## 📚 참고 문헌

1. **Attention is All You Need** (Vaswani et al., 2017)
2. **Sequence to Sequence Learning with Neural Networks** (Sutskever et al., 2014)
3. **Deep Residual Learning** (He et al., 2016)
4. **Test-Time Augmentation** (Shanmugam et al., 2020)

---

**작성일**: 2025-12-19  
**작성자**: AI Sports Data Analyst

