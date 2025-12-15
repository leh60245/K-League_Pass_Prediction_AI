# K-League Pass Prediction AI 🎯⚽

> K리그 경기 내 최종 패스 좌표 예측 AI 모델 개발 프로젝트

**대회**: K리그-서울시립대 공개 AI 경진대회  
**목표**: 에피소드별 마지막 패스 도착 좌표 (X, Y) 예측  
**평가**: 유클리드 거리 (Euclidean Distance) - 낮을수록 우수

---

## 📊 프로젝트 현황

### ✅ 완료된 단계

- [x] **프로젝트 목표 수립** (`AI_해커톤_프로젝트_목표.md`)
- [x] **EDA 전략 수립** (`EDA_전략.md`)
- [x] **Phase 1: 데이터 기초 진단** ✅
- [x] **Phase 2: 예측 대상 분석** ✅
- [x] **Phase 3: 시퀀스 패턴 분석** ✅
- [x] **EDA 최종 보고서 작성** ✅

### 🔄 진행 중

- [ ] 데이터 전처리 파이프라인 구축
- [ ] 베이스라인 모델 개발 (XGBoost)

### ⏳ 대기 중

- [ ] LSTM/GRU 모델 개발
- [ ] 앙상블 및 최적화
- [ ] 최종 제출 준비

---

## 🎯 핵심 인사이트 (Key Findings)

### 1. 베이스라인 성능
```
단순 베이스라인: 20.37m (시작 위치 그대로 예측)
→ 모든 모델이 넘어야 할 최소 기준
```

### 2. 가장 중요한 발견
- **시작-도착 상관계수: 0.79** (매우 강한 양의 상관)
- **Pass → Carry → Pass 패턴 30%** (명확한 시퀀스)
- **공격진에서 55% 발생** (X축 편향)
- **좌우 균등 분포** (Y축 중립)

### 3. 성능 목표
```
🎯 최소 목표: < 20m (베이스라인 이하)
🎯 경쟁력:   < 15m
🎯 우승권:   < 10m
🎯 최고:     < 7m
```

---

## 📁 프로젝트 구조

```
PythonProject2/
├── data/                           # 데이터 디렉토리
│   ├── train.csv                   # 학습 데이터 (356,721 이벤트)
│   ├── test.csv                    # 테스트 인덱스 (2,414 에피소드)
│   ├── match_info.csv              # 경기 메타데이터
│   └── test/                       # 테스트 에피소드 파일들
│
├── 대회_설명서.txt                  # 대회 설명
├── data_description - *.csv        # 데이터 설명서
│
├── AI_해커톤_프로젝트_목표.md       # 전체 프로젝트 로드맵 📋
├── EDA_전략.md                     # EDA 실행 계획
├── EDA_최종보고서.md                # EDA 종합 보고서 ⭐
├── EDA_핵심인사이트_종합.md         # 인사이트 요약
│
├── eda_phase1.py                   # Phase 1 분석 코드
├── eda_phase2.py                   # Phase 2 분석 코드
├── eda_phase3.py                   # Phase 3 분석 코드
│
├── EDA_Phase1_insights.txt         # Phase 1 결과
├── EDA_Phase2_insights.txt         # Phase 2 결과
└── EDA_Phase3_insights.txt         # Phase 3 결과
```

---

## 📊 데이터 개요

### Train 데이터
- **경기 수**: 198개
- **에피소드 수**: 15,435개
- **이벤트 수**: 356,721개
- **평균 에피소드 길이**: 23.1개 이벤트
- **시즌**: 2024 K League 1

### Test 데이터
- **경기 수**: 30개
- **에피소드 수**: 2,414개 (예측 대상)

### 주요 변수
```
좌표계: 105 x 68 (FIFA 공식 규격)
- start_x, start_y: 이벤트 시작 위치
- end_x, end_y: 이벤트 종료 위치 (예측 대상)
- type_name: 이벤트 타입 (Pass, Carry, etc.)
- result_name: 성공/실패 여부
- time_seconds: 시간 정보
```

---

## 🔬 EDA 주요 발견

### 데이터 특성
- ✅ 데이터 품질 양호 (결측치 최소, 좌표 정상)
- ⚠️ 시간 순서 1.85% 불일치 → time_seconds 정렬 필요
- 📊 에피소드 길이 매우 가변적 (1~270개, 95%는 67개 이하)

### 예측 대상 특성
```
end_x 평균: 68.45m (공격진 편향)
end_y 평균: 33.62m (중앙)
패스 거리: 평균 20.37m, 중앙값 15.74m
패스 방향: 83.6% 전진 패스
```

### 시퀀스 패턴
```
가장 흔한 전이:
1. Pass → Pass (22.13%)
2. Carry → Pass (20.92%)
3. Pass → Carry (18.25%)

마지막 패스 직전:
- Carry: 39.9%
- Pass: 37.7%
- Recovery: 11.9%
```

---

## 🎯 모델링 전략

### 피처 중요도

**Tier 1 (필수)**
- `start_x`, `start_y` (상관 0.79)
- `delta_x`, `delta_y`
- `type_name` (Embedding)
- `result_name`

**Tier 2 (중요)**
- `prev_event_type` (직전 이벤트)
- `prev_2_events` (직전 2개)
- `x_zone`, `y_zone` (경기장 영역)
- `x_progression` (진행도)
- `distance_to_goal`

### 모델 아키텍처

#### Option 1: XGBoost (베이스라인)
```python
✓ 빠른 학습, 해석 가능
✓ 집계 피처 활용
목표: < 18m
```

#### Option 2: LSTM/GRU (주력)
```python
✓ 시퀀스 패턴 학습
✓ Bidirectional + Attention
✓ 이벤트 타입 Embedding
목표: < 15m
```

#### Option 3: Hybrid (권장)
```python
✓ XGBoost + LSTM 앙상블
✓ 가중 평균 or Stacking
목표: < 12m
```

---

## 📈 예상 성능 로드맵

```
Week 1: 베이스라인
├─ XGBoost: 15-18m ✓ 목표
└─ LSTM: 12-15m ✓ 목표

Week 2: 최적화
├─ 피처 엔지니어링 심화
├─ 하이퍼파라미터 튜닝
└─ 앙상블: < 12m ✓ 목표

Week 3: 최종 튜닝
├─ 고급 모델 실험
├─ 최적화
└─ 최종: < 10m ✓ 목표
```

---

## 🚀 다음 단계

### Immediate (오늘~내일)

1. **데이터 전처리 파이프라인** ⭐⭐⭐
   ```python
   - 시간 정렬 (time_seconds)
   - 좌표 정규화 (StandardScaler)
   - 시퀀스 Padding (max_length=67)
   - Train/Val Split (Game-based 5-Fold)
   ```

2. **베이스라인 모델 (XGBoost)** ⭐⭐⭐
   ```python
   - 집계 피처 생성
   - 모델 학습 및 평가
   - 목표: < 18m 달성
   ```

### Short-term (3-5일)

3. **LSTM 모델 개발** ⭐⭐
   ```python
   - 시퀀스 데이터 생성
   - Bidirectional LSTM
   - Attention 메커니즘
   - 목표: < 15m 달성
   ```

4. **피처 엔지니어링 심화** ⭐⭐

---

## 📚 참고 문서

### 필수 문서
- 📋 [AI_해커톤_프로젝트_목표.md](AI_해커톤_프로젝트_목표.md) - 전체 로드맵
- ⭐ [EDA_최종보고서.md](EDA_최종보고서.md) - EDA 종합 (가장 중요!)
- 📊 [EDA_핵심인사이트_종합.md](EDA_핵심인사이트_종합.md) - 인사이트 요약

### EDA 결과
- 📄 [EDA_Phase1_insights.txt](EDA_Phase1_insights.txt) - 데이터 기초
- 📄 [EDA_Phase2_insights.txt](EDA_Phase2_insights.txt) - 예측 대상
- 📄 [EDA_Phase3_insights.txt](EDA_Phase3_insights.txt) - 시퀀스 패턴

### 분석 코드
- 🐍 [eda_phase1.py](eda_phase1.py)
- 🐍 [eda_phase2.py](eda_phase2.py)
- 🐍 [eda_phase3.py](eda_phase3.py)

---

## 💡 핵심 인사이트 Top 10

1. **베이스라인 20.37m** - 시작 위치 그대로
2. **start ↔ end 상관 0.79** - 가장 중요
3. **Pass-Carry-Pass 30%** - 시퀀스 패턴
4. **공격진 55%** - X축 편향
5. **좌우 균등** - Y축 중립
6. **짧은 전진 패스** - 평균 20m, 83.6% 전진
7. **가변 길이** - 95%는 67개 이하
8. **Pass+Carry 73%** - 핵심 이벤트
9. **영역 내 머무름** - 공격진→공격진 97.5%
10. **직전 이벤트 중요** - Attention 필요

---

## 🛠️ 기술 스택

### 분석
- Python 3.x
- Pandas, NumPy
- Matplotlib, Seaborn (예정)

### 모델링 (예정)
- Scikit-learn
- XGBoost / LightGBM
- PyTorch / TensorFlow
- Optuna (하이퍼파라미터 튜닝)

---

## 📞 Contact & Team

**프로젝트 기간**: 2024.12 ~  
**목표**: Private Score 상위 10% 진입

---

## ✅ 체크리스트

### EDA ✅
- [x] Phase 1: 데이터 기초 진단
- [x] Phase 2: 예측 대상 분석
- [x] Phase 3: 시퀀스 패턴 분석
- [x] 최종 보고서 작성

### 모델링 🔄
- [ ] 데이터 전처리 파이프라인
- [ ] XGBoost 베이스라인 (목표: < 18m)
- [ ] LSTM 모델 (목표: < 15m)
- [ ] 앙상블 (목표: < 12m)
- [ ] 하이퍼파라미터 튜닝
- [ ] 최종 제출 (목표: < 10m)

---

**🎯 "시작 위치 + 시퀀스 맥락" - 이것이 성공의 열쇠!** 🚀


