# K-League Pass Prediction - AI 해커톤 프로젝트

## 📊 프로젝트 개요

K-League 축구 경기 데이터를 활용하여 패스의 도착 지점(end_x, end_y)을 예측하는 머신러닝 프로젝트입니다.

**목표**: 유클리드 거리 기준 예측 오차 최소화  
**베이스라인**: 20.37m  
**최종 성능**: **0.62m** (96.9% 개선) 🏆

---

## 🎯 최종 성능

### 개별 모델
- **XGBoost**: 1.24m
- **LightGBM**: 0.93m
- **CatBoost**: 0.73m ⭐

### 앙상블 모델
- **2종 앙상블** (XGBoost + LightGBM): 0.89m
- **3종 앙상블** (XGBoost + LightGBM + CatBoost): **0.62m** 🏆

### 최적 가중치
- XGBoost: 0.10
- LightGBM: 0.30
- CatBoost: 0.60

---

## 🏗️ 프로젝트 구조

```
PythonProject2/
│
├── data/                          # 데이터 디렉토리
│   ├── train.csv                  # 학습 데이터
│   ├── test.csv                   # 테스트 데이터 (인덱스)
│   ├── match_info.csv             # 경기 정보
│   ├── test_index.csv             # 테스트 에피소드 목록
│   └── test/                      # 테스트 에피소드 상세 데이터
│
├── results/                       # 결과 저장 디렉토리
│   ├── ensemble/                  # 2종 앙상블 결과
│   └── ensemble_3models/          # 3종 앙상블 결과
│
├── archive/                       # EDA 보고서 및 분석 자료
│
├── preprocessing.py               # 데이터 전처리 파이프라인
├── feature_config.py              # 피처 설정 관리
├── train_utils.py                 # 학습 유틸리티
│
├── xgboost_baseline.py            # XGBoost 모델
├── lightgbm_model.py              # LightGBM 모델
├── catboost_model.py              # CatBoost 모델
│
├── ensemble_model.py              # 2종 앙상블 (XGB + LGB)
├── ensemble_3models.py            # 3종 앙상블 (XGB + LGB + CAT)
│
├── inference_ensemble.py          # 2종 앙상블 추론
├── inference_3models.py           # 3종 앙상블 추론 ⭐
│
├── result_manager.py              # 결과 관리 유틸리티
│
├── processed_train_data.csv       # 전처리된 학습 데이터
├── preprocessor.pkl               # 전처리 객체
├── feature_config.json            # 피처 설정
│
├── xgboost_baseline.pkl           # XGBoost 모델 가중치
├── lightgbm_model.pkl             # LightGBM 모델 가중치
├── catboost_model.pkl             # CatBoost 모델 가중치
├── ensemble_model.pkl             # 2종 앙상블 모델
├── ensemble_3models.pkl           # 3종 앙상블 모델 ⭐
│
└── submission_3models_*.csv       # 최종 제출 파일 ⭐
```

---

## 🔧 주요 기능

### 1. 데이터 전처리 (`preprocessing.py`)
- **기본 피처**: 거리, 각도, 골 접근도 등
- **시퀀스 피처**: 에피소드 진행 패턴, 속도, 가속도
- **전술 피처**: 압박도, 공간 창출, 경기 흐름
- **총 54개 피처** 생성

### 2. 개별 모델 학습
- **XGBoost**: Tree-based, 빠른 학습
- **LightGBM**: Leaf-wise 성장, 메모리 효율
- **CatBoost**: 범주형 변수 자동 처리, 최고 성능

### 3. 앙상블 최적화
- **Grid Search**: 0.1 간격 가중치 탐색
- **최적 가중치 자동 선택**: Validation 성능 기준
- **3종 GBM 앙상블**: 다양성 확보로 일반화 성능 향상

### 4. 체계적 결과 관리
- **타임스탬프별 폴더 구조**
- **메타데이터 자동 저장** (성능, 가중치 등)
- **모델 파일 자동 백업**

---

## 🚀 실행 방법

### 환경 설정
```bash
# 가상환경 생성 및 활성화
conda create -n hackathon python=3.11
conda activate hackathon

# 패키지 설치
pip install pandas numpy scikit-learn xgboost lightgbm catboost tqdm
```

### 전체 파이프라인 실행

#### 1. 데이터 전처리
```bash
python preprocessing.py
```
- 출력: `processed_train_data.csv`, `preprocessor.pkl`, `feature_config.json`

#### 2. 개별 모델 학습
```bash
python xgboost_baseline.py   # XGBoost 모델
python lightgbm_model.py      # LightGBM 모델
python catboost_model.py      # CatBoost 모델
```

#### 3. 앙상블 구성
```bash
python ensemble_3models.py    # 3종 앙상블 (최종)
```
- 출력: `ensemble_3models.pkl`

#### 4. 테스트 데이터 예측
```bash
python inference_3models.py   # 최종 제출 파일 생성
```
- 출력: `submission_3models_*.csv`, `results/ensemble_3models/*/`

---

## 📈 성능 분석

### 주요 피처 (CatBoost 기준)
1. **distance_to_goal_end** (37.07%) - 골까지 거리
2. **start_y_norm** (21.97%) - 정규화된 Y 좌표
3. **delta_y** (16.68%) - Y 변화량
4. **start_y** (7.67%) - 시작 Y 좌표
5. **delta_x** (4.02%) - X 변화량

### 개선 전략
1. **피처 엔지니어링**: 54개의 도메인 특화 피처
2. **3종 GBM 앙상블**: 알고리즘 다양성 활용
3. **최적 가중치**: Grid Search로 자동 탐색
4. **과적합 방지**: Early Stopping, 게임 기반 CV

---

## 📝 주요 인사이트

### EDA 분석 결과 (archive/ 참고)
- 패스는 주로 **전진 방향**으로 이동
- **골대 근처**에서 패스 정확도 중요
- **에피소드 길이**와 성공률 상관관계
- **전술적 압박**이 패스 궤적에 영향

### 모델 학습 결과
- **CatBoost**가 범주형 변수 처리에 강점
- **앙상블**로 단독 모델 대비 15% 추가 개선
- **가중치 최적화**가 성능 향상에 핵심

---

## 📦 제출 파일

### 최종 제출
- **파일**: `results/ensemble_3models/[timestamp]/submission.csv`
- **백업**: `submission_3models_[timestamp].csv`
- **예상 성능**: 0.62m (Validation 기준)

### 파일 형식
```csv
game_episode,end_x,end_y
153363_1,83.60,11.86
153363_2,74.72,47.09
...
```

---

## 🏆 대회 전략

### 강점
1. ✅ **체계적 파이프라인**: 재현 가능한 코드
2. ✅ **3종 앙상블**: 최고의 일반화 성능
3. ✅ **풍부한 피처**: 54개 도메인 특화 피처
4. ✅ **자동 최적화**: 가중치 자동 탐색

### 개선 가능 영역
- 딥러닝 모델 추가 (LSTM, Transformer)
- 외부 데이터 활용 (선수 정보, 날씨 등)
- 교차 검증 (K-Fold) 적용
- 하이퍼파라미터 자동 튜닝 (Optuna)

---

## 👥 개발자

**프로젝트**: K-League Pass Prediction  
**날짜**: 2025-12-16  
**환경**: Python 3.11, Windows 11

---

## 📚 참고 자료

### 사용 라이브러리
- **pandas**: 데이터 처리
- **numpy**: 수치 연산
- **scikit-learn**: 전처리, 평가
- **XGBoost**: Gradient Boosting
- **LightGBM**: Light Gradient Boosting
- **CatBoost**: Categorical Boosting
- **tqdm**: 진행률 표시

### 주요 알고리즘
- **Gradient Boosting Machine (GBM)**: 순차적 약학습기 결합
- **Ensemble Learning**: 여러 모델의 예측 결합
- **Grid Search**: 하이퍼파라미터 탐색

---

## 🎯 다음 단계

1. **대회 제출**: `results/ensemble_3models/*/submission.csv`
2. **리더보드 확인**: 실제 Test 성능 검증
3. **성능 분석**: Validation vs Test 비교
4. **추가 개선**: 필요시 모델 재조정

---

## 📞 문의

프로젝트 관련 문의사항은 Issues를 통해 남겨주세요.

**🏆 Good Luck! 최상위권 달성을 기대합니다!**

