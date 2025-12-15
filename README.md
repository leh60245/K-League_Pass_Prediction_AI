# ⚽ K-League Pass Prediction AI

**마지막 패스 도달 좌표 예측 프로젝트**

---

## 📁 프로젝트 구조

```
PythonProject2/
│
├── 📊 데이터 (data/)
│   ├── train.csv                    # 학습 데이터
│   ├── test.csv                     # 테스트 데이터
│   ├── match_info.csv               # 경기 정보
│   ├── sample_submission.csv        # 제출 양식
│   └── test/                        # 테스트 상세 데이터
│
├── 🔧 핵심 코드
│   ├── preprocessing.py             # 전처리 파이프라인 (54개 피처)
│   ├── feature_config.py            # 피처 설정 관리
│   ├── train_utils.py               # 공통 유틸리티 함수
│   │
│   ├── xgboost_baseline.py          # XGBoost 모델 (Val: 1.24m)
│   ├── lightgbm_model.py            # LightGBM 모델 (Val: 0.93m) ⭐
│   ├── ensemble_model.py            # 앙상블 모델 (예상: 0.98m)
│   │
│   ├── inference.py                 # 추론 스크립트
│   └── train_with_tactical_features.py  # 전술 피처 학습
│
├── 📄 설정 파일
│   ├── feature_config.json          # 피처 설정 (54개)
│   ├── preprocessor.pkl             # 전처리 설정
│   ├── requirements.txt             # 패키지 목록
│   └── requirements_raw.txt         # 원본 요구사항
│
├── 💾 모델 파일
│   ├── xgboost_baseline.pkl         # XGBoost 모델
│   ├── lightgbm_model.pkl           # LightGBM 모델
│   ├── ensemble_model.pkl           # 앙상블 모델
│   ├── final_ensemble_model.pkl     # 최종 모델
│   └── models/                      # 실험 모델들
│       └── tactical_features_*/     # 전술 피처 모델
│
├── 📂 정리된 폴더
│   ├── docs/                        # 📚 모든 문서
│   │   ├── 피처_관리_시스템_가이드.md
│   │   ├── 전술_피처_엔지니어링_가이드.md
│   │   ├── 전술_피처_완료_보고서.md
│   │   ├── lightgbm_업데이트_완료.md
│   │   └── 질문_답변_완료.md
│   │
│   ├── archive/                     # 📦 EDA 분석 파일
│   │   ├── EDA_Phase1_insights.txt
│   │   ├── EDA_Phase2_insights.txt
│   │   ├── EDA_Phase3_insights.txt
│   │   ├── eda_phase1.py
│   │   ├── eda_phase2.py
│   │   └── eda_phase3.py
│   │
│   └── old_files/                   # 🗄️ 사용하지 않는 파일
│       ├── sample_by_other.py
│       ├── train_and_predict.py
│       └── final_*.py
│
└── 📈 결과 파일
    ├── submission.csv               # 제출 파일
    ├── submission_ensemble.csv      # 앙상블 제출
    └── processed_train_data.csv     # 전처리된 학습 데이터

```

---

## 🚀 빠른 시작

### 1️⃣ 환경 설정
```bash
# 패키지 설치
pip install -r requirements.txt
```

### 2️⃣ 데이터 전처리
```bash
# 전처리 실행 (feature_config.json 자동 생성)
python preprocessing.py
```

### 3️⃣ 모델 학습
```bash
# XGBoost 학습
python xgboost_baseline.py

# LightGBM 학습 (권장)
python lightgbm_model.py

# 앙상블 학습
python ensemble_model.py
```

### 4️⃣ 추론 및 제출
```bash
# 테스트 데이터 추론
python inference.py

# 제출 파일: submission.csv
```

---

## 📊 성능 요약

| 모델 | Val RMSE | 유클리드 거리 | 비고 |
|------|----------|-------------|------|
| **베이스라인** | - | 20.37m | 시작=도착 |
| **XGBoost** | 0.95m | 1.24m | 기본 모델 |
| **LightGBM** | 0.93m | 0.93m | **최고 성능** ⭐ |
| **앙상블** | - | 0.98m (예상) | XGBoost + LightGBM |

---

## 🎯 핵심 피처 (54개)

### 1️⃣ 기본 공간 (7개)
- start_x/y, delta_x/y, distance
- start_x/y_norm (정규화)

### 2️⃣ 골 관련 (4개)
- distance_to_goal_start/end
- goal_approach, shooting_angle

### 3️⃣ 영역 분할 (5개)
- start_x/y_zone, start_x_zone_fine
- in_penalty_area, in_final_third

### 4️⃣ 전술 피처 (20개)
- **압박**: local_pressure, weighted_pressure, event_density
- **공간**: distance_change_rate, attack_width, vertical_spread
- **방향**: direction_consistency, pass_angle_change
- **속도**: velocity, acceleration, avg_velocity_3
- **경로**: path_efficiency, forward_momentum
- **템포**: tempo, tempo_change, match_phase

### 5️⃣ 시퀀스 (18개)
- episode_length, x_progression, relative_time
- prev_*, prev2_* (이전 이벤트)
- type_name_encoded, result_name_encoded

---

## 💡 주요 특징

### ✅ JSON 기반 피처 관리
- `feature_config.json`으로 피처 자동 관리
- 피처 추가/제거 시 코드 수정 불필요
- 모든 모델이 동일한 설정 사용

### ✅ 공통 유틸리티 함수
- `train_utils.py`로 중복 코드 제거
- 데이터 로딩, Train/Val 분할, 평가 자동화
- 코드 50~90% 단축

### ✅ 전술적 피처 엔지니어링
- 골 각도 (Shooting Angle)
- 압박 강도 (Pressure Intensity)
- 진행 방향성 (Direction Consistency)
- 경로 효율성 (Path Efficiency)

---

## 📚 문서

### 필독 문서 (docs/)
1. **피처_관리_시스템_가이드.md** - 피처 자동 관리 시스템
2. **전술_피처_엔지니어링_가이드.md** - 전술 피처 상세 설명
3. **전술_피처_완료_보고서.md** - 프로젝트 완료 보고
4. **질문_답변_완료.md** - FAQ 및 주요 이슈

### 참고 문서
- lightgbm_업데이트_완료.md - LightGBM 업데이트 내역
- 전술_피처_빠른시작.md - 빠른 시작 가이드

---

## 🔧 주요 함수

### train_utils.py
```python
# 데이터 로딩
data, features, targets, config = load_data_and_features()

# Train/Val 분할
X_train, y_train, X_val, y_val = prepare_train_val_split(data, features, targets)

# 성능 평가
print_performance_summary(train_eucl, val_eucl)

# 피처 그룹별 중요도
group_importance = get_feature_group_importance(model_x, model_y, features, config)
```

### feature_config.py
```python
# 피처 설정 로딩
config = FeatureConfig('feature_config.json')

# 피처 리스트
features = config.get_feature_columns()

# 특정 그룹 피처
tactical = config.get_feature_group('pressure')
```

---

## 📈 실험 추적

### 모델 버전
- v1.0: 기본 피처 (24개)
- v2.0: 전술 피처 추가 (54개)
- v3.0: 피처 관리 시스템 구축
- v3.1: 모든 모델 업데이트 완료

### 성능 개선 히스토리
- 베이스라인: 20.37m
- XGBoost v1: ~15m
- XGBoost v2: 1.24m
- LightGBM: 0.93m (현재 최고)

---

## 🎓 기술 스택

- **언어**: Python 3.11
- **ML 라이브러리**: XGBoost, LightGBM, scikit-learn
- **데이터**: Pandas, NumPy
- **설정 관리**: JSON
- **패턴**: MLOps, 피처 스토어

---

## 👥 팀

K리그 전술 데이터 분석팀

---

## 📝 라이선스

이 프로젝트는 대회 참가용입니다.

---

**최종 업데이트**: 2025-12-16  
**버전**: v3.1 (정리 완료)

