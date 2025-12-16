# V3 프로젝트 최종 보고서

**날짜**: 2025-12-16  
**목표**: Data Leakage 제거 + 시퀀스 모델링으로 Test 성능 개선

---

## 🎯 완료된 작업

### 1. preprocessing_v3.py 개발
✅ Data Leakage 완전 제거
- 마지막 이벤트의 end_x, end_y를 NaN으로 마스킹
- V1의 0.93m은 Leakage로 인한 거짓 성능

✅ 시퀀스 모델링 도입
- 마지막 K=20개 이벤트 사용
- Wide format 변환 (437개 피처)
- 다른 사람의 우수 방식 채택

✅ 5-Fold GroupKFold
- Game-based split
- 안정적인 검증

### 2. train_lightgbm_v3.py 학습
✅ 5-Fold 앙상블 완료
- Validation: 14.40m
- 각 Fold별 안정적 성능

### 3. inference_v3.py 추론
⚠️ **문제 발생**: Wide format 변환 로직 에러
- 모든 예측이 평균값으로 대체됨
- 추가 디버깅 필요

---

## 📊 핵심 인사이트

### V1 vs V3 성능 차이의 진실

**V1 (Data Leakage 있음):**
```
Validation: 0.93m (거짓 성능)
Test: 24점대 (실제 성능)

이유: end_x, end_y를 피처로 사용 → 정답을 보고 학습
→ Train에서는 완벽, Test에서는 폭망
```

**V3 (Leakage 제거):**
```
Validation: 14.40m (정상)
Test: 15~18점대 예상 (Validation과 일치)

이유: 실제 예측력을 정확히 반영
→ Train/Test 성능 일치
```

### 왜 V3 Validation이 더 높은가?

**V1의 0.93m은 착시현상!**
- end_x를 보고 end_x 예측 → 당연히 정확
- Test에는 end_x가 없어서 24점대

**V3의 14.40m이 정상값!**
- 실제 어려운 문제 (과거로 미래 예측)
- Baseline 20.37m 대비 30% 개선
- 다른 사람도 16점대 → 비슷한 난이도

---

## ⚠️ 남은 문제

### 1. inference_v3.py 버그
**현상**: Wide format 변환 에러
```python
Length mismatch: Expected axis has 20 elements, new values have 440 elements
```

**원인**: Pivot 로직이 train과 test에서 다르게 작동

**해결 방안**:
1. sample_from_other.py의 추론 로직 참고
2. 또는 V1 방식 유지 + Leakage만 제거

### 2. 성능 개선 가능성
V3가 제대로 작동하면:
- 예상 Test: 15~18점대
- 추가 튜닝: 12~15점대 가능

---

## 🏆 최종 추천 전략

### Option 1: V1 수정 (빠른 해결)
```python
# preprocessing.py의 extract_last_events만 수정
last_events = last_events.drop(columns=['end_x', 'end_y'], errors='ignore')
```

**장점**:
- 간단한 수정
- 즉시 실행 가능
- 예상 Test: 18~20점대

**단점**:
- 시퀀스 정보 활용 못함

### Option 2: sample_from_other.py 활용
다른 사람의 코드를 그대로 실행

**장점**:
- 검증된 코드
- 16점대 보장

**단점**:
- 우리 코드가 아님

### Option 3: V3 디버깅 (시간 필요)
inference_v3.py의 Wide format 변환 수정

**장점**:
- 가장 우수한 성능 기대

**단점**:
- 추가 시간 필요

---

## ✅ 결론

### 학습한 것
1. **Data Leakage의 위험성**
   - Train 0.93m vs Test 24점 → 극단적 차이
   - Validation 점수만 믿으면 안됨

2. **정상적인 난이도 파악**
   - V3 Validation 14.40m이 정상
   - 과거 이벤트로 미래 예측은 원래 어려움

3. **시퀀스 모델링의 중요성**
   - 마지막 1개 vs 마지막 20개
   - 패스 체인 패턴 학습 필수

### 다음 단계
1. V1 + Leakage 제거로 빠르게 제출
2. V3 디버깅 후 재시도
3. 하이퍼파라미터 튜닝

---

**작성**: AI System  
**상태**: V3 개발 완료 (추론 버그 수정 필요)

