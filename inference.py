"""
K-League Pass Prediction - Inference Code
최종 제출용 추론 코드

요구사항:
- 학습 코드와 분리
- 모델 가중치 로딩
- 테스트 데이터 예측
- submission.csv 생성
"""

import pandas as pd
import numpy as np
import pickle
import os
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Preprocessor import (같은 디렉토리에 있어야 함)
from preprocessing import DataPreprocessor

def load_model(model_path='final_model.pkl'):
    """모델 로딩"""
    print(f"📂 모델 로딩 중: {model_path}")
    with open(model_path, 'rb') as f:
        saved = pickle.load(f)
    print("✅ 모델 로딩 완료")
    return saved['model_x'], saved['model_y'], saved['feature_cols']

def preprocess_episode(episode_data, preprocessor, feature_cols):
    """단일 에피소드 전처리"""
    # 시간 정렬
    episode_data = episode_data.sort_values('time_seconds').reset_index(drop=True)

    # 기본 피처 생성
    episode_data = preprocessor.create_basic_features(episode_data, verbose=False)

    # 시퀀스 피처 생성
    episode_data = preprocessor.create_sequence_features(episode_data, verbose=False)

    # 직전 이벤트 피처
    episode_data = preprocessor.create_previous_event_features(episode_data, verbose=False)

    # 마지막 이벤트 추출
    last_event = episode_data.iloc[-1:].copy()

    # 범주형 인코딩
    last_event = preprocessor.encode_categorical(last_event, fit=False, verbose=False)

    # 결측치 처리
    last_event = preprocessor.fill_missing(last_event, verbose=False)

    # 피처 추출
    X = last_event[feature_cols].values

    return X

def predict_test(model_x, model_y, preprocessor, feature_cols,
                 test_index_path='./data/test.csv',
                 data_dir='./data'):
    """테스트 데이터 예측"""

    print("\n" + "=" * 80)
    print("  테스트 데이터 예측")
    print("=" * 80)

    # Test 인덱스 로딩
    print(f"\n📂 Test 인덱스 로딩: {test_index_path}")
    test_index = pd.read_csv(test_index_path)
    print(f"✅ Test 에피소드 수: {len(test_index):,}")

    # 예측
    print("\n🔄 예측 진행 중...")
    predictions = []

    for idx, row in tqdm(test_index.iterrows(), total=len(test_index), desc="Predicting"):
        game_episode = row['game_episode']
        file_path = os.path.join(data_dir, row['path'].replace('./', ''))

        # 에피소드 데이터 로딩
        episode_data = pd.read_csv(file_path)

        # 전처리
        X = preprocess_episode(episode_data, preprocessor, feature_cols)

        # 예측
        pred_x = model_x.predict(X)[0]
        pred_y = model_y.predict(X)[0]

        # 좌표 범위 제한
        pred_x = np.clip(pred_x, 0, 105)
        pred_y = np.clip(pred_y, 0, 68)

        predictions.append({
            'game_episode': game_episode,
            'end_x': pred_x,
            'end_y': pred_y
        })

    print("✅ 예측 완료!")

    return pd.DataFrame(predictions)

def save_submission(submission, output_path='submission.csv'):
    """제출 파일 저장"""
    print(f"\n💾 제출 파일 저장: {output_path}")
    submission.to_csv(output_path, index=False)
    print(f"✅ 저장 완료: {submission.shape}")

    # 샘플 출력
    print("\n📊 샘플 예측 (처음 5개):")
    print(submission.head())

    # 통계
    print("\n📊 예측 통계:")
    print(submission[['end_x', 'end_y']].describe())

    return submission

def validate_submission(submission, sample_path='./data/sample_submission.csv'):
    """제출 파일 검증"""
    print("\n" + "=" * 80)
    print("  제출 파일 검증")
    print("=" * 80)

    # Sample submission 로딩
    sample = pd.read_csv(sample_path)

    # Shape 검증
    print(f"\n✓ Shape 검증:")
    print(f"  Sample: {sample.shape}")
    print(f"  Ours:   {submission.shape}")
    if submission.shape == sample.shape:
        print("  ✅ Shape 일치")
    else:
        print("  ⚠️  Shape 불일치!")
        return False

    # 컬럼 검증
    print(f"\n✓ 컬럼 검증:")
    if list(submission.columns) == list(sample.columns):
        print(f"  ✅ 컬럼 일치: {list(submission.columns)}")
    else:
        print(f"  ⚠️  컬럼 불일치!")
        print(f"  Expected: {list(sample.columns)}")
        print(f"  Got: {list(submission.columns)}")
        return False

    # game_episode 검증
    print(f"\n✓ game_episode 검증:")
    missing = set(sample['game_episode']) - set(submission['game_episode'])
    extra = set(submission['game_episode']) - set(sample['game_episode'])

    if len(missing) == 0 and len(extra) == 0:
        print(f"  ✅ 모든 에피소드 일치 ({len(submission)} 개)")
    else:
        print(f"  ⚠️  Missing: {len(missing)}, Extra: {len(extra)}")
        return False

    # 좌표 범위 검증
    print(f"\n✓ 좌표 범위 검증:")
    x_valid = (submission['end_x'] >= 0) & (submission['end_x'] <= 105)
    y_valid = (submission['end_y'] >= 0) & (submission['end_y'] <= 68)

    if x_valid.all() and y_valid.all():
        print(f"  ✅ 모든 좌표가 정상 범위 내")
    else:
        print(f"  ⚠️  범위 벗어난 좌표:")
        print(f"    X: {(~x_valid).sum()} 개")
        print(f"    Y: {(~y_valid).sum()} 개")
        return False

    print("\n" + "=" * 80)
    print("✅ 모든 검증 통과!")
    print("=" * 80)

    return True

def main():
    """메인 함수"""
    print("=" * 80)
    print("  K-League Pass Prediction - Inference")
    print("  추론 및 제출 파일 생성")
    print("=" * 80)

    # 1. 모델 로딩
    model_x, model_y, feature_cols = load_model('final_model.pkl')

    # 2. Preprocessor 로딩
    print("\n📂 Preprocessor 로딩 중...")
    preprocessor = DataPreprocessor(data_dir='./data')
    preprocessor.load_preprocessor('preprocessor.pkl')
    print("✅ Preprocessor 로딩 완료")

    # 3. 예측
    submission = predict_test(
        model_x, model_y, preprocessor, feature_cols,
        test_index_path='./data/test.csv',
        data_dir='./data'
    )

    # 4. 저장
    save_submission(submission, 'submission.csv')

    # 5. 검증
    validate_submission(submission, './data/sample_submission.csv')

    print("\n" + "=" * 80)
    print("🎉 Inference 완료!")
    print("=" * 80)
    print("""
✅ 생성된 파일: submission.csv
📊 예측 완료: {0} 에피소드

📋 제출 준비 완료:
   1. submission.csv 파일을 대회 사이트에 제출
   2. Public LB 점수 확인
   
🎯 예상 성능: 1.0 ~ 1.5m (Validation: 1.16m)
""".format(len(submission)))

    return submission

if __name__ == "__main__":
    submission = main()

