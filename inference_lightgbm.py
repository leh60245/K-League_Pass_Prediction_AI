"""
K-League Pass Prediction - LightGBM 추론 스크립트

학습된 LightGBM 모델로 테스트 데이터 예측 및 제출 파일 생성
"""

import pandas as pd
import numpy as np
import pickle
import os
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

from preprocessing import DataPreprocessor
from feature_config import FeatureConfig

def load_lightgbm_model(model_path='lightgbm_model.pkl'):
    """LightGBM 모델 로딩"""
    print(f"📂 모델 로딩 중: {model_path}")
    with open(model_path, 'rb') as f:
        saved = pickle.load(f)
    print("✅ 모델 로딩 완료")
    return saved['model_x'], saved['model_y']

def preprocess_test_episode(episode_data, preprocessor):
    """테스트 에피소드 전처리"""
    # 시간 정렬
    episode_data = episode_data.sort_values('time_seconds').reset_index(drop=True)

    # 기본 피처 생성
    episode_data = preprocessor.create_basic_features(episode_data, verbose=False)

    # 시퀀스 피처 생성
    episode_data = preprocessor.create_sequence_features(episode_data, verbose=False)

    # 직전 이벤트 피처
    episode_data = preprocessor.create_previous_event_features(episode_data, verbose=False)

    # 고급 전술 피처 생성
    episode_data = preprocessor.create_advanced_tactical_features(episode_data, verbose=False)

    # 마지막 이벤트 추출
    last_event = episode_data.iloc[-1:].copy()

    # 범주형 인코딩
    last_event = preprocessor.encode_categorical(last_event, fit=False, verbose=False)

    # 결측치 처리
    last_event = last_event.fillna(0)

    return last_event

def predict_test_lightgbm(output_path='submission_lightgbm.csv'):
    """LightGBM 모델로 테스트 데이터 예측"""

    print("=" * 80)
    print("  K-League Pass Prediction - LightGBM 추론")
    print("=" * 80)
    print()

    # 1. 모델 로딩
    model_x, model_y = load_lightgbm_model('lightgbm_model.pkl')

    # 2. Preprocessor 로딩
    print("\n🔧 Preprocessor 로딩...")
    preprocessor = DataPreprocessor(data_dir='./data')
    preprocessor.load_preprocessor('preprocessor.pkl')
    print("✅ Preprocessor 로딩 완료")

    # 3. 피처 설정 로딩
    print("\n📊 피처 설정 로딩...")
    config = FeatureConfig('feature_config.json')
    feature_cols = config.get_feature_columns()
    print(f"✅ 피처 개수: {len(feature_cols)}")

    # 4. Test 인덱스 로딩
    print("\n📂 Test 인덱스 로딩...")
    test_index_path = './data/test.csv'
    test_index = pd.read_csv(test_index_path)
    print(f"✅ Test 에피소드 수: {len(test_index):,}")

    # 5. 예측
    print("\n🔄 예측 진행 중...")
    predictions = []

    for idx, row in tqdm(test_index.iterrows(), total=len(test_index), desc="Predicting"):
        try:
            game_episode = row['game_episode']
            file_path = os.path.join('./data', row['path'].replace('./', ''))

            # 에피소드 데이터 로딩
            episode_data = pd.read_csv(file_path)

            # 전처리
            last_event = preprocess_test_episode(episode_data, preprocessor)

            # 피처 추출 (존재하는 피처만)
            available_features = [col for col in feature_cols if col in last_event.columns]
            X = last_event[available_features].values

            # 예측
            pred_x = model_x.predict(X)[0]
            pred_y = model_y.predict(X)[0]

            # 좌표 범위 제한 (105x68 그리드)
            pred_x = np.clip(pred_x, 0, 105)
            pred_y = np.clip(pred_y, 0, 68)

            predictions.append({
                'game_episode': game_episode,
                'end_x': pred_x,
                'end_y': pred_y
            })

        except Exception as e:
            print(f"\n⚠️  에러 발생 (Episode {game_episode}): {e}")
            # 에러 발생 시 중앙값 예측
            predictions.append({
                'game_episode': game_episode,
                'end_x': 52.5,
                'end_y': 34.0
            })

    print("\n✅ 예측 완료!")

    # 6. 제출 파일 생성
    print("\n📝 제출 파일 생성 중...")
    submission = pd.DataFrame(predictions)

    # game_episode 순서대로 정렬
    submission = submission.sort_values('game_episode').reset_index(drop=True)

    # 저장
    submission.to_csv(output_path, index=False)
    print(f"✅ 제출 파일 저장: {output_path}")

    # 7. 결과 미리보기
    print("\n" + "=" * 80)
    print("  제출 파일 미리보기")
    print("=" * 80)
    print(submission.head(10))

    print("\n📊 통계:")
    print(f"  - 총 예측 수: {len(submission):,}")
    print(f"  - end_x 범위: [{submission['end_x'].min():.2f}, {submission['end_x'].max():.2f}]")
    print(f"  - end_y 범위: [{submission['end_y'].min():.2f}, {submission['end_y'].max():.2f}]")
    print(f"  - end_x 평균: {submission['end_x'].mean():.2f}")
    print(f"  - end_y 평균: {submission['end_y'].mean():.2f}")

    print("\n" + "=" * 80)
    print("✅ 추론 완료!")
    print("=" * 80)
    print(f"\n📤 제출 파일: {output_path}")
    print("📤 이 파일을 대회 시스템에 제출하세요!")

    return submission

if __name__ == "__main__":
    # LightGBM 모델로 예측
    submission = predict_test_lightgbm(output_path='submission_lightgbm.csv')

