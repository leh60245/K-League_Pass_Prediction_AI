"""
K-League Pass Prediction - 유연한 추론 스크립트

특징:
- 모델 경로 자동 감지 또는 지정 가능
- V1/V2/V2.1 모델 모두 지원
- 전처리 버전 자동 매칭
- 상세한 로깅 및 에러 핸들링
"""

import pandas as pd
import numpy as np
import pickle
import os
from tqdm import tqdm
import warnings
from datetime import datetime
warnings.filterwarnings('ignore')

class FlexibleInference:
    """유연한 추론 클래스"""

    def __init__(self, model_path=None, preprocessor_path=None, data_dir='./data'):
        """
        초기화

        Args:
            model_path: 모델 파일 경로 (None이면 자동 감지)
            preprocessor_path: 전처리기 경로 (None이면 자동 감지)
            data_dir: 데이터 디렉토리
        """
        self.data_dir = data_dir
        self.model_path = model_path
        self.preprocessor_path = preprocessor_path
        self.model_x = None
        self.model_y = None
        self.preprocessor = None
        self.feature_cols = None
        self.version = None

    def auto_detect_model(self):
        """모델 자동 감지"""
        print("🔍 모델 파일 자동 감지 중...")

        # 우선순위: V2.1 > V2 > V1 > 기본
        candidates = [
            ('lightgbm_model_v2.1.pkl', 'V2.1'),
            ('lightgbm_model_v2.pkl', 'V2'),
            ('lightgbm_model.pkl', 'V1'),
        ]

        for model_file, version in candidates:
            if os.path.exists(model_file):
                print(f"✅ 발견: {model_file} ({version})")
                self.version = version
                return model_file

        raise FileNotFoundError("❌ 모델 파일을 찾을 수 없습니다!")

    def auto_detect_preprocessor(self):
        """전처리기 자동 감지"""
        print("🔍 전처리기 자동 감지 중...")

        # 모델 버전에 맞는 전처리기 선택
        if self.version in ['V2', 'V2.1']:
            candidates = ['preprocessor_v2.pkl', 'preprocessor.pkl']
        else:
            candidates = ['preprocessor.pkl']

        for prep_file in candidates:
            if os.path.exists(prep_file):
                print(f"✅ 발견: {prep_file}")
                return prep_file

        raise FileNotFoundError("❌ 전처리기 파일을 찾을 수 없습니다!")

    def load_model(self):
        """모델 로딩"""
        if self.model_path is None:
            self.model_path = self.auto_detect_model()
        else:
            # 버전 추론
            if 'v2.1' in self.model_path.lower():
                self.version = 'V2.1'
            elif 'v2' in self.model_path.lower():
                self.version = 'V2'
            else:
                self.version = 'V1'

        print(f"\n📂 모델 로딩: {self.model_path}")

        with open(self.model_path, 'rb') as f:
            saved = pickle.load(f)

        self.model_x = saved['model_x']
        self.model_y = saved['model_y']

        # 피처 컬럼 정보 (있으면)
        if 'feature_cols' in saved:
            self.feature_cols = saved['feature_cols']
            print(f"✅ 모델 로딩 완료 ({self.version}, {len(self.feature_cols)}개 피처)")
        else:
            print(f"✅ 모델 로딩 완료 ({self.version})")

    def load_preprocessor(self):
        """전처리기 로딩"""
        if self.preprocessor_path is None:
            self.preprocessor_path = self.auto_detect_preprocessor()

        print(f"\n🔧 전처리기 로딩: {self.preprocessor_path}")

        # 버전에 맞는 전처리기 클래스 선택
        if 'v2' in self.preprocessor_path.lower():
            from preprocessing_v2 import DataPreprocessorV2
            self.preprocessor = DataPreprocessorV2(data_dir=self.data_dir)
        else:
            from preprocessing import DataPreprocessor
            self.preprocessor = DataPreprocessor(data_dir=self.data_dir)

        self.preprocessor.load_preprocessor(self.preprocessor_path)

        # 피처 컬럼 가져오기 (모델에 없으면)
        if self.feature_cols is None:
            self.feature_cols = self.preprocessor.get_feature_columns()
            print(f"✅ 전처리기에서 피처 정보 로딩: {len(self.feature_cols)}개")
        else:
            print("✅ 전처리기 로딩 완료")

    def preprocess_episode(self, episode_data):
        """에피소드 전처리"""
        try:
            # 시간 정렬
            episode_data = episode_data.sort_values('time_seconds').reset_index(drop=True)

            # 기본 피처
            episode_data = self.preprocessor.create_basic_features(episode_data, verbose=False)

            # 버전별 추가 피처
            if self.version in ['V2', 'V2.1']:
                # V2 전용 피처
                if hasattr(self.preprocessor, 'create_nonlinear_features'):
                    episode_data = self.preprocessor.create_nonlinear_features(episode_data, verbose=False)
                if hasattr(self.preprocessor, 'create_position_specific_features'):
                    episode_data = self.preprocessor.create_position_specific_features(episode_data, verbose=False)

            # 시퀀스 피처
            episode_data = self.preprocessor.create_sequence_features(episode_data, verbose=False)

            # 직전 이벤트
            episode_data = self.preprocessor.create_previous_event_features(episode_data, verbose=False)

            # 고급 전술 피처
            episode_data = self.preprocessor.create_advanced_tactical_features(episode_data, verbose=False)

            # V2 컨텍스트 피처
            if self.version in ['V2', 'V2.1']:
                if hasattr(self.preprocessor, 'create_context_features'):
                    episode_data = self.preprocessor.create_context_features(episode_data, verbose=False)

            # 마지막 이벤트
            last_event = episode_data.iloc[-1:].copy()

            # 인코딩
            last_event = self.preprocessor.encode_categorical(last_event, fit=False, verbose=False)

            # 결측치 처리
            last_event = self.preprocessor.fill_missing(last_event, verbose=False)

            return last_event

        except Exception as e:
            print(f"⚠️  전처리 에러: {e}")
            return None

    def predict(self, X):
        """예측"""
        try:
            pred_x = self.model_x.predict(X)[0]
            pred_y = self.model_y.predict(X)[0]

            # 좌표 범위 제한
            pred_x = np.clip(pred_x, 0, 105)
            pred_y = np.clip(pred_y, 0, 68)

            return pred_x, pred_y

        except Exception as e:
            print(f"⚠️  예측 에러: {e}")
            return None, None

    def run_inference(self, output_path=None, test_index_path=None):
        """전체 추론 실행"""

        print("=" * 80)
        print("  K-League Pass Prediction - 유연한 추론 시스템")
        print("=" * 80)
        print()

        # 1. 모델 로딩
        self.load_model()

        # 2. 전처리기 로딩
        self.load_preprocessor()

        # 3. Test 인덱스 로딩
        if test_index_path is None:
            test_index_path = os.path.join(self.data_dir, 'test.csv')

        print(f"\n📂 Test 인덱스 로딩: {test_index_path}")
        test_index = pd.read_csv(test_index_path)
        print(f"✅ Test 에피소드 수: {len(test_index):,}")

        # 4. 출력 경로 설정
        if output_path is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_path = f'submission_{self.version.lower()}_{timestamp}.csv'

        print(f"\n📝 출력 파일: {output_path}")

        # 5. 예측 진행
        print("\n🔄 예측 진행 중...")
        predictions = []
        success_count = 0
        error_count = 0

        for idx, row in tqdm(test_index.iterrows(), total=len(test_index), desc="Predicting"):
            game_episode = row['game_episode']

            try:
                # 파일 경로 (test.csv 또는 test_index.csv 형식 모두 지원)
                if 'path' in row:
                    file_path = os.path.join(self.data_dir, row['path'].replace('./', ''))
                else:
                    # game_episode에서 game_id 추출
                    game_id = game_episode.split('_')[0]
                    episode_num = game_episode.split('_')[1]
                    file_path = os.path.join(self.data_dir, 'test', game_id, f'{game_id}_{episode_num}.csv')

                # 데이터 로딩
                episode_data = pd.read_csv(file_path)

                # 전처리
                last_event = self.preprocess_episode(episode_data)

                if last_event is None:
                    raise ValueError("전처리 실패")

                # 피처 추출
                available_features = [col for col in self.feature_cols if col in last_event.columns]

                if len(available_features) < len(self.feature_cols) * 0.8:  # 80% 미만이면 경고
                    missing = set(self.feature_cols) - set(available_features)
                    if error_count == 0:  # 첫 에러만 출력
                        print(f"\n⚠️  누락된 피처 ({len(missing)}개): {list(missing)[:5]}...")

                X = last_event[available_features].values

                # 예측
                pred_x, pred_y = self.predict(X)

                if pred_x is None or pred_y is None:
                    raise ValueError("예측 실패")

                predictions.append({
                    'game_episode': game_episode,
                    'end_x': pred_x,
                    'end_y': pred_y
                })

                success_count += 1

            except Exception as e:
                error_count += 1
                if error_count <= 3:  # 처음 3개 에러만 출력
                    print(f"\n⚠️  에러 (Episode {game_episode}): {str(e)[:100]}")

                # 에러 시 중앙값 예측
                predictions.append({
                    'game_episode': game_episode,
                    'end_x': 68.45,  # train 평균
                    'end_y': 33.62
                })

        print(f"\n✅ 예측 완료! (성공: {success_count}, 실패: {error_count})")

        # 6. 제출 파일 생성
        print("\n📝 제출 파일 생성 중...")
        submission = pd.DataFrame(predictions)
        submission = submission.sort_values('game_episode').reset_index(drop=True)
        submission.to_csv(output_path, index=False)
        print(f"✅ 제출 파일 저장: {output_path}")

        # 7. 결과 요약
        self.print_summary(submission, output_path)

        return submission

    def print_summary(self, submission, output_path):
        """결과 요약 출력"""
        print("\n" + "=" * 80)
        print("  추론 결과 요약")
        print("=" * 80)

        print("\n📊 제출 파일 미리보기:")
        print(submission.head(10).to_string(index=False))

        print("\n📊 통계:")
        print(f"  - 모델 버전: {self.version}")
        print(f"  - 총 예측 수: {len(submission):,}")
        print(f"  - end_x 범위: [{submission['end_x'].min():.2f}, {submission['end_x'].max():.2f}]")
        print(f"  - end_y 범위: [{submission['end_y'].min():.2f}, {submission['end_y'].max():.2f}]")
        print(f"  - end_x 평균: {submission['end_x'].mean():.2f} (train: 68.45)")
        print(f"  - end_y 평균: {submission['end_y'].mean():.2f} (train: 33.62)")

        # 분포 체크
        print("\n📊 X축 분포:")
        x_dist = pd.cut(submission['end_x'], bins=[0, 35, 70, 105], labels=['수비진', '중원', '공격진'])
        print(x_dist.value_counts(normalize=True).sort_index().to_string())

        print("\n📊 Y축 분포:")
        y_dist = pd.cut(submission['end_y'], bins=[0, 22.67, 45.33, 68], labels=['좌측', '중앙', '우측'])
        print(y_dist.value_counts(normalize=True).sort_index().to_string())

        print("\n" + "=" * 80)
        print("✅ 추론 완료!")
        print("=" * 80)
        print(f"\n📤 제출 파일: {output_path}")
        print("📤 이 파일을 대회 시스템에 제출하세요!")
        print("\n💡 다른 모델로 추론하려면:")
        print("   python flexible_inference.py --model lightgbm_model_v2.pkl")


def main():
    """메인 함수"""
    import argparse

    parser = argparse.ArgumentParser(description='K-League 패스 예측 추론')
    parser.add_argument('--model', type=str, default=None, help='모델 파일 경로 (기본: 자동 감지)')
    parser.add_argument('--preprocessor', type=str, default=None, help='전처리기 경로 (기본: 자동 감지)')
    parser.add_argument('--data-dir', type=str, default='./data', help='데이터 디렉토리')
    parser.add_argument('--output', type=str, default=None, help='출력 파일 경로 (기본: 자동 생성)')
    parser.add_argument('--test-index', type=str, default=None, help='Test 인덱스 파일')

    args = parser.parse_args()

    # 추론 실행
    inference = FlexibleInference(
        model_path=args.model,
        preprocessor_path=args.preprocessor,
        data_dir=args.data_dir
    )

    submission = inference.run_inference(
        output_path=args.output,
        test_index_path=args.test_index
    )

    return submission


if __name__ == "__main__":
    # 커맨드라인 인자가 있으면 파싱, 없으면 기본 실행
    import sys
    if len(sys.argv) > 1:
        submission = main()
    else:
        # 기본 실행 (자동 감지)
        print("💡 기본 모드: 모델 자동 감지")
        print("💡 옵션 사용: python flexible_inference.py --help\n")

        inference = FlexibleInference(data_dir='./data')
        submission = inference.run_inference()

