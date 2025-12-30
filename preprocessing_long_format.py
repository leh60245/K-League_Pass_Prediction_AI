"""
K-League Pass Prediction - Long Format 전처리 (LSTM 최적화)

🎯 핵심: Wide format → True Sequence format
- 각 episode를 (seq_len, features) 형태로 변환
- LSTM이 실제 시간 순서를 학습할 수 있도록 구조 변경

작성일: 2025-12-18
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import pickle
import warnings
warnings.filterwarnings('ignore')


class LongFormatPreprocessor:
    """
    LSTM을 위한 Long Format 전처리
    - 각 episode를 시퀀스로 변환
    - 동적 길이 처리 (패딩)
    - 실제 시간 순서 보존
    """

    def __init__(self, data_dir='./data', max_seq_len=20):
        self.data_dir = data_dir
        self.max_seq_len = max_seq_len

        # Encoders
        self.type_encoder = LabelEncoder()
        self.result_encoder = LabelEncoder()
        self.team_encoder = LabelEncoder()

        # Scalers
        self.coord_scaler = StandardScaler()  # 좌표용
        self.feature_scaler = StandardScaler()  # 기타 피처용

    def load_raw_data(self):
        """원본 이벤트 데이터 로딩"""
        import os

        # Train
        train_path = os.path.join(self.data_dir, 'train.csv')
        train_data = pd.read_csv(train_path)
        train_data['is_train'] = 1

        # Test
        test_index_path = os.path.join(self.data_dir, 'test_index.csv')
        test_index = pd.read_csv(test_index_path)

        test_events_list = []
        for _, row in test_index.iterrows():
            ep_path = os.path.join(self.data_dir, row['path'].replace('./', ''))
            df_ep = pd.read_csv(ep_path)
            test_events_list.append(df_ep)

        test_events = pd.concat(test_events_list, ignore_index=True)
        test_events['is_train'] = 0

        # 결합
        data = pd.concat([train_data, test_events], ignore_index=True)

        # 정렬
        data = data.sort_values(['game_episode', 'time_seconds', 'action_id']).reset_index(drop=True)

        print(f"✅ Raw 데이터 로딩:")
        print(f"   - Train: {len(train_data):,} events")
        print(f"   - Test: {len(test_events):,} events")
        print(f"   - Total Episodes: {data['game_episode'].nunique():,}")

        return data

    def create_event_features(self, data):
        """각 이벤트별 피처 생성"""
        print("\n🔧 이벤트별 피처 생성 중...")

        # 1. 시간 차이
        data['prev_time'] = data.groupby('game_episode')['time_seconds'].shift(1)
        data['dt'] = data['time_seconds'] - data['prev_time']
        data['dt'] = data['dt'].fillna(0.0)

        # 2. 이동 거리/방향
        data['dx'] = data['end_x'] - data['start_x']
        data['dy'] = data['end_y'] - data['start_y']
        data['dist'] = np.sqrt(data['dx']**2 + data['dy']**2)
        data['speed'] = data['dist'] / data['dt'].replace(0, 1e-3)

        # 3. 골문 방향
        goal_x, goal_y = 105, 34
        data['distance_to_goal'] = np.sqrt(
            (data['start_x'] - goal_x)**2 +
            (data['start_y'] - goal_y)**2
        )

        # 4. 진행 방향 (각도)
        data['direction'] = np.arctan2(data['dy'], data['dx'])

        # 5. 범주형 인코딩
        data['type_name'] = data['type_name'].fillna('__NA__')
        data['type_id'] = self.type_encoder.fit_transform(data['type_name'])

        data['result_name'] = data['result_name'].fillna('__NA__')
        data['result_id'] = self.result_encoder.fit_transform(data['result_name'])

        if data['team_id'].dtype == 'object':
            data['team_id_enc'] = self.team_encoder.fit_transform(data['team_id'])
        else:
            data['team_id_enc'] = data['team_id']

        print("✅ 이벤트 피처 생성 완료")
        return data

    def create_sequences(self, data):
        """
        에피소드별로 시퀀스 데이터 생성

        Returns:
            sequences: List of (seq_len, num_features) arrays
            targets: (N, 2) array of (target_x, target_y)
            metadata: episode 정보
        """
        print("\n📦 시퀀스 데이터 생성 중...")

        # 사용할 피처 선택
        numerical_features = [
            'start_x', 'start_y', 'end_x', 'end_y',
            'dx', 'dy', 'dist', 'speed', 'dt',
            'distance_to_goal', 'direction',
            'time_seconds'
        ]

        categorical_features = [
            'type_id', 'result_id', 'team_id_enc',
            'is_home', 'period_id'
        ]

        all_features = numerical_features + categorical_features

        sequences = []
        targets = []
        seq_lengths = []
        episodes = []
        is_train_flags = []

        for ep, group in data.groupby('game_episode'):
            group = group.sort_values(['time_seconds', 'action_id'])

            # 마지막 이벤트 제외 (target)
            if len(group) < 2:
                continue

            seq_data = group.iloc[:-1][all_features].values
            target_event = group.iloc[-1]

            # 마지막 K개만 사용
            if len(seq_data) > self.max_seq_len:
                seq_data = seq_data[-self.max_seq_len:]

            sequences.append(seq_data)
            seq_lengths.append(len(seq_data))

            # Target (마지막 이벤트의 end 위치)
            targets.append([target_event['end_x'], target_event['end_y']])

            episodes.append(ep)
            is_train_flags.append(target_event['is_train'])

        print(f"✅ 시퀀스 생성 완료:")
        print(f"   - Total Episodes: {len(sequences):,}")
        print(f"   - Avg Seq Length: {np.mean(seq_lengths):.1f}")
        print(f"   - Max Seq Length: {max(seq_lengths)}")
        print(f"   - Min Seq Length: {min(seq_lengths)}")

        return sequences, np.array(targets), seq_lengths, episodes, is_train_flags, \
               numerical_features, categorical_features

    def save_sequences(self, sequences, targets, seq_lengths, episodes, is_train_flags,
                       numerical_features, categorical_features):
        """시퀀스 데이터 저장"""
        print("\n💾 시퀀스 데이터 저장 중...")

        # Train/Test 분리
        train_mask = np.array(is_train_flags) == 1

        train_data = {
            'sequences': [seq for i, seq in enumerate(sequences) if train_mask[i]],
            'targets': targets[train_mask],
            'seq_lengths': [l for i, l in enumerate(seq_lengths) if train_mask[i]],
            'episodes': [e for i, e in enumerate(episodes) if train_mask[i]],
            'numerical_features': numerical_features,
            'categorical_features': categorical_features
        }

        test_data = {
            'sequences': [seq for i, seq in enumerate(sequences) if not train_mask[i]],
            'targets': targets[~train_mask],
            'seq_lengths': [l for i, l in enumerate(seq_lengths) if not train_mask[i]],
            'episodes': [e for i, e in enumerate(episodes) if not train_mask[i]],
            'numerical_features': numerical_features,
            'categorical_features': categorical_features
        }

        # 저장
        with open('train_sequences_long.pkl', 'wb') as f:
            pickle.dump(train_data, f)

        with open('test_sequences_long.pkl', 'wb') as f:
            pickle.dump(test_data, f)

        # Encoders 저장
        with open('encoders_long.pkl', 'wb') as f:
            pickle.dump({
                'type_encoder': self.type_encoder,
                'result_encoder': self.result_encoder,
                'team_encoder': self.team_encoder,
                'max_seq_len': self.max_seq_len
            }, f)

        print(f"✅ 저장 완료:")
        print(f"   - train_sequences_long.pkl (Train: {len(train_data['sequences']):,})")
        print(f"   - test_sequences_long.pkl (Test: {len(test_data['sequences']):,})")
        print(f"   - encoders_long.pkl")

    def run(self):
        """전체 파이프라인 실행"""
        print("=" * 80)
        print("  Long Format Preprocessing for LSTM")
        print("  진짜 시퀀스 데이터 생성")
        print("=" * 80)
        print()

        # 1. 원본 데이터 로딩
        data = self.load_raw_data()

        # 2. 이벤트 피처 생성
        data = self.create_event_features(data)

        # 3. 시퀀스 생성
        sequences, targets, seq_lengths, episodes, is_train_flags, \
        numerical_features, categorical_features = self.create_sequences(data)

        # 4. 저장
        self.save_sequences(sequences, targets, seq_lengths, episodes, is_train_flags,
                          numerical_features, categorical_features)

        print("\n" + "=" * 80)
        print("✅ Long Format 전처리 완료!")
        print("=" * 80)


if __name__ == "__main__":
    preprocessor = LongFormatPreprocessor(data_dir='./data', max_seq_len=20)
    preprocessor.run()

