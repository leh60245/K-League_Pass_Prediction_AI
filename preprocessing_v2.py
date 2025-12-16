"""
K-League Pass Prediction - Data Preprocessing Pipeline V2

개선 사항 (2025-12-16):
1. 다중공선성 피처 제거 (6개)
2. 비선형 변환 피처 추가 (8개)
3. 위치별 특화 피처 추가 (6개)
4. 결측치 처리 개선
5. 선수/팀 컨텍스트 피처 추가 (4개)

목표: EDA Phase 4 인사이트 반영하여 예측 성능 향상
"""

import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import GroupKFold
from scipy.spatial.distance import cdist
import pickle
import warnings
warnings.filterwarnings('ignore')

class DataPreprocessorV2:
    def __init__(self, data_dir='./data'):
        self.data_dir = data_dir
        self.scaler_x = StandardScaler()
        self.scaler_y = StandardScaler()
        self.type_encoder = LabelEncoder()
        self.result_encoder = LabelEncoder()

        # 선수/팀 통계 저장
        self.player_stats = None
        self.team_stats = None

    def load_data(self, verbose=True):
        """데이터 로딩"""
        if verbose:
            print("📊 데이터 로딩 중...")

        # Train 데이터
        train_path = os.path.join(self.data_dir, 'train.csv')
        train_data = pd.read_csv(train_path)

        # Match info
        match_info_path = os.path.join(self.data_dir, 'match_info.csv')
        match_info = pd.read_csv(match_info_path)

        if verbose:
            print(f"✅ Train: {len(train_data):,} 이벤트, {train_data['game_episode'].nunique():,} 에피소드")
            print(f"✅ Match Info: {len(match_info)} 경기\n")

        return train_data, match_info

    def sort_by_time(self, data, verbose=True):
        """시간 순서로 정렬"""
        if verbose:
            print("⏰ 시간 순서로 정렬 중...")

        data = data.sort_values(['game_episode', 'time_seconds']).reset_index(drop=True)

        if verbose:
            print("✅ 정렬 완료\n")

        return data

    def create_basic_features(self, data, verbose=True):
        """기본 피처 생성"""
        if verbose:
            print("🔧 기본 피처 생성 중...")

        # 1. 이동 거리/방향
        data['delta_x'] = data['end_x'] - data['start_x']
        data['delta_y'] = data['end_y'] - data['start_y']
        data['distance'] = np.sqrt(data['delta_x']**2 + data['delta_y']**2)

        # 2. 골문 거리 (오른쪽 골문 기준: 105, 34)
        goal_x, goal_y = 105, 34
        data['distance_to_goal_start'] = np.sqrt(
            (data['start_x'] - goal_x)**2 +
            (data['start_y'] - goal_y)**2
        )
        data['distance_to_goal_end'] = np.sqrt(
            (data['end_x'] - goal_x)**2 +
            (data['end_y'] - goal_y)**2
        )

        # 2-1. 골문 진행도 (골문으로 가까워지는 정도)
        data['goal_approach'] = data['distance_to_goal_start'] - data['distance_to_goal_end']

        # 3. 경기장 영역 (3등분) - start_x_zone_fine은 제거 (다중공선성)
        data['start_x_zone'] = pd.cut(data['start_x'], bins=[0, 35, 70, 105], labels=[0, 1, 2])
        data['start_y_zone'] = pd.cut(data['start_y'], bins=[0, 22.67, 45.33, 68], labels=[0, 1, 2])

        # 3-2. 위험 지역 플래그 (페널티 박스: x > 87.5, 22.9 < y < 45.1)
        data['in_penalty_area'] = ((data['start_x'] > 87.5) &
                                   (data['start_y'] > 22.9) &
                                   (data['start_y'] < 45.1)).astype(int)

        # 3-3. 최종 1/3 지역 (Final Third)
        data['in_final_third'] = (data['start_x'] > 70).astype(int)

        # 4. 에피소드 내 순서
        data['event_order'] = data.groupby('game_episode').cumcount()

        # 4-1. 에피소드 첫 이벤트 플래그 (결측치 처리용)
        data['is_first_event'] = (data['event_order'] == 0).astype(int)

        # 5. 골 각도 (Shooting Angle)
        post_left_y = 30.34
        post_right_y = 37.66

        vec_left_x = goal_x - data['start_x']
        vec_left_y = post_left_y - data['start_y']
        vec_right_x = goal_x - data['start_x']
        vec_right_y = post_right_y - data['start_y']

        dot_product = vec_left_x * vec_right_x + vec_left_y * vec_right_y
        cross_product = vec_left_x * vec_right_y - vec_left_y * vec_right_x

        data['shooting_angle'] = np.abs(np.arctan2(cross_product, dot_product))

        # 6. 정규화된 좌표는 제거 (다중공선성) - 원본만 사용

        if verbose:
            print(f"✅ 기본 피처 생성 완료\n")

        return data

    def create_nonlinear_features(self, data, verbose=True):
        """🔥 NEW: 비선형 변환 피처 생성 (EDA Phase 4 인사이트)"""
        if verbose:
            print("🔥 비선형 변환 피처 생성 중...")

        # 1. 골문 거리 역수 (가까울수록 큰 가중치)
        data['distance_to_goal_inv'] = 1 / (data['distance_to_goal_start'] + 1)

        # 2. 골문 거리 제곱근 (비선형 패턴 포착)
        data['distance_to_goal_sqrt'] = np.sqrt(data['distance_to_goal_start'])

        # 3. 각도의 삼각함수 (주기성 포착)
        data['shooting_angle_sin'] = np.sin(data['shooting_angle'])
        data['shooting_angle_cos'] = np.cos(data['shooting_angle'])

        # 4. 위치의 제곱 (비선형 관계)
        data['start_x_squared'] = data['start_x'] ** 2
        data['start_y_squared'] = data['start_y'] ** 2

        # 5. 상호작용 피처
        data['x_y_interaction'] = data['start_x'] * data['start_y']
        data['goal_dist_angle_interaction'] = data['distance_to_goal_start'] * data['shooting_angle']

        if verbose:
            print(f"✅ 비선형 피처 8개 생성 완료\n")

        return data

    def create_position_specific_features(self, data, verbose=True):
        """🎯 NEW: 위치별 특화 피처 (EDA Phase 4 인사이트)"""
        if verbose:
            print("🎯 위치별 특화 피처 생성 중...")

        # 1. 수비진 특화 (불확실성 높은 구간)
        data['is_defensive_third'] = (data['start_x'] < 35).astype(int)

        # 2. 공격진 특화 - 골 긴급도 (exponential decay)
        data['goal_urgency'] = np.exp(-data['distance_to_goal_start'] / 20)

        # 3. Y축 중앙 복도 (예측 어려운 구간)
        data['is_central_corridor'] = ((data['start_y'] > 20) &
                                       (data['start_y'] < 48)).astype(int)

        # 4. 골문 근접도 (페널티 박스 근처)
        data['near_goal_zone'] = ((data['distance_to_goal_start'] < 25) &
                                  (data['start_x'] > 80)).astype(int)

        # 5. 사이드 공격 여부
        data['is_wing_attack'] = ((data['start_x'] > 70) &
                                  ((data['start_y'] < 15) | (data['start_y'] > 53))).astype(int)

        # 6. 중원 지배 영역
        data['is_midfield_control'] = ((data['start_x'] >= 35) &
                                       (data['start_x'] <= 70) &
                                       (data['start_y'] >= 20) &
                                       (data['start_y'] <= 48)).astype(int)

        if verbose:
            print(f"✅ 위치 특화 피처 6개 생성 완료\n")

        return data

    def create_sequence_features(self, data, verbose=True):
        """시퀀스 피처 생성 (각 에피소드별)"""
        if verbose:
            print("🔄 시퀀스 피처 생성 중...")

        episode_features = []

        for episode_id, group in data.groupby('game_episode'):
            group = group.copy()

            # 1. 에피소드 길이
            episode_length = len(group)
            group['episode_length'] = episode_length

            # 2. X축 누적 진행도
            first_x = group.iloc[0]['start_x']
            last_x = group.iloc[-1]['end_x']
            group['x_progression'] = group['start_x'] - first_x
            group['x_total_progression'] = last_x - first_x

            # 3. 상대 시간
            start_time = group.iloc[0]['time_seconds']
            group['relative_time'] = group['time_seconds'] - start_time

            # 4. 템포
            if episode_length > 1:
                duration = group.iloc[-1]['time_seconds'] - group.iloc[0]['time_seconds']
                tempo = duration / episode_length if episode_length > 0 else 0
            else:
                tempo = 0
            group['tempo'] = tempo

            # 5. 속도 계산 (개선: 에피소드 평균으로 결측치 대체)
            time_diff = group['time_seconds'].diff()
            group['velocity_x'] = group['start_x'].diff() / time_diff.replace(0, np.nan)
            group['velocity_y'] = group['start_y'].diff() / time_diff.replace(0, np.nan)
            group['velocity'] = np.sqrt(group['velocity_x']**2 + group['velocity_y']**2)

            # 6. 가속도
            group['acceleration'] = group['velocity'].diff() / time_diff.replace(0, np.nan)

            # 7. 템포 변화율
            group['tempo_change'] = tempo - group['relative_time'].rolling(2).mean().fillna(0)

            # 8. 진행 방향 일관성
            prev_delta_x = group['delta_x'].shift(1)
            prev_delta_y = group['delta_y'].shift(1)

            dot_prod = group['delta_x'] * prev_delta_x + group['delta_y'] * prev_delta_y
            mag_curr = np.sqrt(group['delta_x']**2 + group['delta_y']**2)
            mag_prev = np.sqrt(prev_delta_x**2 + prev_delta_y**2)
            group['direction_consistency'] = dot_prod / (mag_curr * mag_prev + 1e-10)

            # 9. 최종 1/3 지역 체류 시간 비율
            if 'in_final_third' in group.columns:
                final_third_time = group[group['in_final_third'] == 1]['relative_time'].sum()
                total_time = group['relative_time'].iloc[-1] if len(group) > 0 else 1
                group['final_third_time_ratio'] = final_third_time / (total_time + 1e-10)

            # 10. 수평/수직 진행도 비율
            total_horizontal = group['delta_x'].abs().sum()
            total_vertical = group['delta_y'].abs().sum()
            group['horizontal_vertical_ratio'] = total_horizontal / (total_vertical + 1e-10)

            episode_features.append(group)

        data = pd.concat(episode_features, ignore_index=True)

        if verbose:
            print(f"✅ 시퀀스 피처 생성 완료\n")

        return data

    def create_previous_event_features(self, data, verbose=True):
        """직전 이벤트 피처 생성"""
        if verbose:
            print("⬅️  직전 이벤트 피처 생성 중...")

        episode_features = []

        for episode_id, group in data.groupby('game_episode'):
            group = group.copy()

            # 직전 이벤트 정보
            group['prev_type_name'] = group['type_name'].shift(1)
            group['prev_result_name'] = group['result_name'].shift(1)
            group['prev_start_x'] = group['start_x'].shift(1)
            group['prev_start_y'] = group['start_y'].shift(1)
            group['prev_end_x'] = group['end_x'].shift(1)
            group['prev_end_y'] = group['end_y'].shift(1)

            # 직전 2개 이벤트
            group['prev2_type_name'] = group['type_name'].shift(2)

            episode_features.append(group)

        data = pd.concat(episode_features, ignore_index=True)

        if verbose:
            print(f"✅ 직전 이벤트 피처 생성 완료\n")

        return data

    def create_advanced_tactical_features(self, data, verbose=True):
        """고급 전술 피처 생성"""
        if verbose:
            print("⚽ 고급 전술 피처 생성 중...")

        episode_features = []

        for episode_id, group in data.groupby('game_episode'):
            group = group.copy()

            # 1. 압박 강도
            if len(group) > 1:
                x_range = group['start_x'].max() - group['start_x'].min() + 1
                y_range = group['start_y'].max() - group['start_y'].min() + 1
                area = x_range * y_range
                event_density = len(group) / area
            else:
                event_density = 0
            group['event_density'] = event_density

            pressure_radius = 10

            if len(group) > 50:
                group['local_pressure'] = event_density * 10
                group['weighted_pressure'] = event_density * 5
            else:
                positions = group[['start_x', 'start_y']].values
                dist_matrix = cdist(positions, positions, metric='euclidean')

                nearby_mask = (dist_matrix <= pressure_radius) & (dist_matrix > 0)
                group['local_pressure'] = nearby_mask.sum(axis=1)

                weights_matrix = 1 / (dist_matrix + 1)
                np.fill_diagonal(weights_matrix, 0)
                group['weighted_pressure'] = weights_matrix.sum(axis=1)

            # 2. 공간 창출
            prev_distance = group['distance'].shift(1)
            group['distance_change_rate'] = (group['distance'] - prev_distance) / (prev_distance + 1e-10)

            group['vertical_spread'] = group['start_y'].rolling(window=3, min_periods=1).std()

            group['attack_width'] = group['start_y'].rolling(window=5, min_periods=1).apply(
                lambda x: x.max() - x.min(), raw=True
            )

            # 3. 전술적 벡터
            group['forward_momentum'] = group['delta_x'].rolling(window=3, min_periods=1).sum()

            current_angle = np.arctan2(group['delta_y'], group['delta_x'])
            prev_angle = current_angle.shift(1)
            angle_change = np.abs(current_angle - prev_angle)
            angle_change = np.minimum(angle_change, 2*np.pi - angle_change)
            group['pass_angle_change'] = angle_change

            # 4. 히스토리 기반
            if 'velocity' in group.columns:
                group['avg_velocity_3'] = group['velocity'].rolling(window=3, min_periods=1).mean()

            if 'distance_to_goal_end' in group.columns:
                group['goal_approach_trend'] = group['distance_to_goal_end'].rolling(
                    window=3, min_periods=1
                ).apply(lambda x: x.iloc[0] - x.iloc[-1] if len(x) > 1 else 0, raw=False)

            # 5. 최적 경로
            if len(group) > 1:
                first_goal_dist = group.iloc[0]['distance_to_goal_start']
                last_goal_dist = group.iloc[-1]['distance_to_goal_end']
                actual_distance = group['distance'].sum()

                direct_progress = first_goal_dist - last_goal_dist
                path_efficiency = direct_progress / (actual_distance + 1e-10)
            else:
                path_efficiency = 0

            group['path_efficiency'] = path_efficiency

            # 6. 팀 중심점 대비 위치
            team_center_x = group['start_x'].mean()
            team_center_y = group['start_y'].mean()

            group['dist_from_team_center'] = np.sqrt(
                (group['start_x'] - team_center_x)**2 +
                (group['start_y'] - team_center_y)**2
            )

            # 7. 경기 페이즈
            if 'time_seconds' in group.columns:
                max_time = 5400
                group['match_phase'] = pd.cut(
                    group['time_seconds'],
                    bins=[0, 1800, 3600, max_time],
                    labels=[0, 1, 2]
                ).astype(float)

            episode_features.append(group)

        data = pd.concat(episode_features, ignore_index=True)

        if verbose:
            print(f"✅ 고급 전술 피처 생성 완료\n")

        return data

    def create_context_features(self, data, verbose=True):
        """💎 NEW: 선수/팀 컨텍스트 피처 (EDA Phase 4 인사이트)"""
        if verbose:
            print("💎 선수/팀 컨텍스트 피처 생성 중...")

        # 1. 선수 통계 계산 (전체 데이터 기준)
        if self.player_stats is None:
            self.player_stats = data.groupby('player_id').agg({
                'start_x': 'mean',
                'start_y': 'mean',
                'distance': 'mean',
                'velocity': 'mean'
            }).rename(columns={
                'start_x': 'player_avg_x',
                'start_y': 'player_avg_y',
                'distance': 'player_avg_pass_dist',
                'velocity': 'player_avg_velocity'
            })

        # 선수 통계 병합
        data = data.merge(self.player_stats, left_on='player_id', right_index=True, how='left')

        # 2. 팀 통계 계산
        if self.team_stats is None:
            self.team_stats = data.groupby('team_id').agg({
                'x_total_progression': 'mean',
                'episode_length': 'mean',
                'tempo': 'mean'
            }).rename(columns={
                'x_total_progression': 'team_aggression',
                'episode_length': 'team_avg_episode_length',
                'tempo': 'team_avg_tempo'
            })

        # 팀 통계 병합
        data = data.merge(self.team_stats, left_on='team_id', right_index=True, how='left')

        # 3. 시간 압박 (경기 종료 임박)
        max_time_by_period = {1: 2700, 2: 2700}  # 45분씩
        data['time_pressure'] = data.apply(
            lambda row: np.maximum(0, (max_time_by_period.get(row['period_id'], 2700) - row['time_seconds']) / 2700),
            axis=1
        )

        # 4. 선수 위치 이탈도 (평소와 다른 위치)
        data['player_position_deviation'] = np.sqrt(
            (data['start_x'] - data['player_avg_x'])**2 +
            (data['start_y'] - data['player_avg_y'])**2
        )

        if verbose:
            print(f"✅ 컨텍스트 피처 8개 생성 완료\n")

        return data

    def extract_last_events(self, data, verbose=True):
        """각 에피소드의 마지막 이벤트만 추출"""
        if verbose:
            print("🎯 마지막 이벤트 추출 중...")

        last_events = data.groupby('game_episode').tail(1).copy()

        if verbose:
            print(f"✅ {len(last_events):,}개 에피소드의 마지막 이벤트 추출 완료\n")

        return last_events

    def encode_categorical(self, data, fit=True, verbose=True):
        """범주형 변수 인코딩"""
        if verbose:
            print("🔤 범주형 변수 인코딩 중...")

        # type_name 인코딩 - 하지만 모두 Pass이므로 제거 가능
        # 일단 호환성을 위해 유지
        data['type_name'] = data['type_name'].fillna('Unknown')
        if fit:
            data['type_name_encoded'] = self.type_encoder.fit_transform(data['type_name'])
        else:
            data['type_name_encoded'] = self.type_encoder.transform(data['type_name'])

        if 'prev_type_name' in data.columns:
            data['prev_type_name'] = data['prev_type_name'].fillna('Unknown')
            data['prev_type_name_encoded'] = self.type_encoder.transform(data['prev_type_name'])

        if 'prev2_type_name' in data.columns:
            data['prev2_type_name'] = data['prev2_type_name'].fillna('Unknown')
            data['prev2_type_name_encoded'] = self.type_encoder.transform(data['prev2_type_name'])

        # result_name 인코딩
        if 'result_name' in data.columns and len(self.result_encoder.classes_) > 0:
            data['result_name'] = data['result_name'].fillna('Unknown')
            if fit:
                data['result_name_encoded'] = self.result_encoder.fit_transform(data['result_name'])
            else:
                data['result_name_encoded'] = self.result_encoder.transform(data['result_name'])

            if 'prev_result_name' in data.columns:
                data['prev_result_name'] = data['prev_result_name'].fillna('Unknown')
                data['prev_result_name_encoded'] = self.result_encoder.transform(data['prev_result_name'])

        if verbose:
            print(f"✅ 인코딩 완료\n")

        return data

    def fill_missing(self, data, verbose=True):
        """결측치 처리 (개선: 에피소드 평균 활용)"""
        if verbose:
            print("🔧 결측치 처리 중...")

        # 속도/가속도 피처의 결측치를 에피소드 평균으로 대체
        for col in ['velocity', 'velocity_x', 'velocity_y', 'acceleration']:
            if col in data.columns:
                # 에피소드별 평균 계산
                episode_mean = data.groupby('game_episode')[col].transform('mean')
                # 결측치를 평균으로 대체
                data[col] = data[col].fillna(episode_mean)
                # 여전히 NaN이면 0으로 (에피소드 전체가 NaN인 경우)
                data[col] = data[col].fillna(0)

        # 직전 이벤트 피처
        prev_cols = [col for col in data.columns if col.startswith('prev_')]

        for col in prev_cols:
            if col.endswith('_encoded'):
                data[col] = data[col].fillna(-1)
            elif data[col].dtype in ['float64', 'int64']:
                data[col] = data[col].fillna(0)
            else:
                data[col] = data[col].fillna('Unknown')

        # zone 피처
        if 'start_x_zone' in data.columns:
            data['start_x_zone'] = data['start_x_zone'].astype(float).fillna(-1)
            data['start_y_zone'] = data['start_y_zone'].astype(float).fillna(-1)

        # 기타 수치형 피처
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if data[col].isna().sum() > 0:
                data[col] = data[col].fillna(0)

        if verbose:
            print(f"✅ 결측치 처리 완료\n")

        return data

    def create_train_val_split(self, data, n_splits=5, random_state=42, verbose=True):
        """Game-based K-Fold Split"""
        if verbose:
            print(f"📊 {n_splits}-Fold Game-based Split 생성 중...")

        gkf = GroupKFold(n_splits=n_splits)

        splits = []
        for fold, (train_idx, val_idx) in enumerate(gkf.split(data, groups=data['game_id'])):
            train_games = data.iloc[train_idx]['game_id'].unique()
            val_games = data.iloc[val_idx]['game_id'].unique()

            splits.append({
                'fold': fold,
                'train_idx': train_idx,
                'val_idx': val_idx,
                'train_games': train_games,
                'val_games': val_games
            })

            if verbose:
                print(f"  Fold {fold+1}: Train {len(train_games)} games ({len(train_idx):,} episodes), "
                      f"Val {len(val_games)} games ({len(val_idx):,} episodes)")

        if verbose:
            print()

        return splits

    def fit_encoders(self, data, verbose=True):
        """전체 데이터에서 인코더 학습"""
        if verbose:
            print("🔤 인코더 학습 중...")

        # type_name
        all_types = set()
        for col in ['type_name', 'prev_type_name', 'prev2_type_name']:
            if col in data.columns:
                all_types.update(data[col].fillna('Unknown').unique())

        self.type_encoder.fit(list(all_types))

        # result_name
        all_results = set()
        for col in ['result_name', 'prev_result_name']:
            if col in data.columns:
                all_results.update(data[col].fillna('Unknown').unique())

        if len(all_results) > 0:
            self.result_encoder.fit(list(all_results))

        if verbose:
            print(f"✅ 인코더 학습 완료\n")

    def preprocess_pipeline(self, normalize_coords=False, verbose=True):
        """전체 전처리 파이프라인 V2"""
        print("=" * 80)
        print("  K-League Pass Prediction - 전처리 파이프라인 V2")
        print("  개선: 다중공선성 제거 + 비선형 변환 + 위치 특화 + 컨텍스트")
        print("=" * 80)
        print()

        # 1. 데이터 로딩
        train_data, match_info = self.load_data(verbose=verbose)

        # 2. 시간 정렬
        train_data = self.sort_by_time(train_data, verbose=verbose)

        # 3. 기본 피처
        train_data = self.create_basic_features(train_data, verbose=verbose)

        # 4. 🔥 NEW: 비선형 변환 피처
        train_data = self.create_nonlinear_features(train_data, verbose=verbose)

        # 5. 🎯 NEW: 위치별 특화 피처
        train_data = self.create_position_specific_features(train_data, verbose=verbose)

        # 6. 시퀀스 피처
        train_data = self.create_sequence_features(train_data, verbose=verbose)

        # 7. 직전 이벤트 피처
        train_data = self.create_previous_event_features(train_data, verbose=verbose)

        # 8. 고급 전술 피처
        train_data = self.create_advanced_tactical_features(train_data, verbose=verbose)

        # 9. 💎 NEW: 컨텍스트 피처 (마지막 이벤트 추출 전에 전체 데이터로 통계 계산)
        train_data = self.create_context_features(train_data, verbose=verbose)

        # 10. 인코더 학습
        self.fit_encoders(train_data, verbose=verbose)

        # 11. 마지막 이벤트만 추출
        last_events = self.extract_last_events(train_data, verbose=verbose)

        # 12. 범주형 인코딩
        last_events = self.encode_categorical(last_events, fit=False, verbose=verbose)

        # 13. 결측치 처리
        last_events = self.fill_missing(last_events, verbose=verbose)

        # 14. Train/Val Split
        splits = self.create_train_val_split(last_events, n_splits=5, verbose=verbose)

        print("=" * 80)
        print("✅ 전처리 V2 완료!")
        print(f"📊 총 피처 개수: {len(self.get_feature_columns())}개")
        print("=" * 80)

        return last_events, splits

    def get_feature_columns(self):
        """피처 컬럼 목록 반환 (V2 - 개선된 버전)"""
        feature_cols = [
            # ===== 기본 위치 및 이동 =====
            'start_x', 'start_y',
            'delta_x', 'delta_y', 'distance',
            # 제거: start_x_norm, start_y_norm (다중공선성)

            # ===== 골 관련 =====
            'distance_to_goal_start', 'distance_to_goal_end',
            'goal_approach',
            'shooting_angle',

            # ===== 🔥 NEW: 비선형 변환 =====
            'distance_to_goal_inv',
            'distance_to_goal_sqrt',
            'shooting_angle_sin',
            'shooting_angle_cos',
            'start_x_squared',
            'start_y_squared',
            'x_y_interaction',
            'goal_dist_angle_interaction',

            # ===== 영역 분할 =====
            'start_x_zone', 'start_y_zone',
            # 제거: start_x_zone_fine (다중공선성)
            'in_penalty_area', 'in_final_third',

            # ===== 🎯 NEW: 위치별 특화 =====
            'is_defensive_third',
            'goal_urgency',
            'is_central_corridor',
            'near_goal_zone',
            'is_wing_attack',
            'is_midfield_control',

            # ===== 에피소드 정보 =====
            'episode_length', 'event_order',
            'is_first_event',
            'x_progression', 'x_total_progression',
            'relative_time', 'tempo',

            # ===== 속도 및 가속도 =====
            'velocity', 'velocity_x', 'velocity_y',
            'acceleration',

            # ===== 전술적 흐름 =====
            'tempo_change',
            'direction_consistency',
            'horizontal_vertical_ratio',
            'final_third_time_ratio',

            # ===== 압박 강도 =====
            'event_density',
            'local_pressure',
            'weighted_pressure',

            # ===== 공간 창출 =====
            'distance_change_rate',
            'vertical_spread',
            'attack_width',

            # ===== 전술적 벡터 =====
            'forward_momentum',
            'pass_angle_change',

            # ===== 히스토리 기반 =====
            'avg_velocity_3',
            'goal_approach_trend',

            # ===== 최적 경로 =====
            'path_efficiency',

            # ===== 팀 포지셔닝 =====
            'dist_from_team_center',

            # ===== 경기 페이즈 =====
            'match_phase',

            # ===== 💎 NEW: 컨텍스트 피처 =====
            'player_avg_x',
            'player_avg_y',
            'player_avg_pass_dist',
            'player_avg_velocity',
            'team_aggression',
            'team_avg_episode_length',
            'team_avg_tempo',
            'time_pressure',
            'player_position_deviation',

            # ===== 이벤트 타입 =====
            # 제거 가능하지만 호환성 유지: type_name_encoded
            'type_name_encoded',

            # ===== 직전 이벤트 =====
            'prev_type_name_encoded',
            'prev_start_x', 'prev_start_y',
            'prev_end_x', 'prev_end_y',

            # ===== 직전 2개 =====
            'prev2_type_name_encoded',

            # ===== 경기 정보 =====
            'period_id', 'is_home'
        ]

        # result_name이 있으면 추가
        if hasattr(self, 'result_encoder') and len(self.result_encoder.classes_) > 0:
            feature_cols.extend(['result_name_encoded', 'prev_result_name_encoded'])

        return feature_cols

    def save_preprocessor(self, filename='preprocessor_v2.pkl'):
        """전처리 객체 저장"""
        with open(filename, 'wb') as f:
            pickle.dump({
                'scaler_x': self.scaler_x,
                'scaler_y': self.scaler_y,
                'type_encoder': self.type_encoder,
                'result_encoder': self.result_encoder,
                'player_stats': self.player_stats,
                'team_stats': self.team_stats
            }, f)
        print(f"✅ Preprocessor V2 저장: {filename}")

    def load_preprocessor(self, filename='preprocessor_v2.pkl'):
        """전처리 객체 로딩"""
        with open(filename, 'rb') as f:
            saved = pickle.load(f)
            self.scaler_x = saved['scaler_x']
            self.scaler_y = saved['scaler_y']
            self.type_encoder = saved['type_encoder']
            self.result_encoder = saved['result_encoder']
            self.player_stats = saved.get('player_stats')
            self.team_stats = saved.get('team_stats')
        print(f"✅ Preprocessor V2 로딩: {filename}")


def main():
    """V2 테스트 실행"""
    preprocessor = DataPreprocessorV2(data_dir='./data')

    # 전처리 실행
    processed_data, splits = preprocessor.preprocess_pipeline(
        normalize_coords=False,
        verbose=True
    )

    # 피처 컬럼 확인
    feature_cols = preprocessor.get_feature_columns()

    print("\n" + "=" * 80)
    print(f"📊 최종 데이터 Shape: {processed_data.shape}")
    print(f"📊 피처 개수: {len(feature_cols)}")
    print(f"📊 Fold 개수: {len(splits)}")
    print("=" * 80)

    print("\n✅ 사용 가능한 피처:")
    for i, col in enumerate(feature_cols, 1):
        if col in processed_data.columns:
            status = "✓"
        else:
            status = "✗"
        print(f"  {status} {i:2d}. {col}")

    # Preprocessor 저장
    preprocessor.save_preprocessor('preprocessor_v2.pkl')

    # 처리된 데이터 저장
    processed_data.to_csv('processed_train_data_v2.csv', index=False)
    print(f"\n✅ 처리된 데이터 저장: processed_train_data_v2.csv")

    # 개선 사항 요약
    print("\n" + "=" * 80)
    print("📈 V2 개선 사항 요약")
    print("=" * 80)
    print("✅ 제거된 피처 (다중공선성): 5개")
    print("   - start_x_norm, start_y_norm, start_x_zone_fine")
    print("🔥 추가된 비선형 피처: 8개")
    print("   - distance_to_goal_inv, sqrt, angle_sin/cos, squared 등")
    print("🎯 추가된 위치 특화 피처: 6개")
    print("   - is_defensive_third, goal_urgency, is_central_corridor 등")
    print("💎 추가된 컨텍스트 피처: 9개")
    print("   - player_avg_*, team_*, time_pressure 등")
    print("🔧 개선된 결측치 처리: 에피소드 평균 활용")
    print("=" * 80)

    return processed_data, splits, preprocessor


if __name__ == "__main__":
    processed_data, splits, preprocessor = main()

