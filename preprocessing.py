"""
K-League Pass Prediction - Data Preprocessing Pipeline

목표: 재사용 가능한 전처리 파이프라인 구축
기능:
- 데이터 로딩 및 정렬
- 좌표 정규화
- 피처 생성
- Train/Val Split (Game-based)
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

class DataPreprocessor:
    def __init__(self, data_dir='./data'):
        self.data_dir = data_dir
        self.scaler_x = StandardScaler()
        self.scaler_y = StandardScaler()
        self.type_encoder = LabelEncoder()
        self.result_encoder = LabelEncoder()

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

        # 3. 경기장 영역 (3등분)
        data['start_x_zone'] = pd.cut(data['start_x'], bins=[0, 35, 70, 105], labels=[0, 1, 2])
        data['start_y_zone'] = pd.cut(data['start_y'], bins=[0, 22.67, 45.33, 68], labels=[0, 1, 2])

        # 3-1. 세밀한 영역 (5등분) - 전술적 중요도
        data['start_x_zone_fine'] = pd.cut(data['start_x'], bins=[0, 21, 42, 63, 84, 105], labels=[0, 1, 2, 3, 4])

        # 3-2. 위험 지역 플래그 (페널티 박스: x > 87.5, 22.9 < y < 45.1)
        data['in_penalty_area'] = ((data['start_x'] > 87.5) &
                                   (data['start_y'] > 22.9) &
                                   (data['start_y'] < 45.1)).astype(int)

        # 3-3. 최종 1/3 지역 (Final Third)
        data['in_final_third'] = (data['start_x'] > 70).astype(int)

        # 4. 에피소드 내 순서
        data['event_order'] = data.groupby('game_episode').cumcount()

        # 5. 골 각도 (Shooting Angle) - 골대 양 포스트와 이루는 각도
        # 골대: (105, 30.34) ~ (105, 37.66) - 약 7.32m
        post_left_y = 30.34
        post_right_y = 37.66

        # 양 포스트까지의 벡터
        vec_left_x = goal_x - data['start_x']
        vec_left_y = post_left_y - data['start_y']
        vec_right_x = goal_x - data['start_x']
        vec_right_y = post_right_y - data['start_y']

        # 내적과 외적으로 각도 계산
        dot_product = vec_left_x * vec_right_x + vec_left_y * vec_right_y
        cross_product = vec_left_x * vec_right_y - vec_left_y * vec_right_x

        data['shooting_angle'] = np.abs(np.arctan2(cross_product, dot_product))

        # 6. 정규화된 좌표 (0~1 스케일) - 경기장 규격 반영
        data['start_x_norm'] = data['start_x'] / 105.0
        data['start_y_norm'] = data['start_y'] / 68.0
        data['end_x_norm'] = data['end_x'] / 105.0
        data['end_y_norm'] = data['end_y'] / 68.0

        if verbose:
            print(f"✅ 생성된 피처: delta_x/y, distance, distance_to_goal, goal_approach,")
            print(f"   x/y_zone, penalty_area, final_third, shooting_angle, normalized_coords\n")

        return data

    def create_sequence_features(self, data, verbose=True):
        """시퀀스 피처 생성 (각 에피소드별) - 전술적 요소 강화"""
        if verbose:
            print("🔄 시퀀스 피처 생성 중...")

        # 에피소드별 처리
        episode_features = []

        for episode_id, group in data.groupby('game_episode'):
            group = group.copy()

            # 1. 에피소드 길이
            episode_length = len(group)
            group['episode_length'] = episode_length

            # 2. X축 누적 진행도 (공격 전개)
            first_x = group.iloc[0]['start_x']
            last_x = group.iloc[-1]['end_x']
            group['x_progression'] = group['start_x'] - first_x
            group['x_total_progression'] = last_x - first_x  # 전체 진행도

            # 3. 상대 시간 (에피소드 내)
            start_time = group.iloc[0]['time_seconds']
            group['relative_time'] = group['time_seconds'] - start_time

            # 4. 템포 (이벤트당 평균 시간)
            if episode_length > 1:
                duration = group.iloc[-1]['time_seconds'] - group.iloc[0]['time_seconds']
                tempo = duration / episode_length if episode_length > 0 else 0
            else:
                tempo = 0
            group['tempo'] = tempo

            # 5. 속도 계산 (m/s) - 이벤트 간
            time_diff = group['time_seconds'].diff()
            group['velocity_x'] = group['start_x'].diff() / time_diff.replace(0, np.nan)
            group['velocity_y'] = group['start_y'].diff() / time_diff.replace(0, np.nan)
            group['velocity'] = np.sqrt(group['velocity_x']**2 + group['velocity_y']**2)

            # 6. 가속도 (m/s²)
            group['acceleration'] = group['velocity'].diff() / time_diff.replace(0, np.nan)

            # 7. 템포 변화율 (급격한 플레이 변화 감지)
            group['tempo_change'] = tempo - group['relative_time'].rolling(2).mean().fillna(0)

            # 8. 진행 방향 벡터의 일관성 (공격 전개 방향성)
            # 이전 이벤트와 현재 이벤트의 방향 유사도
            prev_delta_x = group['delta_x'].shift(1)
            prev_delta_y = group['delta_y'].shift(1)

            # 코사인 유사도
            dot_prod = group['delta_x'] * prev_delta_x + group['delta_y'] * prev_delta_y
            mag_curr = np.sqrt(group['delta_x']**2 + group['delta_y']**2)
            mag_prev = np.sqrt(prev_delta_x**2 + prev_delta_y**2)
            group['direction_consistency'] = dot_prod / (mag_curr * mag_prev + 1e-10)

            # 9. 지역별 체류 시간
            # 최종 1/3 지역에서의 시간 비율
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
            print(f"✅ 생성된 피처: episode_length, x_progression, velocity, acceleration,")
            print(f"   tempo_change, direction_consistency, spatial_ratios\n")

        return data

    def create_previous_event_features(self, data, verbose=True):
        """직전 이벤트 피처 생성"""
        if verbose:
            print("⬅️  직전 이벤트 피처 생성 중...")

        # 에피소드별 처리
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
            print(f"✅ 생성된 피처: prev_type_name, prev_result_name, prev_start_x/y, prev_end_x/y\n")

        return data

    def create_advanced_tactical_features(self, data, verbose=True):
        """고급 전술 피처 생성 - 압박, 공간 창출, 패스 네트워크"""
        if verbose:
            print("⚽ 고급 전술 피처 생성 중...")

        episode_features = []

        for episode_id, group in data.groupby('game_episode'):
            group = group.copy()

            # ===== 1. 압박 강도 (Pressure Intensity) =====
            # 단순화: 같은 팀의 이벤트 밀도를 압박으로 간주
            # 실제로는 상대 팀 선수 데이터가 필요하지만,
            # 여기서는 이벤트 간 거리와 템포로 압박을 추정

            # 1-1. 이벤트 밀도 (단위 면적당 이벤트 수)
            if len(group) > 1:
                # 에피소드 전체의 활동 영역 계산
                x_range = group['start_x'].max() - group['start_x'].min() + 1
                y_range = group['start_y'].max() - group['start_y'].min() + 1
                area = x_range * y_range
                event_density = len(group) / area
            else:
                event_density = 0
            group['event_density'] = event_density

            # 1-2. 로컬 압박 점수 (주변 반경 내 이벤트 수) - 최적화 버전
            pressure_radius = 10  # 10m 반경

            # 에피소드가 너무 길면 샘플링 (성능 최적화)
            if len(group) > 50:
                # 대표값 사용: 평균적인 압박 강도
                group['local_pressure'] = event_density * 10  # 근사값
                group['weighted_pressure'] = event_density * 5
            else:
                # 정확한 계산 (에피소드가 짧을 때만)
                positions = group[['start_x', 'start_y']].values
                n_events = len(positions)

                # 거리 행렬 계산 (벡터화)
                dist_matrix = cdist(positions, positions, metric='euclidean')

                # 로컬 압박 (반경 내 개수)
                nearby_mask = (dist_matrix <= pressure_radius) & (dist_matrix > 0)
                group['local_pressure'] = nearby_mask.sum(axis=1)

                # 가중 압박 (거리 역수 합)
                weights_matrix = 1 / (dist_matrix + 1)
                np.fill_diagonal(weights_matrix, 0)  # 자기 자신 제외
                group['weighted_pressure'] = weights_matrix.sum(axis=1)

            # ===== 2. 공간 창출 지표 (Space Creation) =====
            # 2-1. 이벤트 간 거리 변화율 (공간이 열리는지)
            prev_distance = group['distance'].shift(1)
            group['distance_change_rate'] = (group['distance'] - prev_distance) / (prev_distance + 1e-10)

            # 2-2. 수직 공간 활용 (넓이 확장)
            group['vertical_spread'] = group['start_y'].rolling(window=3, min_periods=1).std()

            # 2-3. 공격 폭 (Attack Width) - 최근 N개 이벤트의 Y 범위
            group['attack_width'] = group['start_y'].rolling(window=5, min_periods=1).apply(
                lambda x: x.max() - x.min(), raw=True
            )

            # ===== 3. 전술적 벡터 (Tactical Vectors) =====
            # 3-1. 공격 모멘텀 (누적 전진 거리)
            group['forward_momentum'] = group['delta_x'].rolling(window=3, min_periods=1).sum()

            # 3-2. 패스 체인 각도 변화 (전술적 변화)
            # 이전 패스 각도와 현재 패스 각도의 차이
            current_angle = np.arctan2(group['delta_y'], group['delta_x'])
            prev_angle = current_angle.shift(1)
            angle_change = np.abs(current_angle - prev_angle)
            # 라디안 범위를 0~pi로 정규화
            angle_change = np.minimum(angle_change, 2*np.pi - angle_change)
            group['pass_angle_change'] = angle_change

            # ===== 4. 히스토리 기반 패턴 (Rolling Statistics) =====
            # 4-1. 최근 3개 이벤트의 평균 속도
            if 'velocity' in group.columns:
                group['avg_velocity_3'] = group['velocity'].rolling(window=3, min_periods=1).mean()

            # 4-2. 최근 3개 이벤트의 골 접근도
            if 'distance_to_goal_end' in group.columns:
                group['goal_approach_trend'] = group['distance_to_goal_end'].rolling(
                    window=3, min_periods=1
                ).apply(lambda x: x.iloc[0] - x.iloc[-1] if len(x) > 1 else 0, raw=False)

            # ===== 5. 최적 경로 탐색 (Optimal Path) =====
            # 5-1. 골문까지 직선 거리 vs 실제 이동 거리 비율 (효율성)
            if len(group) > 1:
                first_goal_dist = group.iloc[0]['distance_to_goal_start']
                last_goal_dist = group.iloc[-1]['distance_to_goal_end']
                actual_distance = group['distance'].sum()

                # 직선 거리 변화 vs 실제 이동
                direct_progress = first_goal_dist - last_goal_dist
                path_efficiency = direct_progress / (actual_distance + 1e-10)
            else:
                path_efficiency = 0

            group['path_efficiency'] = path_efficiency

            # ===== 6. 팀 중심점 대비 위치 (Relative Positioning) =====
            # 에피소드 내 모든 이벤트의 평균 위치를 팀 중심으로 간주
            team_center_x = group['start_x'].mean()
            team_center_y = group['start_y'].mean()

            group['dist_from_team_center'] = np.sqrt(
                (group['start_x'] - team_center_x)**2 +
                (group['start_y'] - team_center_y)**2
            )

            # ===== 7. 경기 페이즈 분석 =====
            # 시간대별 경기 특성 (초반/중반/후반)
            if 'time_seconds' in group.columns:
                max_time = 5400  # 90분 = 5400초
                group['match_phase'] = pd.cut(
                    group['time_seconds'],
                    bins=[0, 1800, 3600, max_time],
                    labels=[0, 1, 2]  # 초반/중반/후반
                ).astype(float)

            episode_features.append(group)

        data = pd.concat(episode_features, ignore_index=True)

        if verbose:
            print(f"✅ 생성된 고급 피처: pressure, space_creation, momentum, path_efficiency,")
            print(f"   team_positioning, match_phase\n")

        return data

    def extract_last_events(self, data, verbose=True):
        """각 에피소드의 마지막 이벤트만 추출 (예측 대상)"""
        if verbose:
            print("🎯 마지막 이벤트 추출 중...")

        last_events = data.groupby('game_episode').tail(1).copy()

        if verbose:
            print(f"✅ {len(last_events):,}개 에피소드의 마지막 이벤트 추출 완료\n")

        return last_events

    def encode_categorical(self, data, fit=True, verbose=True):
        """범주형 변수 인코딩 (fit=False일 때는 이미 학습된 인코더 사용)"""
        if verbose:
            print("🔤 범주형 변수 인코딩 적용 중...")

        # type_name 인코딩 (Unknown으로 fillna)
        data['type_name'] = data['type_name'].fillna('Unknown')
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
            data['result_name_encoded'] = self.result_encoder.transform(data['result_name'])

            if 'prev_result_name' in data.columns:
                data['prev_result_name'] = data['prev_result_name'].fillna('Unknown')
                data['prev_result_name_encoded'] = self.result_encoder.transform(data['prev_result_name'])

        if verbose:
            print(f"✅ 인코딩 적용 완료\n")

        return data

    def fill_missing(self, data, verbose=True):
        """결측치 처리"""
        if verbose:
            print("🔧 결측치 처리 중...")

        # 직전 이벤트 피처의 NaN (첫 번째 이벤트)
        prev_cols = [col for col in data.columns if col.startswith('prev_')]

        for col in prev_cols:
            if col.endswith('_encoded'):
                data[col] = data[col].fillna(-1)  # 특수 값
            elif data[col].dtype in ['float64', 'int64']:
                data[col] = data[col].fillna(0)  # 숫자는 0
            else:
                data[col] = data[col].fillna('Unknown')  # 문자는 Unknown

        # zone 피처 (categorical을 numeric으로 변환)
        if 'start_x_zone' in data.columns:
            data['start_x_zone'] = data['start_x_zone'].astype(float).fillna(-1)
            data['start_y_zone'] = data['start_y_zone'].astype(float).fillna(-1)

        if verbose:
            print(f"✅ 결측치 처리 완료\n")

        return data

    def normalize_coordinates(self, data, fit=True, verbose=True):
        """좌표 정규화 (선택적)"""
        if verbose:
            print("📏 좌표 정규화 중...")

        coord_cols = ['start_x', 'start_y']

        if fit:
            data[coord_cols] = self.scaler_x.fit_transform(data[coord_cols])
        else:
            data[coord_cols] = self.scaler_x.transform(data[coord_cols])

        # 직전 이벤트 좌표도 정규화
        if 'prev_start_x' in data.columns:
            prev_coord_cols = ['prev_start_x', 'prev_start_y', 'prev_end_x', 'prev_end_y']
            prev_coord_cols = [col for col in prev_coord_cols if col in data.columns]

            # NaN이 아닌 값만 정규화
            for col in prev_coord_cols:
                mask = data[col].notna()
                if mask.sum() > 0:
                    if 'x' in col:
                        data.loc[mask, col] = self.scaler_x.transform(data.loc[mask, [col.replace('prev_', '').replace('end_', 'start_')]])[col.replace('prev_', '').replace('end_', 'start_')]

        if verbose:
            print(f"✅ 좌표 정규화 완료\n")

        return data

    def create_train_val_split(self, data, n_splits=5, random_state=42, verbose=True):
        """Game-based K-Fold Split"""
        if verbose:
            print(f"📊 {n_splits}-Fold Game-based Split 생성 중...")

        # 게임 ID 추출
        games = data['game_id'].unique()

        # GroupKFold (게임 단위)
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
        """전체 데이터에서 인코더를 먼저 fit"""
        if verbose:
            print("🔤 인코더 학습 중 (전체 데이터)...")

        # type_name 수집
        all_types = set()
        for col in ['type_name', 'prev_type_name', 'prev2_type_name']:
            if col in data.columns:
                all_types.update(data[col].fillna('Unknown').unique())

        self.type_encoder.fit(list(all_types))

        # result_name 수집
        all_results = set()
        for col in ['result_name', 'prev_result_name']:
            if col in data.columns:
                all_results.update(data[col].fillna('Unknown').unique())

        if len(all_results) > 0:
            self.result_encoder.fit(list(all_results))

        if verbose:
            print(f"✅ 인코더 학습 완료: {len(self.type_encoder.classes_)}개 이벤트 타입, "
                  f"{len(self.result_encoder.classes_) if len(all_results) > 0 else 0}개 결과 타입\n")

    def preprocess_pipeline(self, normalize_coords=False, verbose=True):
        """전체 전처리 파이프라인"""
        print("=" * 80)
        print("  K-League Pass Prediction - 데이터 전처리 파이프라인")
        print("=" * 80)
        print()

        # 1. 데이터 로딩
        train_data, match_info = self.load_data(verbose=verbose)

        # 2. 시간 정렬
        train_data = self.sort_by_time(train_data, verbose=verbose)

        # 3. 기본 피처 생성
        train_data = self.create_basic_features(train_data, verbose=verbose)

        # 4. 시퀀스 피처 생성
        train_data = self.create_sequence_features(train_data, verbose=verbose)

        # 5. 직전 이벤트 피처
        train_data = self.create_previous_event_features(train_data, verbose=verbose)

        # 5.5. 고급 전술 피처 생성 (NEW!)
        train_data = self.create_advanced_tactical_features(train_data, verbose=verbose)

        # 5.6. 인코더 학습 (마지막 이벤트 추출 전에 전체 데이터로 fit)
        self.fit_encoders(train_data, verbose=verbose)

        # 6. 마지막 이벤트만 추출
        last_events = self.extract_last_events(train_data, verbose=verbose)

        # 7. 범주형 인코딩
        last_events = self.encode_categorical(last_events, fit=False, verbose=verbose)

        # 8. 결측치 처리
        last_events = self.fill_missing(last_events, verbose=verbose)

        # 9. 좌표 정규화 (선택적)
        if normalize_coords:
            last_events = self.normalize_coordinates(last_events, fit=True, verbose=verbose)

        # 10. Train/Val Split
        splits = self.create_train_val_split(last_events, n_splits=5, verbose=verbose)

        print("=" * 80)
        print("✅ 전처리 완료!")
        print("=" * 80)

        return last_events, splits

    def get_feature_columns(self):
        """피처 컬럼 목록 반환 (모든 전술 피처 포함)"""
        feature_cols = [
            # ===== 기본 위치 및 이동 =====
            'start_x', 'start_y',
            'delta_x', 'delta_y', 'distance',

            # 정규화 좌표
            'start_x_norm', 'start_y_norm',

            # ===== 골 관련 =====
            'distance_to_goal_start', 'distance_to_goal_end',
            'goal_approach',
            'shooting_angle',

            # ===== 영역 분할 =====
            'start_x_zone', 'start_y_zone', 'start_x_zone_fine',
            'in_penalty_area', 'in_final_third',

            # ===== 에피소드 정보 =====
            'episode_length', 'event_order',
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

            # ===== 압박 강도 (Pressure) =====
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

            # ===== 이벤트 타입 =====
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

    def save_preprocessor(self, filename='preprocessor.pkl'):
        """전처리 객체 저장"""
        with open(filename, 'wb') as f:
            pickle.dump({
                'scaler_x': self.scaler_x,
                'scaler_y': self.scaler_y,
                'type_encoder': self.type_encoder,
                'result_encoder': self.result_encoder
            }, f)
        print(f"✅ Preprocessor 저장: {filename}")

    def load_preprocessor(self, filename='preprocessor.pkl'):
        """전처리 객체 로딩"""
        with open(filename, 'rb') as f:
            saved = pickle.load(f)
            self.scaler_x = saved['scaler_x']
            self.scaler_y = saved['scaler_y']
            self.type_encoder = saved['type_encoder']
            self.result_encoder = saved['result_encoder']
        print(f"✅ Preprocessor 로딩: {filename}")

def main():
    """테스트 실행"""
    # Preprocessor 초기화
    preprocessor = DataPreprocessor(data_dir='./data')

    # 전처리 실행
    processed_data, splits = preprocessor.preprocess_pipeline(
        normalize_coords=False,  # 좌표 정규화 안함 (XGBoost는 불필요)
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
            print(f"  {i:2d}. {col}")
        else:
            print(f"  {i:2d}. {col} (없음)")

    # 샘플 데이터 확인
    print("\n" + "=" * 80)
    print("📊 샘플 데이터 (처음 5행):")
    print("=" * 80)
    sample_cols = ['game_episode', 'start_x', 'start_y', 'end_x', 'end_y',
                   'episode_length', 'type_name_encoded', 'prev_type_name_encoded']
    print(processed_data[sample_cols].head())

    # Preprocessor 저장
    preprocessor.save_preprocessor('preprocessor.pkl')

    # 처리된 데이터 저장
    processed_data.to_csv('processed_train_data.csv', index=False)
    print(f"\n✅ 처리된 데이터 저장: processed_train_data.csv")

    return processed_data, splits, preprocessor

if __name__ == "__main__":
    processed_data, splits, preprocessor = main()

