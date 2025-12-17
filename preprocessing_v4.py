"""
K-League Pass Prediction - Data Preprocessing Pipeline V4

V2 + V3의 핵심 개선사항 통합:
✅ Data Leakage 완전 제거 - 마지막 이벤트의 end_x, end_y 마스킹
✅ 시퀀스 모델링 - 마지막 20개 이벤트의 패턴 학습 (Wide format)
✅ Train/Test 통합 처리 - 일관된 범주형 인코딩
✅ V2의 풍부한 피처 엔지니어링 (비선형 변환, 위치 특화, 컨텍스트)
✅ 5-Fold GroupKFold 앙상블

목표: V2의 도메인 지식 + V3의 시퀀스 모델링 = 최고 성능
작성일: 2025-12-17
"""

import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GroupKFold
from scipy.spatial.distance import cdist
import pickle
import warnings
warnings.filterwarnings('ignore')


class DataPreprocessorV4:
    def __init__(self, data_dir='./data', K=20):
        """
        Args:
            data_dir: 데이터 디렉토리
            K: 마지막 K개 이벤트 사용 (기본 20)
        """
        self.data_dir = data_dir
        self.K = K
        self.type_encoder = LabelEncoder()
        self.result_encoder = LabelEncoder()

        # 선수/팀 통계 저장
        self.player_stats = None
        self.team_stats = None

    def load_data(self, verbose=True):
        """데이터 로딩 (Train + Test 통합)"""
        if verbose:
            print("📊 데이터 로딩 중...")

        # Train 데이터
        train_path = os.path.join(self.data_dir, 'train.csv')
        train_data = pd.read_csv(train_path)
        train_data['is_train'] = 1

        # Test 데이터 (test_index.csv 사용)
        test_index_path = os.path.join(self.data_dir, 'test_index.csv')
        test_index = pd.read_csv(test_index_path)

        test_events_list = []
        for _, row in test_index.iterrows():
            ep_path = os.path.join(self.data_dir, row['path'].replace('./', ''))
            df_ep = pd.read_csv(ep_path)
            test_events_list.append(df_ep)

        test_events = pd.concat(test_events_list, ignore_index=True)
        test_events['is_train'] = 0

        # Train + Test 결합
        data = pd.concat([train_data, test_events], ignore_index=True)

        if verbose:
            print(f"✅ Train: {len(train_data):,} 이벤트, {train_data['game_episode'].nunique():,} 에피소드")
            print(f"✅ Test: {len(test_events):,} 이벤트, {test_events['game_episode'].nunique():,} 에피소드\n")

        return data

    def sort_and_index(self, data, verbose=True):
        """시간 정렬 및 인덱싱"""
        if verbose:
            print("⏰ 시간 정렬 및 인덱싱...")

        # 정렬
        data = data.sort_values(['game_episode', 'time_seconds', 'action_id']).reset_index(drop=True)

        # 에피소드 내 인덱스
        data['event_idx'] = data.groupby('game_episode').cumcount()
        data['n_events'] = data.groupby('game_episode')['event_idx'].transform('max') + 1
        data['ep_idx_norm'] = data['event_idx'] / (data['n_events'] - 1).clip(lower=1)

        # 역인덱스 (0이 마지막 이벤트)
        data['rev_idx'] = data.groupby('game_episode')['event_idx'].transform(
            lambda s: s.max() - s
        )

        # 마지막 이벤트 플래그
        data['last_idx'] = data.groupby('game_episode')['event_idx'].transform('max')
        data['is_last'] = (data['event_idx'] == data['last_idx']).astype(int)

        if verbose:
            print("✅ 정렬 및 인덱싱 완료\n")

        return data

    def create_basic_features(self, data, verbose=True):
        """기본 피처 생성 (V2 기반)"""
        if verbose:
            print("🔧 기본 피처 생성 중...")

        # 시간 차이
        data['prev_time'] = data.groupby('game_episode')['time_seconds'].shift(1)
        data['dt'] = data['time_seconds'] - data['prev_time']
        data['dt'] = data['dt'].fillna(0.0)

        # 이동량/거리
        data['dx'] = data['end_x'] - data['start_x']
        data['dy'] = data['end_y'] - data['start_y']
        data['dist'] = np.sqrt(data['dx']**2 + data['dy']**2)

        # 속도 (dt=0 보호)
        data['speed'] = data['dist'] / data['dt'].replace(0, 1e-3)

        # Zone 분할
        data['x_zone'] = (data['start_x'] / (105/7)).astype(int).clip(0, 6)
        data['lane'] = pd.cut(
            data['start_y'],
            bins=[0, 68/3, 2*68/3, 68],
            labels=[0, 1, 2],
            include_lowest=True
        ).astype(int)

        # 골문 거리 (오른쪽 골문: 105, 34)
        goal_x, goal_y = 105, 34
        data['distance_to_goal_start'] = np.sqrt(
            (data['start_x'] - goal_x)**2 +
            (data['start_y'] - goal_y)**2
        )
        data['distance_to_goal_end'] = np.sqrt(
            (data['end_x'] - goal_x)**2 +
            (data['end_y'] - goal_y)**2
        )

        # 골문 진행도
        data['goal_approach'] = data['distance_to_goal_start'] - data['distance_to_goal_end']

        # 페널티 박스
        data['in_penalty_area'] = ((data['start_x'] > 87.5) &
                                   (data['start_y'] > 22.9) &
                                   (data['start_y'] < 45.1)).astype(int)

        # Final third
        data['in_final_third'] = (data['start_x'] > 70).astype(int)

        # 골 각도
        post_left_y = 30.34
        post_right_y = 37.66

        vec_left_x = goal_x - data['start_x']
        vec_left_y = post_left_y - data['start_y']
        vec_right_x = goal_x - data['start_x']
        vec_right_y = post_right_y - data['start_y']

        dot_product = vec_left_x * vec_right_x + vec_left_y * vec_right_y
        cross_product = vec_left_x * vec_right_y - vec_left_y * vec_right_x

        data['shooting_angle'] = np.abs(np.arctan2(cross_product, dot_product))

        # 경기 시간 (분)
        data['game_clock_min'] = np.where(
            data['period_id'] == 1,
            data['time_seconds'] / 60.0,
            45.0 + data['time_seconds'] / 60.0
        )

        if verbose:
            print("✅ 기본 피처 생성 완료\n")

        return data

    def create_nonlinear_features(self, data, verbose=True):
        """🔥 비선형 변환 피처 생성 (V2)"""
        if verbose:
            print("🔥 비선형 변환 피처 생성 중...")

        # 1. 골문 거리 역수
        data['distance_to_goal_inv'] = 1 / (data['distance_to_goal_start'] + 1)

        # 2. 골문 거리 제곱근
        data['distance_to_goal_sqrt'] = np.sqrt(data['distance_to_goal_start'])

        # 3. 각도의 삼각함수
        data['shooting_angle_sin'] = np.sin(data['shooting_angle'])
        data['shooting_angle_cos'] = np.cos(data['shooting_angle'])

        # 4. 위치의 제곱
        data['start_x_squared'] = data['start_x'] ** 2
        data['start_y_squared'] = data['start_y'] ** 2

        # 5. 상호작용 피처
        data['x_y_interaction'] = data['start_x'] * data['start_y']
        data['goal_dist_angle_interaction'] = data['distance_to_goal_start'] * data['shooting_angle']

        if verbose:
            print(f"✅ 비선형 피처 8개 생성 완료\n")

        return data

    def create_position_specific_features(self, data, verbose=True):
        """🎯 위치별 특화 피처 (V2)"""
        if verbose:
            print("🎯 위치별 특화 피처 생성 중...")

        # 1. 수비진 특화
        data['is_defensive_third'] = (data['start_x'] < 35).astype(int)

        # 2. 공격진 특화 - 골 긴급도
        data['goal_urgency'] = np.exp(-data['distance_to_goal_start'] / 20)

        # 3. Y축 중앙 복도
        data['is_central_corridor'] = ((data['start_y'] > 20) &
                                       (data['start_y'] < 48)).astype(int)

        # 4. 골문 근접도
        data['near_goal_zone'] = ((data['distance_to_goal_start'] < 25) &
                                  (data['start_x'] > 80)).astype(int)

        # 5. 사이드 공격
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

    def extract_labels(self, data, verbose=True):
        """🎯 타겟 레이블 추출 (Train 전용)"""
        if verbose:
            print("🎯 타겟 레이블 추출 중...")

        # Train 이벤트만 필터링
        train_events = data[data['is_train'] == 1].copy()
        last_events = train_events[train_events['is_last'] == 1].copy()

        labels = last_events[['game_episode', 'end_x', 'end_y']].rename(
            columns={'end_x': 'target_x', 'end_y': 'target_y'}
        )

        # Episode 메타 정보
        ep_meta = last_events[['game_episode', 'game_id', 'team_id', 'is_home',
                               'period_id', 'time_seconds', 'game_clock_min']].copy()
        ep_meta = ep_meta.rename(columns={'team_id': 'final_team_id'})

        if verbose:
            print(f"✅ {len(labels):,}개 Train 에피소드의 타겟 추출 완료\n")

        return labels, ep_meta

    def add_final_team_flag(self, data, ep_meta, verbose=True):
        """공격 팀 플래그 추가"""
        if verbose:
            print("⚽ 공격 팀 플래그 추가 중...")

        data = data.merge(
            ep_meta[['game_episode', 'final_team_id']],
            on='game_episode',
            how='left'
        )

        data['is_final_team'] = (data['team_id'] == data['final_team_id']).astype(int)

        if verbose:
            print("✅ 공격 팀 플래그 추가 완료\n")

        return data

    def mask_target_leakage(self, data, verbose=True):
        """🚨 Data Leakage 제거 (V3 핵심)"""
        if verbose:
            print("🚨 Data Leakage 제거 중...")

        mask_last = data['is_last'] == 1

        # 마지막 이벤트의 end 정보 제거
        leakage_cols = ['end_x', 'end_y', 'dx', 'dy', 'dist', 'speed',
                       'distance_to_goal_end', 'goal_approach']
        for col in leakage_cols:
            if col in data.columns:
                data.loc[mask_last, col] = np.nan

        if verbose:
            print(f"✅ {len(leakage_cols)}개 컬럼의 Leakage 제거 완료")
            print("   → 마지막 이벤트의 end 정보 NaN 처리\n")

        return data

    def encode_categorical(self, data, verbose=True):
        """범주형 변수 인코딩"""
        if verbose:
            print("🔤 범주형 변수 인코딩 중...")

        # type_name 인코딩
        data['type_name'] = data['type_name'].fillna('__NA_TYPE__')
        data['type_id'] = self.type_encoder.fit_transform(data['type_name'])

        # result_name 인코딩
        data['result_name'] = data['result_name'].fillna('__NA_RES__')
        data['res_id'] = self.result_encoder.fit_transform(data['result_name'])

        # team_id 인코딩 (문자열인 경우)
        if data['team_id'].dtype == 'object':
            le_team = LabelEncoder()
            data['team_id_enc'] = le_team.fit_transform(data['team_id'])
        else:
            data['team_id_enc'] = data['team_id'].astype(int)

        if verbose:
            print("✅ 인코딩 완료\n")

        return data

    def filter_last_k_events(self, data, verbose=True):
        """🎯 마지막 K개 이벤트만 필터링 (V3 핵심)"""
        if verbose:
            print(f"🎯 마지막 {self.K}개 이벤트 필터링 중...")

        lastK = data[data['rev_idx'] < self.K].copy()

        # pos_in_K: 0~(K-1), 앞쪽 패딩 고려
        def assign_pos_in_K(df):
            df = df.sort_values('event_idx')
            L = len(df)
            df = df.copy()
            df['pos_in_K'] = np.arange(self.K - L, self.K)
            return df

        lastK = lastK.groupby('game_episode', group_keys=False).apply(assign_pos_in_K)

        if verbose:
            print(f"✅ {len(lastK):,}개 이벤트 필터링 완료\n")

        return lastK

    def create_wide_features(self, lastK, ep_meta, labels, verbose=True):
        """🌐 Wide format 피처 생성 (V3 핵심)"""
        if verbose:
            print("🌐 Wide format 피처 생성 중...")

        # 사용할 이벤트 피처
        num_cols = [
            'start_x', 'start_y',
            'end_x', 'end_y',
            'dx', 'dy', 'dist', 'speed',
            'dt',
            'ep_idx_norm',
            'x_zone', 'lane',
            'is_final_team',
            'distance_to_goal_start',
            'distance_to_goal_end',
            'goal_approach',
            'in_penalty_area',
            'in_final_third',
            'shooting_angle',
            'distance_to_goal_inv',
            'distance_to_goal_sqrt',
            'shooting_angle_sin',
            'shooting_angle_cos',
            'start_x_squared',
            'start_y_squared',
            'x_y_interaction',
            'goal_dist_angle_interaction',
            'is_defensive_third',
            'goal_urgency',
            'is_central_corridor',
            'near_goal_zone',
            'is_wing_attack',
            'is_midfield_control',
        ]

        cat_cols = [
            'type_id',
            'res_id',
            'team_id_enc',
            'is_home',
            'period_id',
            'is_last',
        ]

        feature_cols = num_cols + cat_cols

        # 실제 존재하는 컬럼만 사용
        num_cols = [c for c in num_cols if c in lastK.columns]
        cat_cols = [c for c in cat_cols if c in lastK.columns]
        feature_cols = num_cols + cat_cols

        wide = lastK[['game_episode', 'pos_in_K'] + feature_cols].copy()

        # 숫자형 pivot
        wide_num = wide.pivot_table(
            index='game_episode',
            columns='pos_in_K',
            values=num_cols,
            aggfunc='first'
        )

        # 범주형 pivot
        wide_cat = wide.pivot_table(
            index='game_episode',
            columns='pos_in_K',
            values=cat_cols,
            aggfunc='first'
        )

        # 컬럼 이름 평탄화
        wide_num.columns = [f"{c}_{int(pos)}" for (c, pos) in wide_num.columns]
        wide_cat.columns = [f"{c}_{int(pos)}" for (c, pos) in wide_cat.columns]

        X = pd.concat([wide_num, wide_cat], axis=1).reset_index()

        # Episode-level 메타 붙이기
        X = X.merge(
            ep_meta[['game_episode', 'game_id', 'game_clock_min', 'final_team_id', 'is_home', 'period_id']],
            on='game_episode',
            how='left'
        )

        # Train 라벨 붙이기
        X = X.merge(labels, on='game_episode', how='left')

        if verbose:
            print(f"✅ Wide format 피처 생성 완료")
            print(f"   - 피처 차원: {X.shape[1] - 4} (episode, game_id, target 제외)\n")

        return X

    def split_train_test(self, X, verbose=True):
        """Train/Test 분리"""
        if verbose:
            print("📊 Train/Test 분리 중...")

        # Train: target이 있는 데이터
        train_mask = X['target_x'].notna()
        X_train = X[train_mask].copy()
        X_test = X[~train_mask].copy()

        if verbose:
            print(f"✅ Train: {len(X_train):,}, Test: {len(X_test):,}\n")

        return X_train, X_test

    def preprocess_pipeline(self, verbose=True):
        """전체 전처리 파이프라인 V4"""
        print("=" * 80)
        print("  K-League Pass Prediction - 전처리 파이프라인 V4")
        print("  V2 피처 + V3 시퀀스 모델링 = 최고 성능")
        print("=" * 80)
        print()

        # 1. 데이터 로딩 (Train + Test)
        data = self.load_data(verbose=verbose)

        # 2. 정렬 및 인덱싱
        data = self.sort_and_index(data, verbose=verbose)

        # 3. 기본 피처
        data = self.create_basic_features(data, verbose=verbose)

        # 4. 비선형 변환 피처 (V2)
        data = self.create_nonlinear_features(data, verbose=verbose)

        # 5. 위치별 특화 피처 (V2)
        data = self.create_position_specific_features(data, verbose=verbose)

        # 6. 타겟 레이블 추출
        labels, ep_meta = self.extract_labels(data, verbose=verbose)

        # 7. 공격 팀 플래그
        data = self.add_final_team_flag(data, ep_meta, verbose=verbose)

        # 8. 🚨 Data Leakage 제거 (V3 핵심)
        data = self.mask_target_leakage(data, verbose=verbose)

        # 9. 범주형 인코딩
        data = self.encode_categorical(data, verbose=verbose)

        # 10. 🎯 마지막 K개 이벤트 필터링 (V3 핵심)
        lastK = self.filter_last_k_events(data, verbose=verbose)

        # 11. 🌐 Wide format 피처 생성 (V3 핵심)
        X = self.create_wide_features(lastK, ep_meta, labels, verbose=verbose)

        # 12. Train/Test 분리
        X_train, X_test = self.split_train_test(X, verbose=verbose)

        print("=" * 80)
        print("✅ 전처리 V4 완료!")
        print(f"📊 Train Shape: {X_train.shape}")
        print(f"📊 Test Shape: {X_test.shape}")
        print(f"📊 총 피처 개수: {X_train.shape[1] - 4}개")
        print("=" * 80)

        return X_train, X_test

    def save_preprocessor(self, filename='preprocessor_v4.pkl'):
        """전처리 객체 저장"""
        with open(filename, 'wb') as f:
            pickle.dump({
                'type_encoder': self.type_encoder,
                'result_encoder': self.result_encoder,
                'K': self.K
            }, f)
        print(f"✅ Preprocessor V4 저장: {filename}")

    def load_preprocessor(self, filename='preprocessor_v4.pkl'):
        """전처리 객체 로딩"""
        with open(filename, 'rb') as f:
            saved = pickle.load(f)
            self.type_encoder = saved['type_encoder']
            self.result_encoder = saved['result_encoder']
            self.K = saved['K']
        print(f"✅ Preprocessor V4 로딩: {filename}")


def main():
    """V4 테스트 실행"""
    preprocessor = DataPreprocessorV4(data_dir='./data', K=20)

    # 전처리 실행
    X_train, X_test = preprocessor.preprocess_pipeline(verbose=True)

    # 데이터 저장
    X_train.to_csv('processed_train_data_v4.csv', index=False)
    X_test.to_csv('processed_test_data_v4.csv', index=False)

    print(f"\n✅ 처리된 데이터 저장:")
    print(f"   - processed_train_data_v4.csv")
    print(f"   - processed_test_data_v4.csv")

    # Preprocessor 저장
    preprocessor.save_preprocessor('preprocessor_v4.pkl')

    # 개선 사항 요약
    print("\n" + "=" * 80)
    print("📈 V4 개선 사항 요약")
    print("=" * 80)
    print("✅ V2의 도메인 지식 피처:")
    print("   - 비선형 변환 피처 8개")
    print("   - 위치 특화 피처 6개")
    print("   - 골 관련 고급 피처")
    print("\n✅ V3의 시퀀스 모델링:")
    print("   - 마지막 20개 이벤트 사용")
    print("   - Wide format (시간 순서 보존)")
    print("   - Data Leakage 완전 제거")
    print("   - Train/Test 통합 처리")
    print("\n🎯 기대 효과:")
    print("   - V2의 풍부한 도메인 지식")
    print("   - V3의 안정적인 일반화 성능")
    print("   - 예상 Test 성능: 14~16점대")
    print("=" * 80)

    return X_train, X_test, preprocessor


if __name__ == "__main__":
    X_train, X_test, preprocessor = main()

