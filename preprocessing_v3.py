"""
K-League Pass Prediction - Data Preprocessing Pipeline V3

핵심 개선사항:
1. ✅ Data Leakage 제거 (end_x, end_y 제거)
2. ✅ 시퀀스 모델링 도입 (마지막 K개 이벤트, Wide format)
3. ✅ 다른 사람의 우수 방식 채택
4. ✅ 5-Fold GroupKFold 지원

목표: Test 성능 16점대 이하
작성일: 2025-12-16
"""

import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GroupKFold
import pickle
import warnings
warnings.filterwarnings('ignore')


class DataPreprocessorV3:
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
        self.team_encoder = LabelEncoder()

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
            # 상대 경로를 절대 경로로 변환
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

        # 마지막 이벤트 플래그
        data['last_idx'] = data.groupby('game_episode')['event_idx'].transform('max')
        data['is_last'] = (data['event_idx'] == data['last_idx']).astype(int)

        if verbose:
            print("✅ 정렬 및 인덱싱 완료\n")

        return data

    def create_basic_features(self, data, verbose=True):
        """기본 피처 생성"""
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

        # 속도
        data['speed'] = data['dist'] / data['dt'].replace(0, 1e-3)

        # Zone 분할
        data['x_zone'] = (data['start_x'] / (105/7)).astype(int).clip(0, 6)
        data['lane'] = pd.cut(
            data['start_y'],
            bins=[0, 68/3, 2*68/3, 68],
            labels=[0, 1, 2],
            include_lowest=True
        ).astype(int)

        # 골문 거리
        goal_x, goal_y = 105, 34
        data['distance_to_goal_start'] = np.sqrt(
            (data['start_x'] - goal_x)**2 +
            (data['start_y'] - goal_y)**2
        )

        # 페널티 박스
        data['in_penalty_area'] = ((data['start_x'] > 87.5) &
                                   (data['start_y'] > 22.9) &
                                   (data['start_y'] < 45.1)).astype(int)

        # Final third
        data['in_final_third'] = (data['start_x'] > 70).astype(int)

        # 경기 시간 (분)
        data['game_clock_min'] = np.where(
            data['period_id'] == 1,
            data['time_seconds'] / 60.0,
            45.0 + data['time_seconds'] / 60.0
        )

        if verbose:
            print("✅ 기본 피처 생성 완료\n")

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
        """🚨 Data Leakage 제거"""
        if verbose:
            print("🚨 Data Leakage 제거 중...")

        mask_last = data['is_last'] == 1

        leakage_cols = ['end_x', 'end_y', 'dx', 'dy', 'dist', 'speed']
        for col in leakage_cols:
            if col in data.columns:
                data.loc[mask_last, col] = np.nan

        if verbose:
            print(f"✅ {len(leakage_cols)}개 컬럼의 Leakage 제거 완료")
            print("   → 마지막 이벤트의 end_x, end_y 등 NaN 처리\n")

        return data

    def encode_categorical(self, data, verbose=True):
        """범주형 인코딩"""
        if verbose:
            print("🔤 범주형 인코딩 중...")

        data['type_name'] = data['type_name'].fillna('__NA_TYPE__')
        data['type_id'] = self.type_encoder.fit_transform(data['type_name'])

        data['result_name'] = data['result_name'].fillna('__NA_RES__')
        data['res_id'] = self.result_encoder.fit_transform(data['result_name'])

        if data['team_id'].dtype == 'object':
            data['team_id_enc'] = self.team_encoder.fit_transform(data['team_id'])
        else:
            data['team_id_enc'] = data['team_id'].astype(int)

        if verbose:
            print("✅ 범주형 인코딩 완료\n")

        return data

    def create_lastK_wide_features(self, data, verbose=True):
        """🔥 핵심: 마지막 K개 이벤트 Wide format 변환"""
        if verbose:
            print(f"🔥 마지막 {self.K}개 이벤트 Wide format 변환 중...")

        # 역순 인덱스
        data['rev_idx'] = data.groupby('game_episode')['event_idx'].transform(
            lambda s: s.max() - s
        )

        # 마지막 K개만
        lastK = data[data['rev_idx'] < self.K].copy()

        # pos_in_K 할당
        def assign_pos_in_K(df):
            df = df.sort_values('event_idx')
            L = len(df)
            df = df.copy()
            df['pos_in_K'] = np.arange(self.K - L, self.K)
            return df

        lastK = lastK.groupby('game_episode', group_keys=False).apply(assign_pos_in_K)

        # Wide format 피처 선택 (sample_from_other.py와 동일하게)
        num_cols = [
            'start_x', 'start_y',
            'end_x', 'end_y',
            'dx', 'dy', 'dist', 'speed',
            'dt',
            'ep_idx_norm',
            'x_zone', 'lane',
            'is_final_team',
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
        feature_cols = [c for c in feature_cols if c in lastK.columns]

        wide = lastK[['game_episode', 'pos_in_K'] + feature_cols].copy()

        # 숫자형과 범주형 따로 pivot (sample_from_other.py 방식)
        num_cols_available = [c for c in num_cols if c in wide.columns]
        cat_cols_available = [c for c in cat_cols if c in wide.columns]

        wide_num = wide.pivot_table(
            index='game_episode',
            columns='pos_in_K',
            values=num_cols_available,
            aggfunc='first'
        )

        wide_cat = wide.pivot_table(
            index='game_episode',
            columns='pos_in_K',
            values=cat_cols_available,
            aggfunc='first'
        )

        # 컬럼 이름 평탄화
        wide_num.columns = [f"{c}_{int(pos)}" for (c, pos) in wide_num.columns]
        wide_cat.columns = [f"{c}_{int(pos)}" for (c, pos) in wide_cat.columns]

        # 병합
        X = pd.concat([wide_num, wide_cat], axis=1).reset_index()

        if verbose:
            print(f"✅ Wide format 변환 완료")
            print(f"   - 에피소드 수: {len(X):,}")
            print(f"   - 피처 수: {X.shape[1] - 1}\n")

        return X

    def merge_metadata_and_labels(self, X, ep_meta, labels, verbose=True):
        """메타데이터 및 레이블 병합"""
        if verbose:
            print("🔗 메타데이터 및 레이블 병합 중...")

        # 메타 정보 병합 (sample_from_other.py와 동일)
        X = X.merge(
            ep_meta[['game_episode', 'game_id', 'game_clock_min', 'final_team_id', 'is_home', 'period_id']],
            on='game_episode',
            how='left'
        )

        # 레이블 병합 (Test는 NaN으로 남음)
        X = X.merge(labels, on='game_episode', how='left')

        if verbose:
            labeled = X['target_x'].notna().sum()
            total = len(X)
            print(f"✅ 병합 완료 (Train: {labeled:,}, Test: {total-labeled:,})\n")

        return X

    def prepare_model_data(self, X, verbose=True):
        """모델 학습용 데이터 준비"""
        if verbose:
            print("📊 모델 데이터 준비 중...")

        train_mask = X['target_x'].notna()
        X_train = X[train_mask].copy()

        y_train_x = X_train['target_x'].values
        y_train_y = X_train['target_y'].values
        y_train = np.column_stack([y_train_x, y_train_y])

        # 피처 추출
        drop_cols = ['game_episode', 'game_id', 'target_x', 'target_y']
        X_train_feat = X_train.drop(columns=drop_cols)

        # NaN 채우기
        X_train_feat = X_train_feat.fillna(0)

        # game_id 추출
        game_ids = X_train['game_id'].values

        if verbose:
            print(f"✅ 학습 데이터 준비 완료")
            print(f"   - 샘플 수: {len(X_train_feat):,}")
            print(f"   - 피처 수: {X_train_feat.shape[1]}\n")

        return X_train_feat, y_train, game_ids

    def create_train_val_split(self, X_train_feat, y_train, game_ids, n_splits=5, verbose=True):
        """5-Fold GroupKFold Split"""
        if verbose:
            print(f"📊 {n_splits}-Fold GroupKFold 생성 중...")

        gkf = GroupKFold(n_splits=n_splits)

        splits = []
        for fold, (train_idx, val_idx) in enumerate(gkf.split(X_train_feat, groups=game_ids)):
            splits.append({
                'fold': fold,
                'train_idx': train_idx,
                'val_idx': val_idx,
            })

            if verbose:
                print(f"  Fold {fold+1}: Train {len(train_idx):,}, Val {len(val_idx):,}")

        if verbose:
            print()

        return splits

    def preprocess_pipeline(self, verbose=True):
        """전체 전처리 파이프라인 V3"""
        print("=" * 80)
        print("  K-League Pass Prediction - 전처리 파이프라인 V3")
        print(f"  개선: Data Leakage 제거 + 시퀀스 모델링 (K={self.K})")
        print("=" * 80)
        print()

        # 1. 데이터 로딩
        data = self.load_data(verbose=verbose)

        # 2. 정렬 및 인덱싱
        data = self.sort_and_index(data, verbose=verbose)

        # 3. 기본 피처 생성
        data = self.create_basic_features(data, verbose=verbose)

        # 4. 타겟 레이블 추출 (Leakage 제거 전!)
        labels, ep_meta = self.extract_labels(data, verbose=verbose)

        # 5. 공격 팀 플래그 추가
        data = self.add_final_team_flag(data, ep_meta, verbose=verbose)

        # 6. Data Leakage 제거
        data = self.mask_target_leakage(data, verbose=verbose)

        # 7. 범주형 인코딩
        data = self.encode_categorical(data, verbose=verbose)

        # 8. 마지막 K개 이벤트 Wide format 변환
        X = self.create_lastK_wide_features(data, verbose=verbose)

        # 9. 메타데이터 및 레이블 병합
        X = self.merge_metadata_and_labels(X, ep_meta, labels, verbose=verbose)

        # 10. 모델 데이터 준비
        X_train_feat, y_train, game_ids = self.prepare_model_data(X, verbose=verbose)

        # 11. Train/Val Split
        splits = self.create_train_val_split(X_train_feat, y_train, game_ids, n_splits=5, verbose=verbose)

        print("=" * 80)
        print("✅ 전처리 V3 완료!")
        print("=" * 80)
        print(f"\n📊 최종 데이터:")
        print(f"   - 학습 샘플: {len(X_train_feat):,}")
        print(f"   - 피처 수: {X_train_feat.shape[1]}")
        print(f"   - K (시퀀스 길이): {self.K}")
        print(f"   - Fold 수: {len(splits)}")
        print("\n🚨 주요 개선:")
        print("   ✅ Data Leakage 제거 (end_x, end_y 마스킹)")
        print("   ✅ 시퀀스 모델링 (마지막 K개 이벤트)")
        print("   ✅ Wide format (시간 순서 보존)")
        print("   ✅ 5-Fold GroupKFold (안정적 검증)")
        print("=" * 80)

        return X_train_feat, y_train, game_ids, splits, X

    def save_preprocessor(self, filename='preprocessor_v3.pkl'):
        """전처리 객체 저장"""
        with open(filename, 'wb') as f:
            pickle.dump({
                'type_encoder': self.type_encoder,
                'result_encoder': self.result_encoder,
                'team_encoder': self.team_encoder,
                'K': self.K,
            }, f)
        print(f"✅ Preprocessor V3 저장: {filename}")

    def load_preprocessor(self, filename='preprocessor_v3.pkl'):
        """전처리 객체 로딩"""
        with open(filename, 'rb') as f:
            saved = pickle.load(f)
            self.type_encoder = saved['type_encoder']
            self.result_encoder = saved['result_encoder']
            self.team_encoder = saved['team_encoder']
            self.K = saved['K']
        print(f"Preprocessor V3 loaded: {filename}")


def main():
    """V3 테스트 실행"""
    print("\n" + "="*80)
    print("  전처리 V3 실행")
    print("="*80 + "\n")

    # Preprocessor 초기화 (K=20)
    preprocessor = DataPreprocessorV3(data_dir='./data', K=20)

    # 전처리 실행
    X_train, y_train, game_ids, splits, X_full = preprocessor.preprocess_pipeline(verbose=True)

    # Preprocessor 저장
    preprocessor.save_preprocessor('preprocessor_v3.pkl')

    # Train 데이터 저장
    train_mask = X_full['target_x'].notna()
    X_train_full = X_full[train_mask].copy()
    X_train_full.to_csv('processed_train_data_v3.csv', index=False)
    print(f"\n✅ 처리된 Train 데이터 저장: processed_train_data_v3.csv")

    # Test 데이터 저장 (추론용)
    X_test_full = X_full[~train_mask].copy()
    X_test_full.to_csv('processed_test_data_v3.csv', index=False)
    print(f"✅ 처리된 Test 데이터 저장: processed_test_data_v3.csv")

    # 비교
    print("\n" + "="*80)
    print("  V1 vs V3 비교")
    print("="*80)
    print("\nV1 (기존):")
    print("  - 피처 수: 54개")
    print("  - 방식: 마지막 1개 이벤트 + Aggregate")
    print("  - Data Leakage: ⚠️ 있음 (end_x, end_y 포함)")
    print("  - Validation: 0.93m (하지만 부정확)")
    print("  - Test: 24점대")

    print("\nV3 (개선):")
    print(f"  - 피처 수: {X_train.shape[1]}개")
    print(f"  - 방식: 마지막 {preprocessor.K}개 이벤트 + Wide format")
    print("  - Data Leakage: ✅ 제거됨")
    print("  - 예상 Validation: 1.5~2m (정상)")
    print("  - 예상 Test: 15~18점대 (즉시), 12~15점대 (튜닝 후)")

    print("\n" + "="*80)
    print("✅ 다음 단계:")
    print("   1. train_lightgbm_v3.py 실행 (5-Fold 앙상블)")
    print("   2. inference_v3.py 실행 (Test 추론)")
    print("   3. 제출 및 점수 확인")
    print("="*80)

    return X_train, y_train, game_ids, splits, preprocessor


if __name__ == "__main__":
    X_train, y_train, game_ids, splits, preprocessor = main()

