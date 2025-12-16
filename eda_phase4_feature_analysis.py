"""
K-League Pass Prediction - EDA Phase 4
피처 효과성 및 개선 방향 분석

목표:
1. 현재 피처들의 예측력 분석
2. 피처 간 상관관계 및 중복성 파악
3. 오류 패턴 분석 (어떤 상황에서 예측이 어려운가?)
4. 새로운 피처 아이디어 도출
5. 실용적인 피처 개선 방향 제시

작성일: 2025-12-16
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime
from collections import Counter, defaultdict
import pickle
import warnings
warnings.filterwarnings('ignore')

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
pd.set_option('display.float_format', lambda x: f'{x:.4f}')

class Phase4FeatureAnalyzer:
    def __init__(self, data_dir='./data'):
        self.data_dir = data_dir
        self.insights = []
        self.train_data = None
        self.processed_data = None

    def log_insight(self, text):
        """인사이트 로깅"""
        print(text)
        self.insights.append(text)

    def print_section(self, title, level=1):
        """섹션 구분 출력"""
        if level == 1:
            separator = "=" * 80
            self.log_insight(f"\n{separator}")
            self.log_insight(f"  {title}")
            self.log_insight(separator + "\n")
        elif level == 2:
            self.log_insight(f"\n{'─' * 60}")
            self.log_insight(f"[{title}]")
            self.log_insight('─' * 60)

    def load_data(self):
        """데이터 로딩"""
        self.log_insight("📊 데이터 로딩 중...")

        # 원본 데이터
        train_path = os.path.join(self.data_dir, 'train.csv')
        self.train_data = pd.read_csv(train_path)

        # 전처리된 데이터
        if os.path.exists('processed_train_data.csv'):
            self.processed_data = pd.read_csv('processed_train_data.csv')
            self.log_insight(f"✅ 전처리된 데이터 로딩: {self.processed_data.shape}")
        else:
            self.log_insight("⚠️  전처리된 데이터가 없습니다. preprocessing.py를 먼저 실행하세요.")

        self.log_insight(f"✅ 원본 데이터 로딩: {self.train_data.shape}\n")

    def analyze_baseline_performance(self):
        """베이스라인 성능 분석"""
        self.print_section("PHASE 4-1: 베이스라인 성능 분석", level=1)

        if self.processed_data is None:
            self.log_insight("⚠️  전처리된 데이터가 필요합니다.")
            return

        # 실제 타겟 값
        y_true_x = self.processed_data['end_x'].values
        y_true_y = self.processed_data['end_y'].values

        # 1. 시작 위치 = 도착 위치 (베이스라인)
        y_pred_x = self.processed_data['start_x'].values
        y_pred_y = self.processed_data['start_y'].values

        errors = np.sqrt((y_true_x - y_pred_x)**2 + (y_true_y - y_pred_y)**2)

        self.log_insight(f"📊 베이스라인 (start = end) 성능:")
        self.log_insight(f"  - 평균 오차: {errors.mean():.2f}m")
        self.log_insight(f"  - 중앙값 오차: {np.median(errors):.2f}m")
        self.log_insight(f"  - 표준편차: {errors.std():.2f}m")
        self.log_insight(f"  - 최소 오차: {errors.min():.2f}m")
        self.log_insight(f"  - 최대 오차: {errors.max():.2f}m")

        # 백분위수
        self.log_insight(f"\n📊 오차 분포 (백분위수):")
        for p in [25, 50, 75, 90, 95, 99]:
            self.log_insight(f"  - {p}%: {np.percentile(errors, p):.2f}m")

        # 2. 오차 범위별 비율
        self.log_insight(f"\n📊 오차 범위별 에피소드 비율:")
        ranges = [(0, 5), (5, 10), (10, 15), (15, 20), (20, 30), (30, 50), (50, 100)]
        for low, high in ranges:
            count = ((errors >= low) & (errors < high)).sum()
            pct = (count / len(errors)) * 100
            self.log_insight(f"  - {low:3d}m ~ {high:3d}m: {count:5,}개 ({pct:5.1f}%)")

    def analyze_feature_coverage(self):
        """피처 커버리지 분석 - 결측치 및 이상치"""
        self.print_section("PHASE 4-2: 피처 품질 분석", level=1)

        if self.processed_data is None:
            return

        self.print_section("2.1 결측치 현황", level=2)

        missing_stats = []
        for col in self.processed_data.columns:
            missing_count = self.processed_data[col].isna().sum()
            if missing_count > 0:
                missing_pct = (missing_count / len(self.processed_data)) * 100
                missing_stats.append({
                    'column': col,
                    'missing_count': missing_count,
                    'missing_pct': missing_pct
                })

        if missing_stats:
            missing_df = pd.DataFrame(missing_stats).sort_values('missing_pct', ascending=False)
            self.log_insight(f"📊 결측치가 있는 피처: {len(missing_df)}개")
            for _, row in missing_df.head(20).iterrows():
                self.log_insight(f"  - {row['column']:35s}: {row['missing_count']:6,} ({row['missing_pct']:5.1f}%)")
        else:
            self.log_insight("✅ 모든 피처에 결측치가 없습니다.")

        # 2.2 피처 분산 분석
        self.print_section("2.2 피처 분산 분석 (낮은 분산 = 정보량 적음)", level=2)

        numeric_cols = self.processed_data.select_dtypes(include=[np.number]).columns
        variance_stats = []

        for col in numeric_cols:
            if col not in ['end_x', 'end_y', 'game_id']:
                var = self.processed_data[col].var()
                std = self.processed_data[col].std()
                mean = self.processed_data[col].mean()
                cv = std / (abs(mean) + 1e-10)  # Coefficient of Variation

                variance_stats.append({
                    'column': col,
                    'variance': var,
                    'std': std,
                    'mean': mean,
                    'cv': cv
                })

        var_df = pd.DataFrame(variance_stats).sort_values('variance')

        self.log_insight(f"📊 분산이 가장 낮은 피처 Top 15:")
        for _, row in var_df.head(15).iterrows():
            self.log_insight(f"  - {row['column']:35s}: var={row['variance']:10.4f}, CV={row['cv']:6.2f}")

    def analyze_feature_correlation(self):
        """피처 간 상관관계 분석"""
        self.print_section("PHASE 4-3: 피처 상관관계 분석", level=1)

        if self.processed_data is None:
            return

        # 타겟과의 상관관계
        self.print_section("3.1 타겟 변수와의 상관관계", level=2)

        numeric_cols = self.processed_data.select_dtypes(include=[np.number]).columns
        feature_cols = [col for col in numeric_cols
                       if col not in ['end_x', 'end_y', 'game_id', 'episode_id']]

        corr_x = []
        corr_y = []

        for col in feature_cols:
            try:
                cx = self.processed_data[[col, 'end_x']].corr().iloc[0, 1]
                cy = self.processed_data[[col, 'end_y']].corr().iloc[0, 1]
                corr_x.append({'feature': col, 'corr': cx})
                corr_y.append({'feature': col, 'corr': cy})
            except:
                pass

        # end_x와 상관관계 높은 피처
        self.log_insight(f"📊 end_x와 상관관계 높은 피처 Top 20:")
        corr_x_df = pd.DataFrame(corr_x).sort_values('corr', key=abs, ascending=False)
        for i, row in corr_x_df.head(20).iterrows():
            self.log_insight(f"  {i+1:2d}. {row['feature']:35s}: {row['corr']:7.4f}")

        # end_y와 상관관계 높은 피처
        self.log_insight(f"\n📊 end_y와 상관관계 높은 피처 Top 20:")
        corr_y_df = pd.DataFrame(corr_y).sort_values('corr', key=abs, ascending=False)
        for i, row in corr_y_df.head(20).iterrows():
            self.log_insight(f"  {i+1:2d}. {row['feature']:35s}: {row['corr']:7.4f}")

        # 3.2 피처 간 다중공선성 분석
        self.print_section("3.2 피처 간 높은 상관관계 (다중공선성)", level=2)

        # 상위 30개 중요 피처만 분석 (계산 효율성)
        top_features_x = corr_x_df.head(30)['feature'].tolist()
        top_features_y = corr_y_df.head(30)['feature'].tolist()
        top_features = list(set(top_features_x + top_features_y))

        if len(top_features) > 2:
            corr_matrix = self.processed_data[top_features].corr()

            # 높은 상관관계 쌍 찾기 (|r| > 0.8)
            high_corr_pairs = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    corr_val = corr_matrix.iloc[i, j]
                    if abs(corr_val) > 0.8:
                        high_corr_pairs.append({
                            'feature1': corr_matrix.columns[i],
                            'feature2': corr_matrix.columns[j],
                            'correlation': corr_val
                        })

            if high_corr_pairs:
                self.log_insight(f"⚠️  높은 상관관계 피처 쌍 (|r| > 0.8): {len(high_corr_pairs)}개")
                high_corr_df = pd.DataFrame(high_corr_pairs).sort_values('correlation', key=abs, ascending=False)
                for i, row in high_corr_df.head(15).iterrows():
                    self.log_insight(f"  - {row['feature1']:30s} ↔ {row['feature2']:30s}: {row['correlation']:6.3f}")
            else:
                self.log_insight("✅ 다중공선성 문제 없음 (상위 피처 기준)")

    def analyze_error_patterns(self):
        """오류 패턴 분석 - 어떤 상황에서 예측이 어려운가?"""
        self.print_section("PHASE 4-4: 오류 패턴 분석", level=1)

        if self.processed_data is None:
            return

        # 베이스라인 오차 계산
        errors = np.sqrt(
            (self.processed_data['end_x'] - self.processed_data['start_x'])**2 +
            (self.processed_data['end_y'] - self.processed_data['start_y'])**2
        )
        self.processed_data['baseline_error'] = errors

        # 4.1 경기장 위치별 오차
        self.print_section("4.1 경기장 위치별 예측 난이도", level=2)

        # X축 구간별
        self.log_insight("📊 X축 위치별 평균 오차:")
        x_bins = [(0, 35, '수비진'), (35, 70, '중원'), (70, 105, '공격진')]
        for low, high, label in x_bins:
            mask = (self.processed_data['start_x'] >= low) & (self.processed_data['start_x'] < high)
            avg_error = errors[mask].mean()
            count = mask.sum()
            self.log_insight(f"  - {label:10s} ({low:3d}-{high:3d}m): {avg_error:6.2f}m (n={count:,})")

        # Y축 구간별
        self.log_insight("\n📊 Y축 위치별 평균 오차:")
        y_bins = [(0, 22.67, '좌측'), (22.67, 45.33, '중앙'), (45.33, 68, '우측')]
        for low, high, label in y_bins:
            mask = (self.processed_data['start_y'] >= low) & (self.processed_data['start_y'] < high)
            avg_error = errors[mask].mean()
            count = mask.sum()
            self.log_insight(f"  - {label:10s} ({low:5.2f}-{high:5.2f}m): {avg_error:6.2f}m (n={count:,})")

        # 4.2 에피소드 특성별 오차
        self.print_section("4.2 에피소드 특성별 예측 난이도", level=2)

        # 에피소드 길이별
        if 'episode_length' in self.processed_data.columns:
            self.log_insight("📊 에피소드 길이별 평균 오차:")
            length_bins = [(1, 5), (5, 10), (10, 20), (20, 30), (30, 50), (50, 100)]
            for low, high in length_bins:
                mask = (self.processed_data['episode_length'] >= low) & (self.processed_data['episode_length'] < high)
                if mask.sum() > 0:
                    avg_error = errors[mask].mean()
                    count = mask.sum()
                    self.log_insight(f"  - {low:3d} ~ {high:3d}개: {avg_error:6.2f}m (n={count:,})")

        # 이벤트 타입별
        if 'type_name' in self.train_data.columns:
            self.log_insight("\n📊 마지막 이벤트 타입별 평균 오차:")

            # 마지막 이벤트만 추출
            last_events = self.train_data.groupby('game_episode').tail(1).copy()

            # 에피소드별로 매칭
            merged = pd.merge(
                self.processed_data[['game_episode', 'baseline_error']],
                last_events[['game_episode', 'type_name']],
                on='game_episode',
                how='left'
            )

            type_errors = merged.groupby('type_name')['baseline_error'].agg(['mean', 'count'])
            type_errors = type_errors.sort_values('mean', ascending=False)

            for i, (type_name, row) in enumerate(type_errors.head(15).iterrows(), 1):
                if row['count'] >= 10:  # 최소 10개 이상인 경우만
                    self.log_insight(f"  {i:2d}. {type_name:30s}: {row['mean']:6.2f}m (n={int(row['count']):,})")

        # 4.3 특정 피처 값에 따른 오차
        self.print_section("4.3 주요 피처 값에 따른 예측 난이도", level=2)

        # 골문 거리별
        if 'distance_to_goal_start' in self.processed_data.columns:
            self.log_insight("📊 골문 거리별 평균 오차:")
            goal_bins = [(0, 20), (20, 40), (40, 60), (60, 80), (80, 150)]
            for low, high in goal_bins:
                mask = (self.processed_data['distance_to_goal_start'] >= low) & \
                       (self.processed_data['distance_to_goal_start'] < high)
                if mask.sum() > 0:
                    avg_error = errors[mask].mean()
                    count = mask.sum()
                    self.log_insight(f"  - {low:3d} ~ {high:3d}m: {avg_error:6.2f}m (n={count:,})")

        # 페널티 박스 내/외
        if 'in_penalty_area' in self.processed_data.columns:
            self.log_insight("\n📊 페널티 박스 내/외 평균 오차:")
            for val, label in [(1, '페널티 박스 내'), (0, '페널티 박스 외')]:
                mask = self.processed_data['in_penalty_area'] == val
                avg_error = errors[mask].mean()
                count = mask.sum()
                self.log_insight(f"  - {label:20s}: {avg_error:6.2f}m (n={count:,})")

    def analyze_feature_interactions(self):
        """피처 간 상호작용 분석"""
        self.print_section("PHASE 4-5: 피처 상호작용 분석", level=1)

        if self.processed_data is None:
            return

        errors = self.processed_data['baseline_error']

        # 5.1 위치 × 에피소드 길이
        self.print_section("5.1 위치 × 에피소드 길이", level=2)

        if 'episode_length' in self.processed_data.columns:
            self.log_insight("📊 공격진에서의 에피소드 길이별 오차:")
            mask_attack = self.processed_data['start_x'] >= 70

            for length_range, label in [((1, 10), '짧은 에피소드'),
                                       ((10, 30), '중간 에피소드'),
                                       ((30, 100), '긴 에피소드')]:
                low, high = length_range
                mask = mask_attack & \
                       (self.processed_data['episode_length'] >= low) & \
                       (self.processed_data['episode_length'] < high)

                if mask.sum() > 0:
                    avg_error = errors[mask].mean()
                    count = mask.sum()
                    self.log_insight(f"  - {label:20s} ({low:2d}-{high:3d}): {avg_error:6.2f}m (n={count:,})")

        # 5.2 속도 × 거리
        self.print_section("5.2 속도 × 이동 거리", level=2)

        if 'velocity' in self.processed_data.columns and 'distance' in self.processed_data.columns:
            # velocity가 유효한 경우만
            valid_velocity = self.processed_data['velocity'].notna() & \
                           (self.processed_data['velocity'] >= 0) & \
                           (self.processed_data['velocity'] < 100)

            if valid_velocity.sum() > 0:
                self.log_insight("📊 이동 속도 × 거리 조합별 오차:")

                for vel_range, vel_label in [((0, 5), '느린 속도'),
                                            ((5, 15), '중간 속도'),
                                            ((15, 100), '빠른 속도')]:
                    vel_low, vel_high = vel_range

                    for dist_range, dist_label in [((0, 10), '짧은 거리'),
                                                  ((10, 30), '긴 거리')]:
                        dist_low, dist_high = dist_range

                        mask = valid_velocity & \
                               (self.processed_data['velocity'] >= vel_low) & \
                               (self.processed_data['velocity'] < vel_high) & \
                               (self.processed_data['distance'] >= dist_low) & \
                               (self.processed_data['distance'] < dist_high)

                        if mask.sum() > 100:  # 최소 100개 이상
                            avg_error = errors[mask].mean()
                            count = mask.sum()
                            self.log_insight(f"  - {vel_label:15s} × {dist_label:15s}: {avg_error:6.2f}m (n={count:,})")

    def suggest_feature_improvements(self):
        """피처 개선 제안"""
        self.print_section("PHASE 4-6: 피처 개선 제안", level=1)

        suggestions = []

        # 1. 상관관계 분석 기반 제안
        suggestions.append({
            'category': '🎯 타겟 상관관계 개선',
            'suggestions': [
                '1. start_x/y의 비선형 변환 시도 (log, sqrt, polynomial)',
                '2. 골문 거리의 역수 또는 지수 변환',
                '3. 각도 피처의 sin/cos 변환',
                '4. 구간별 더미 변수 생성 (특정 위치에서의 행동 패턴)',
            ]
        })

        # 2. 다중공선성 제거
        suggestions.append({
            'category': '⚠️  다중공선성 제거',
            'suggestions': [
                '1. 고도로 상관된 피처 쌍 중 하나 제거',
                '2. PCA/LDA를 통한 차원 축소',
                '3. 파생 피처 대신 원본 피처 사용 고려',
                '4. Regularization (L1/L2) 강화',
            ]
        })

        # 3. 오류 패턴 기반 제안
        suggestions.append({
            'category': '🔍 오류 패턴 기반',
            'suggestions': [
                '1. 예측 어려운 위치(공격진, 페널티 박스)에 특화된 피처',
                '2. 이벤트 타입별 맞춤 피처 (Cross, Shot 등)',
                '3. 긴 에피소드를 위한 시퀀스 요약 피처',
                '4. 압박 상황 감지 피처 강화',
            ]
        })

        # 4. 새로운 피처 아이디어
        suggestions.append({
            'category': '💡 새로운 피처 아이디어',
            'suggestions': [
                '1. 선수 역할/포지션 기반 피처 (player_id 활용)',
                '2. 팀 전술 스타일 피처 (team_id 기반)',
                '3. 상대 팀 수비 밀집도 추정',
                '4. 시간대별 득점 확률 (경기 종료 임박)',
                '5. 연속 성공 패스 횟수',
                '6. 직전 N개 패스의 평균 각도/거리',
                '7. 에피소드 내 X/Y 좌표의 표준편차 (공간 활용도)',
                '8. 골키퍼 위치 추정 (골문 거리 + 각도)',
            ]
        })

        # 5. 모델링 개선
        suggestions.append({
            'category': '🚀 모델링 개선',
            'suggestions': [
                '1. Feature Selection (Recursive Feature Elimination)',
                '2. Feature Importance 기반 가중치 조정',
                '3. 위치별 개별 모델 학습 (앙상블)',
                '4. Stacking: 1단계 예측을 2단계 피처로 활용',
                '5. Cross-validation fold 수 조정',
            ]
        })

        # 출력
        for item in suggestions:
            self.log_insight(f"\n{item['category']}")
            for suggestion in item['suggestions']:
                self.log_insight(f"  {suggestion}")

    def analyze_low_variance_features(self):
        """낮은 분산 피처 심화 분석"""
        self.print_section("PHASE 4-7: 낮은 정보량 피처 상세 분석", level=1)

        if self.processed_data is None:
            return

        numeric_cols = self.processed_data.select_dtypes(include=[np.number]).columns
        feature_cols = [col for col in numeric_cols
                       if col not in ['end_x', 'end_y', 'game_id', 'episode_id']]

        low_info_features = []

        for col in feature_cols:
            unique_count = self.processed_data[col].nunique()
            total_count = len(self.processed_data)
            unique_ratio = unique_count / total_count

            # 고유값이 10개 이하이거나, 고유값 비율이 1% 이하
            if unique_count <= 10 or unique_ratio < 0.01:
                value_counts = self.processed_data[col].value_counts()
                low_info_features.append({
                    'feature': col,
                    'unique_count': unique_count,
                    'unique_ratio': unique_ratio,
                    'top_value': value_counts.index[0],
                    'top_value_count': value_counts.iloc[0],
                    'top_value_pct': (value_counts.iloc[0] / total_count) * 100
                })

        if low_info_features:
            self.log_insight(f"⚠️  낮은 정보량 피처: {len(low_info_features)}개\n")
            low_info_df = pd.DataFrame(low_info_features).sort_values('unique_ratio')

            for _, row in low_info_df.head(20).iterrows():
                self.log_insight(f"  - {row['feature']:35s}: {row['unique_count']:4d}개 고유값 "
                               f"({row['unique_ratio']*100:5.2f}%), "
                               f"최빈값={row['top_value']:.2f} ({row['top_value_pct']:.1f}%)")
        else:
            self.log_insight("✅ 모든 피처가 충분한 정보량을 가지고 있습니다.")

    def save_insights(self):
        """인사이트 저장"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'archive/EDA_Phase4_Feature_Analysis_{timestamp}.txt'

        os.makedirs('archive', exist_ok=True)

        with open(filename, 'w', encoding='utf-8') as f:
            f.write('\n'.join(self.insights))

        self.log_insight(f"\n✅ 인사이트 저장: {filename}")

    def run_full_analysis(self):
        """전체 분석 실행"""
        self.print_section("K-League Pass Prediction - Phase 4: 피처 효과성 분석", level=1)
        self.log_insight(f"분석 시작 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

        # 데이터 로딩
        self.load_data()

        if self.processed_data is None:
            self.log_insight("⚠️  전처리된 데이터가 없습니다.")
            self.log_insight("다음 명령을 실행하세요: python preprocessing.py")
            return

        # 분석 단계별 실행
        self.analyze_baseline_performance()
        self.analyze_feature_coverage()
        self.analyze_feature_correlation()
        self.analyze_error_patterns()
        self.analyze_feature_interactions()
        self.analyze_low_variance_features()
        self.suggest_feature_improvements()

        # 최종 요약
        self.print_section("분석 완료", level=1)
        self.log_insight(f"총 {len(self.insights)}개의 인사이트 생성")
        self.log_insight(f"분석 종료 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        # 저장
        self.save_insights()

        return self.insights


if __name__ == "__main__":
    analyzer = Phase4FeatureAnalyzer(data_dir='./data')
    insights = analyzer.run_full_analysis()

    print("\n" + "="*80)
    print("✅ Phase 4 분석 완료!")
    print("="*80)

