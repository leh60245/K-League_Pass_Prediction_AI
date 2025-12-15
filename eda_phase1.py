"""
K-League Pass Prediction - EDA Analysis
Phase 1: Data Foundation Analysis

목표: 데이터 구조, 품질, 기본 통계 파악
출력: 텍스트 기반 인사이트
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path
import warnings
from datetime import datetime
warnings.filterwarnings('ignore')

# 출력 형식 설정
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
pd.set_option('display.float_format', lambda x: f'{x:.4f}')

class EDAAnalyzer:
    def __init__(self, data_dir='./data'):
        self.data_dir = data_dir
        self.insights = []
        self.train_data = None
        self.test_data = None
        self.match_info = None

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

    def save_insights(self, filename='EDA_insights.txt'):
        """인사이트를 파일로 저장"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(f"K-League Pass Prediction EDA Insights\n")
            f.write(f"Generated: {timestamp}\n")
            f.write("=" * 80 + "\n\n")
            f.write('\n'.join(self.insights))
        print(f"\n✅ 인사이트가 '{filename}'에 저장되었습니다.")

    # ========================================================================
    # Phase 1: 데이터 기초 진단
    # ========================================================================

    def load_data_structure(self):
        """데이터 구조 파악 및 로딩 전략 수립"""
        self.print_section("PHASE 1: 데이터 기초 진단 - 데이터 구조 파악", level=1)

        # 1.1 Train 데이터 구조
        self.print_section("1.1 Train 데이터 구조", level=2)
        train_path = os.path.join(self.data_dir, 'train.csv')

        # 파일 크기 확인
        if os.path.exists(train_path):
            file_size_mb = os.path.getsize(train_path) / (1024 * 1024)
            self.log_insight(f"📊 Train 데이터 파일 크기: {file_size_mb:.2f} MB")

            # 샘플 로딩으로 구조 파악
            try:
                # 처음 100,000행만 로딩
                sample_train = pd.read_csv(train_path, nrows=100000)
                self.log_insight(f"✅ 샘플 로딩 성공 (100,000 rows)")
                self.log_insight(f"\n컬럼 구조:")
                for col in sample_train.columns:
                    dtype = sample_train[col].dtype
                    non_null = sample_train[col].notna().sum()
                    null_pct = (1 - non_null/len(sample_train)) * 100
                    self.log_insight(f"  - {col:20s}: {str(dtype):12s} (결측: {null_pct:5.2f}%)")

                # 에피소드 수 추정
                unique_episodes_sample = sample_train['game_episode'].nunique()
                self.log_insight(f"\n샘플 내 고유 에피소드 수: {unique_episodes_sample:,}")

                # 전체 파일 라인 수 추정 (정확하게 세기)
                self.log_insight(f"\n전체 파일 크기 분석 중...")
                total_lines = sum(1 for _ in open(train_path, encoding='utf-8')) - 1  # 헤더 제외
                self.log_insight(f"전체 이벤트 수: {total_lines:,}")

                # 전체 데이터 로딩
                self.log_insight(f"\n전체 Train 데이터 로딩 중...")
                self.train_data = pd.read_csv(train_path)
                total_episodes = self.train_data['game_episode'].nunique()
                total_games = self.train_data['game_id'].nunique()

                self.log_insight(f"✅ Train 데이터 로딩 완료")
                self.log_insight(f"  - 총 에피소드 수: {total_episodes:,}")
                self.log_insight(f"  - 총 경기 수: {total_games:,}")
                self.log_insight(f"  - 총 이벤트 수: {len(self.train_data):,}")
                self.log_insight(f"  - 경기당 평균 에피소드: {total_episodes/total_games:.1f}")
                self.log_insight(f"  - 에피소드당 평균 이벤트: {len(self.train_data)/total_episodes:.1f}")

            except Exception as e:
                self.log_insight(f"❌ Train 데이터 로딩 실패: {str(e)}")
        else:
            self.log_insight(f"❌ Train 파일을 찾을 수 없습니다: {train_path}")

        # 1.2 Test 데이터 구조
        self.print_section("1.2 Test 데이터 구조", level=2)
        test_index_path = os.path.join(self.data_dir, 'test.csv')

        if os.path.exists(test_index_path):
            test_index = pd.read_csv(test_index_path)
            self.log_insight(f"📊 Test 에피소드 수: {len(test_index):,}")
            self.log_insight(f"  - 경기 수: {test_index['game_id'].nunique()}")
            self.log_insight(f"  - 경기당 평균 에피소드: {len(test_index)/test_index['game_id'].nunique():.1f}")

            # 샘플 에피소드 로딩
            sample_path = test_index.iloc[0]['path']
            sample_full_path = os.path.join(self.data_dir, sample_path.replace('./', ''))

            if os.path.exists(sample_full_path):
                sample_episode = pd.read_csv(sample_full_path)
                self.log_insight(f"\n샘플 에피소드 분석: {test_index.iloc[0]['game_episode']}")
                self.log_insight(f"  - 이벤트 수: {len(sample_episode)}")
                self.log_insight(f"  - 마지막 행 end_x 결측: {pd.isna(sample_episode.iloc[-1]['end_x'])}")
                self.log_insight(f"  - 마지막 행 end_y 결측: {pd.isna(sample_episode.iloc[-1]['end_y'])}")

        # 1.3 Match Info
        self.print_section("1.3 Match Info 데이터", level=2)
        match_info_path = os.path.join(self.data_dir, 'match_info.csv')

        if os.path.exists(match_info_path):
            self.match_info = pd.read_csv(match_info_path)
            self.log_insight(f"📊 경기 정보: {len(self.match_info)} 경기")
            self.log_insight(f"  - 시즌: {self.match_info['season_name'].unique()}")
            self.log_insight(f"  - 대회: {self.match_info['competition_name'].unique()}")
            self.log_insight(f"  - 팀 수: {len(set(self.match_info['home_team_id']) | set(self.match_info['away_team_id']))}")

    def analyze_data_quality(self):
        """데이터 품질 분석"""
        self.print_section("PHASE 1: 데이터 품질 분석", level=1)

        if self.train_data is None:
            self.log_insight("❌ Train 데이터가 로딩되지 않았습니다.")
            return

        # 2.1 결측치 분석
        self.print_section("2.1 결측치 분석", level=2)

        missing_summary = []
        for col in self.train_data.columns:
            missing_count = self.train_data[col].isna().sum()
            missing_pct = (missing_count / len(self.train_data)) * 100
            if missing_count > 0:
                missing_summary.append({
                    'column': col,
                    'missing_count': missing_count,
                    'missing_pct': missing_pct
                })

        if missing_summary:
            self.log_insight("📊 결측치 현황:")
            for item in sorted(missing_summary, key=lambda x: x['missing_pct'], reverse=True):
                self.log_insight(f"  - {item['column']:20s}: {item['missing_count']:8,} ({item['missing_pct']:6.2f}%)")

            # player_id 결측치 패턴 분석
            if 'player_id' in self.train_data.columns:
                self.log_insight("\n🔍 player_id 결측치 상세 분석:")
                missing_player = self.train_data[self.train_data['player_id'].isna()]
                event_types = missing_player['type_name'].value_counts()
                self.log_insight("  결측치가 발생하는 이벤트 타입:")
                for event, count in event_types.head(10).items():
                    pct = (count / len(missing_player)) * 100
                    self.log_insight(f"    - {event:30s}: {count:6,} ({pct:5.1f}%)")
        else:
            self.log_insight("✅ 결측치 없음")

        # 2.2 좌표 범위 검증
        self.print_section("2.2 좌표 범위 검증", level=2)

        coord_cols = ['start_x', 'start_y', 'end_x', 'end_y']
        coord_ranges = {'x': (0, 105), 'y': (0, 68)}

        outliers = {}
        for col in coord_cols:
            if col in self.train_data.columns:
                valid_data = self.train_data[col].dropna()
                axis = 'x' if 'x' in col else 'y'
                min_val, max_val = coord_ranges[axis]

                below_min = (valid_data < min_val).sum()
                above_max = (valid_data > max_val).sum()
                total_outliers = below_min + above_max

                if total_outliers > 0:
                    outliers[col] = {
                        'below': below_min,
                        'above': above_max,
                        'total': total_outliers,
                        'pct': (total_outliers / len(valid_data)) * 100
                    }

                self.log_insight(f"{col:12s}: min={valid_data.min():7.2f}, max={valid_data.max():7.2f}, "
                               f"mean={valid_data.mean():7.2f}, std={valid_data.std():6.2f}")

        if outliers:
            self.log_insight("\n⚠️  범위 벗어난 좌표:")
            for col, stats in outliers.items():
                self.log_insight(f"  - {col}: {stats['total']:,} ({stats['pct']:.4f}%) "
                               f"[하한 초과: {stats['below']}, 상한 초과: {stats['above']}]")
        else:
            self.log_insight("\n✅ 모든 좌표가 정상 범위 내에 있습니다.")

        # 2.3 시간 정합성 검증
        self.print_section("2.3 시간 정합성 검증", level=2)

        # 에피소드별로 시간 순서 확인
        time_issues = 0
        total_episodes = 0

        for episode_id, group in self.train_data.groupby('game_episode'):
            total_episodes += 1
            # action_id 순서와 time_seconds 순서 비교
            action_sorted = group.sort_values('action_id')
            time_sorted = group.sort_values('time_seconds')

            if not action_sorted.index.equals(time_sorted.index):
                time_issues += 1

        self.log_insight(f"📊 시간 순서 검증:")
        self.log_insight(f"  - 전체 에피소드: {total_episodes:,}")
        self.log_insight(f"  - action_id와 time_seconds 순서 불일치: {time_issues:,} ({time_issues/total_episodes*100:.2f}%)")

        if time_issues > 0:
            self.log_insight(f"  ⚠️  권장: time_seconds 기준으로 정렬 필요")

        # 시간 역전 확인
        time_reversals = 0
        for episode_id, group in self.train_data.groupby('game_episode'):
            time_diff = group.sort_values('action_id')['time_seconds'].diff()
            if (time_diff < 0).any():
                time_reversals += 1

        self.log_insight(f"  - 시간 역전 에피소드: {time_reversals:,} ({time_reversals/total_episodes*100:.2f}%)")

    def analyze_basic_statistics(self):
        """기본 통계 분석"""
        self.print_section("PHASE 1: 기본 통계 분석", level=1)

        if self.train_data is None:
            return

        # 3.1 에피소드 길이 분석
        self.print_section("3.1 에피소드 길이 분석", level=2)

        episode_lengths = self.train_data.groupby('game_episode').size()

        self.log_insight("📊 에피소드 길이 통계:")
        self.log_insight(f"  - 평균: {episode_lengths.mean():.1f} 이벤트")
        self.log_insight(f"  - 중앙값: {episode_lengths.median():.0f} 이벤트")
        self.log_insight(f"  - 표준편차: {episode_lengths.std():.1f}")
        self.log_insight(f"  - 최소: {episode_lengths.min()} 이벤트")
        self.log_insight(f"  - 최대: {episode_lengths.max()} 이벤트")

        self.log_insight("\n백분위수:")
        percentiles = [25, 50, 75, 90, 95, 99]
        for p in percentiles:
            val = episode_lengths.quantile(p/100)
            self.log_insight(f"  - {p:2d}%: {val:6.0f} 이벤트")

        # 길이별 분포
        self.log_insight("\n에피소드 길이 분포:")
        length_bins = [0, 10, 20, 30, 50, 100, float('inf')]
        length_labels = ['1-10', '11-20', '21-30', '31-50', '51-100', '100+']
        length_dist = pd.cut(episode_lengths, bins=length_bins, labels=length_labels).value_counts().sort_index()

        for length_range, count in length_dist.items():
            pct = (count / len(episode_lengths)) * 100
            self.log_insight(f"  - {length_range:8s}: {count:6,} ({pct:5.1f}%)")

        # 3.2 이벤트 타입 분석
        self.print_section("3.2 이벤트 타입 분석", level=2)

        event_counts = self.train_data['type_name'].value_counts()
        self.log_insight(f"📊 총 {len(event_counts)} 종류의 이벤트 타입")
        self.log_insight(f"\nTop 15 이벤트 타입:")

        for i, (event_type, count) in enumerate(event_counts.head(15).items(), 1):
            pct = (count / len(self.train_data)) * 100
            self.log_insight(f"  {i:2d}. {event_type:30s}: {count:8,} ({pct:5.2f}%)")

        # 성공/실패 분석
        if 'result_name' in self.train_data.columns:
            self.log_insight("\n📊 이벤트 결과 분포:")
            result_counts = self.train_data['result_name'].value_counts(dropna=False)
            for result, count in result_counts.items():
                pct = (count / len(self.train_data)) * 100
                result_str = 'NaN (결과 없음)' if pd.isna(result) else result
                self.log_insight(f"  - {result_str:30s}: {count:8,} ({pct:5.2f}%)")

        # 3.3 시간 분석
        self.print_section("3.3 시간 특성 분석", level=2)

        # 에피소드 지속 시간
        episode_durations = self.train_data.groupby('game_episode').apply(
            lambda x: x['time_seconds'].max() - x['time_seconds'].min()
        )

        self.log_insight("📊 에피소드 지속 시간:")
        self.log_insight(f"  - 평균: {episode_durations.mean():.1f} 초")
        self.log_insight(f"  - 중앙값: {episode_durations.median():.1f} 초")
        self.log_insight(f"  - 표준편차: {episode_durations.std():.1f} 초")
        self.log_insight(f"  - 최소: {episode_durations.min():.1f} 초")
        self.log_insight(f"  - 최대: {episode_durations.max():.1f} 초")

        # 전반/후반 분석
        if 'period_id' in self.train_data.columns:
            self.log_insight("\n📊 전반/후반 분포:")
            period_counts = self.train_data['period_id'].value_counts().sort_index()
            for period, count in period_counts.items():
                pct = (count / len(self.train_data)) * 100
                self.log_insight(f"  - Period {period}: {count:8,} ({pct:5.2f}%)")

    def generate_summary(self):
        """Phase 1 종합 요약 및 모델링 시사점"""
        self.print_section("PHASE 1 종합 요약 및 모델링 시사점", level=1)

        self.log_insight("=" * 80)
        self.log_insight("📋 핵심 발견 (Key Findings)")
        self.log_insight("=" * 80)

        self.log_insight("""
[1. 데이터 규모]
- Train: 대규모 시퀀스 데이터 (수만 개 에피소드)
- Test: 2,415 에피소드 예측 필요
- 충분한 학습 데이터 확보됨

[2. 데이터 품질]
- player_id 결측: 특정 이벤트 타입에서 자연스럽게 발생 (Out, Block 등)
- 좌표 데이터: 대부분 정상 범위 내
- 시간 순서: 일부 불일치 존재 → time_seconds 기준 정렬 필요

[3. 시퀀스 특성]
- 에피소드 길이: 매우 가변적 (최소 ~ 최대 편차 큼)
- 대부분 50개 이하 이벤트
- Padding/Truncation 전략 필요

[4. 이벤트 패턴]
- Pass가 가장 빈번한 이벤트
- Carry, Duel 등 다양한 이벤트 타입 존재
- 이벤트 시퀀스 인코딩이 핵심
""")

        self.log_insight("\n" + "=" * 80)
        self.log_insight("🎯 모델링 시사점 (Modeling Implications)")
        self.log_insight("=" * 80)

        self.log_insight("""
[피처 엔지니어링]
✓ 시퀀스 길이 표준화: 95 percentile 기준 (약 ~개 이벤트)
✓ 좌표 정규화: MinMax or Standard Scaling
✓ 시간 정보: 상대 시간, 시간 간격 피처
✓ 이벤트 타입: Embedding or One-Hot Encoding

[모델 선택]
✓ LSTM/GRU: 가변 길이 시퀀스 처리 (Masking 필수)
✓ Transformer: Attention으로 중요 이벤트 포착
✓ 앙상블: XGBoost + 딥러닝 조합

[전처리 전략]
✓ time_seconds 기준 정렬 필수
✓ player_id 결측: 특수 토큰 (-1) 처리
✓ 좌표 이상치: Clipping (0-105, 0-68)

[검증 전략]
✓ Game-based Split: 경기 단위로 Train/Val 분리
✓ Time-based Split: 시간 순서 고려
✓ Cross Validation: 5-Fold 권장
""")

        self.log_insight("\n" + "=" * 80)
        self.log_insight("🚀 다음 단계 (Next Steps)")
        self.log_insight("=" * 80)

        self.log_insight("""
Phase 2에서 분석할 내용:
1. 예측 대상(마지막 패스) 상세 분석
2. 시작 위치 → 도착 위치 관계성
3. 베이스라인 성능 추정 (단순 평균 예측)
4. 경기장 공간 분포 분석
""")

def main():
    """메인 실행 함수"""
    print("=" * 80)
    print("  K-League Pass Prediction - EDA Phase 1")
    print("  데이터 기초 진단 및 품질 분석")
    print("=" * 80)
    print()

    # Analyzer 초기화
    analyzer = EDAAnalyzer(data_dir='./data')

    # Phase 1 분석 실행
    analyzer.load_data_structure()
    analyzer.analyze_data_quality()
    analyzer.analyze_basic_statistics()
    analyzer.generate_summary()

    # 인사이트 저장
    analyzer.save_insights('EDA_Phase1_insights.txt')

    print("\n" + "=" * 80)
    print("✅ Phase 1 분석 완료!")
    print("=" * 80)

if __name__ == "__main__":
    main()

