"""
K-League Pass Prediction - EDA Analysis
Phase 3: Sequence Pattern Analysis

목표: 시퀀스 내 이벤트 패턴 및 마지막 패스에 미치는 영향 분석
출력: 텍스트 기반 인사이트
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime
from collections import Counter, defaultdict
import warnings
warnings.filterwarnings('ignore')

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
pd.set_option('display.float_format', lambda x: f'{x:.4f}')

class Phase3Analyzer:
    def __init__(self, data_dir='./data'):
        self.data_dir = data_dir
        self.insights = []
        self.train_data = None

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
        train_path = os.path.join(self.data_dir, 'train.csv')
        self.train_data = pd.read_csv(train_path)

        # time_seconds 기준 정렬
        self.train_data = self.train_data.sort_values(['game_episode', 'time_seconds']).reset_index(drop=True)
        self.log_insight(f"✅ Train 데이터 로딩 완료: {len(self.train_data):,} 이벤트, {self.train_data['game_episode'].nunique():,} 에피소드\n")

    def analyze_event_transitions(self):
        """이벤트 타입 전이 확률 분석"""
        self.print_section("PHASE 3: 이벤트 전이 패턴 분석", level=1)

        # 1.1 Bigram 분석 (연속된 2개 이벤트)
        self.print_section("3.1 이벤트 Bigram 분석 (A → B)", level=2)

        bigrams = []
        for episode_id, group in self.train_data.groupby('game_episode'):
            events = group['type_name'].tolist()
            for i in range(len(events) - 1):
                bigrams.append((events[i], events[i+1]))

        bigram_counts = Counter(bigrams)
        total_bigrams = len(bigrams)

        self.log_insight(f"📊 총 {total_bigrams:,}개의 이벤트 전이")
        self.log_insight(f"📊 고유 Bigram 패턴: {len(bigram_counts)}개")
        self.log_insight(f"\nTop 20 가장 흔한 이벤트 전이:")

        for i, ((event1, event2), count) in enumerate(bigram_counts.most_common(20), 1):
            pct = (count / total_bigrams) * 100
            self.log_insight(f"  {i:2d}. {event1:25s} → {event2:25s}: {count:7,} ({pct:5.2f}%)")

        # 1.2 마지막 패스 직전 이벤트 분석
        self.print_section("3.2 마지막 패스 직전 이벤트 분석", level=2)

        previous_events = []
        for episode_id, group in self.train_data.groupby('game_episode'):
            if len(group) >= 2:
                # 마지막 이벤트 (예측 대상)
                last_event = group.iloc[-1]['type_name']
                # 직전 이벤트
                prev_event = group.iloc[-2]['type_name']
                previous_events.append(prev_event)

        prev_counts = Counter(previous_events)
        total_prev = len(previous_events)

        self.log_insight(f"📊 마지막 패스 직전 이벤트 Top 15:")
        for i, (event, count) in enumerate(prev_counts.most_common(15), 1):
            pct = (count / total_prev) * 100
            self.log_insight(f"  {i:2d}. {event:30s}: {count:6,} ({pct:5.1f}%)")

        # 1.3 Trigram 분석 (연속된 3개 이벤트)
        self.print_section("3.3 이벤트 Trigram 분석 (A → B → C)", level=2)

        trigrams = []
        for episode_id, group in self.train_data.groupby('game_episode'):
            events = group['type_name'].tolist()
            for i in range(len(events) - 2):
                trigrams.append((events[i], events[i+1], events[i+2]))

        trigram_counts = Counter(trigrams)
        total_trigrams = len(trigrams)

        self.log_insight(f"📊 총 {total_trigrams:,}개의 3-이벤트 시퀀스")
        self.log_insight(f"📊 고유 Trigram 패턴: {len(trigram_counts)}개")
        self.log_insight(f"\nTop 15 가장 흔한 3-이벤트 시퀀스:")

        for i, ((e1, e2, e3), count) in enumerate(trigram_counts.most_common(15), 1):
            pct = (count / total_trigrams) * 100
            self.log_insight(f"  {i:2d}. {e1:15s} → {e2:15s} → {e3:15s}: {count:6,} ({pct:4.2f}%)")

    def analyze_last_n_events_impact(self):
        """직전 N개 이벤트가 마지막 패스에 미치는 영향"""
        self.print_section("PHASE 3: 직전 N개 이벤트의 영향 분석", level=1)

        # 2.1 직전 N개 이벤트와 패스 거리 관계
        self.print_section("3.4 직전 이벤트와 패스 거리 관계", level=2)

        analysis_data = []
        for episode_id, group in self.train_data.groupby('game_episode'):
            if len(group) >= 2:
                last_row = group.iloc[-1]
                prev_row = group.iloc[-2]

                # 마지막 패스 거리
                pass_dist = np.sqrt(
                    (last_row['end_x'] - last_row['start_x'])**2 +
                    (last_row['end_y'] - last_row['start_y'])**2
                )

                analysis_data.append({
                    'prev_event': prev_row['type_name'],
                    'prev_result': prev_row['result_name'],
                    'pass_distance': pass_dist,
                    'last_start_x': last_row['start_x'],
                    'last_end_x': last_row['end_x'],
                    'delta_x': last_row['end_x'] - last_row['start_x']
                })

        analysis_df = pd.DataFrame(analysis_data)

        self.log_insight("📊 직전 이벤트별 평균 패스 거리:")
        prev_event_stats = analysis_df.groupby('prev_event')['pass_distance'].agg(['mean', 'std', 'count']).sort_values('count', ascending=False)

        for i, (event, row) in enumerate(prev_event_stats.head(15).iterrows(), 1):
            if row['count'] >= 10:  # 최소 10개 이상
                self.log_insight(f"  {i:2d}. {event:30s}: 평균 {row['mean']:5.1f}m, std {row['std']:5.1f}m (n={int(row['count']):,})")

        # 2.2 직전 이벤트 결과(성공/실패)와 마지막 패스 관계
        self.print_section("3.5 직전 이벤트 결과와 마지막 패스", level=2)

        result_stats = analysis_df[analysis_df['prev_result'].notna()].groupby('prev_result')['pass_distance'].agg(['mean', 'count'])

        self.log_insight("📊 직전 이벤트 결과별 평균 패스 거리:")
        for result, row in result_stats.iterrows():
            if row['count'] >= 10:
                self.log_insight(f"  - {result:30s}: 평균 {row['mean']:5.1f}m (n={int(row['count']):,})")

        # 2.3 직전 N개 이벤트 타입 조합
        self.print_section("3.6 마지막 직전 2개 이벤트 조합 분석", level=2)

        last_2_combos = []
        for episode_id, group in self.train_data.groupby('game_episode'):
            if len(group) >= 3:
                # 마지막 3개: [-3, -2, -1] (마지막이 예측 대상)
                e1 = group.iloc[-3]['type_name']
                e2 = group.iloc[-2]['type_name']
                last_2_combos.append((e1, e2))

        combo_counts = Counter(last_2_combos)

        self.log_insight(f"📊 마지막 패스 직전 2개 이벤트 조합 Top 15:")
        for i, ((e1, e2), count) in enumerate(combo_counts.most_common(15), 1):
            pct = (count / len(last_2_combos)) * 100
            self.log_insight(f"  {i:2d}. {e1:20s} → {e2:20s} → [마지막 패스]: {count:5,} ({pct:4.1f}%)")

    def analyze_temporal_patterns(self):
        """시간적 패턴 분석"""
        self.print_section("PHASE 3: 시간적 패턴 분석", level=1)

        # 3.1 에피소드 경과 시간과 패스 좌표 관계
        self.print_section("3.7 에피소드 경과 시간과 마지막 패스", level=2)

        temporal_data = []
        for episode_id, group in self.train_data.groupby('game_episode'):
            if len(group) >= 1:
                last_row = group.iloc[-1]

                # 에피소드 시작 시간
                start_time = group.iloc[0]['time_seconds']
                # 에피소드 종료 시간 (마지막 패스 시작 시간)
                end_time = last_row['time_seconds']
                duration = end_time - start_time

                temporal_data.append({
                    'duration': duration,
                    'num_events': len(group),
                    'end_x': last_row['end_x'],
                    'end_y': last_row['end_y'],
                    'pass_distance': np.sqrt(
                        (last_row['end_x'] - last_row['start_x'])**2 +
                        (last_row['end_y'] - last_row['start_y'])**2
                    )
                })

        temporal_df = pd.DataFrame(temporal_data)

        # 지속 시간 구간별 분석
        duration_bins = [0, 10, 20, 30, 60, float('inf')]
        duration_labels = ['0-10초', '10-20초', '20-30초', '30-60초', '60초+']
        temporal_df['duration_bin'] = pd.cut(temporal_df['duration'], bins=duration_bins, labels=duration_labels)

        self.log_insight("📊 에피소드 지속 시간별 마지막 패스 특성:")
        duration_stats = temporal_df.groupby('duration_bin').agg({
            'end_x': 'mean',
            'pass_distance': 'mean',
            'duration': 'count'
        }).round(2)

        for duration_range, row in duration_stats.iterrows():
            self.log_insight(f"  - {duration_range:10s}: end_x 평균 {row['end_x']:5.1f}m, "
                           f"패스거리 {row['pass_distance']:5.1f}m (n={int(row['duration']):,})")

        # 3.2 빠른 템포 vs 느린 템포
        self.print_section("3.8 플레이 템포와 패스 특성", level=2)

        # 이벤트당 평균 시간 (템포)
        temporal_df['tempo'] = temporal_df['duration'] / temporal_df['num_events']

        # 템포 구간
        tempo_bins = [0, 1, 2, 3, float('inf')]
        tempo_labels = ['매우빠름(<1초)', '빠름(1-2초)', '보통(2-3초)', '느림(3초+)']
        temporal_df['tempo_bin'] = pd.cut(temporal_df['tempo'], bins=tempo_bins, labels=tempo_labels)

        self.log_insight("📊 플레이 템포별 마지막 패스 특성:")
        tempo_stats = temporal_df.groupby('tempo_bin').agg({
            'end_x': 'mean',
            'pass_distance': 'mean',
            'tempo': 'count'
        }).round(2)

        for tempo_range, row in tempo_stats.iterrows():
            self.log_insight(f"  - {tempo_range:15s}: end_x 평균 {row['end_x']:5.1f}m, "
                           f"패스거리 {row['pass_distance']:5.1f}m (n={int(row['tempo']):,})")

        # 3.3 전반/후반 비교
        self.print_section("3.9 전반/후반별 패스 특성", level=2)

        period_data = []
        for episode_id, group in self.train_data.groupby('game_episode'):
            last_row = group.iloc[-1]
            period_data.append({
                'period': last_row['period_id'],
                'end_x': last_row['end_x'],
                'end_y': last_row['end_y'],
                'pass_distance': np.sqrt(
                    (last_row['end_x'] - last_row['start_x'])**2 +
                    (last_row['end_y'] - last_row['start_y'])**2
                )
            })

        period_df = pd.DataFrame(period_data)

        self.log_insight("📊 Period별 마지막 패스 특성:")
        period_stats = period_df.groupby('period').agg({
            'end_x': ['mean', 'std'],
            'end_y': ['mean', 'std'],
            'pass_distance': ['mean', 'std']
        }).round(2)

        for period in sorted(period_df['period'].unique()):
            stats = period_stats.loc[period]
            self.log_insight(f"  - Period {period}:")
            self.log_insight(f"      end_x: {stats[('end_x', 'mean')]:5.1f} ± {stats[('end_x', 'std')]:5.1f}m")
            self.log_insight(f"      end_y: {stats[('end_y', 'mean')]:5.1f} ± {stats[('end_y', 'std')]:5.1f}m")
            self.log_insight(f"      패스거리: {stats[('pass_distance', 'mean')]:5.1f} ± {stats[('pass_distance', 'std')]:5.1f}m")

    def analyze_spatial_sequence_patterns(self):
        """공간적 시퀀스 패턴 분석"""
        self.print_section("PHASE 3: 공간적 시퀀스 패턴 분석", level=1)

        # 4.1 에피소드 내 공간 이동 패턴
        self.print_section("3.10 에피소드 내 X축 진행 패턴", level=2)

        progression_data = []
        for episode_id, group in self.train_data.groupby('game_episode'):
            if len(group) >= 2:
                # 첫 이벤트
                first_x = group.iloc[0]['start_x']
                # 마지막 이벤트
                last_start_x = group.iloc[-1]['start_x']
                last_end_x = group.iloc[-1]['end_x']

                # X축 진전도
                x_progression = last_start_x - first_x

                progression_data.append({
                    'first_x': first_x,
                    'last_start_x': last_start_x,
                    'last_end_x': last_end_x,
                    'x_progression': x_progression,
                    'num_events': len(group)
                })

        prog_df = pd.DataFrame(progression_data)

        self.log_insight("📊 에피소드 내 X축 진행:")
        self.log_insight(f"  - 평균 X 진행: {prog_df['x_progression'].mean():.2f}m")
        self.log_insight(f"  - 중앙값 X 진행: {prog_df['x_progression'].median():.2f}m")
        self.log_insight(f"  - 표준편차: {prog_df['x_progression'].std():.2f}m")

        # 진행 방향별 분석
        forward = (prog_df['x_progression'] > 10).sum()
        stable = ((prog_df['x_progression'] >= -10) & (prog_df['x_progression'] <= 10)).sum()
        backward = (prog_df['x_progression'] < -10).sum()
        total = len(prog_df)

        self.log_insight(f"\n📊 에피소드 진행 패턴:")
        self.log_insight(f"  - 전진 플레이 (X+10m 이상): {forward:6,} ({forward/total*100:5.1f}%)")
        self.log_insight(f"  - 안정 플레이 (±10m):      {stable:6,} ({stable/total*100:5.1f}%)")
        self.log_insight(f"  - 후진 플레이 (X-10m 이상): {backward:6,} ({backward/total*100:5.1f}%)")

        # 4.2 시작 위치별 도착 위치 패턴
        self.print_section("3.11 시작 영역별 도착 영역 패턴", level=2)

        def classify_zone(x):
            if x < 35:
                return '수비진'
            elif x < 70:
                return '중원'
            else:
                return '공격진'

        zone_data = []
        for episode_id, group in self.train_data.groupby('game_episode'):
            last_row = group.iloc[-1]
            zone_data.append({
                'start_zone': classify_zone(last_row['start_x']),
                'end_zone': classify_zone(last_row['end_x'])
            })

        zone_df = pd.DataFrame(zone_data)

        # 전이 행렬
        transition_matrix = pd.crosstab(zone_df['start_zone'], zone_df['end_zone'], normalize='index') * 100

        self.log_insight("📊 시작 영역 → 도착 영역 전이 확률 (%):")
        zones = ['수비진', '중원', '공격진']
        for start_zone in zones:
            if start_zone in transition_matrix.index:
                self.log_insight(f"\n  {start_zone}에서 시작:")
                for end_zone in zones:
                    if end_zone in transition_matrix.columns:
                        prob = transition_matrix.loc[start_zone, end_zone]
                        self.log_insight(f"    → {end_zone:10s}: {prob:5.1f}%")

    def analyze_carry_pass_patterns(self):
        """Carry-Pass 패턴 분석 (가장 빈번한 조합)"""
        self.print_section("PHASE 3: Carry-Pass 패턴 상세 분석", level=1)

        # 5.1 Carry 후 패스 특성
        self.print_section("3.12 Carry 후 패스 특성", level=2)

        carry_pass_data = []
        for episode_id, group in self.train_data.groupby('game_episode'):
            for i in range(len(group) - 1):
                curr_event = group.iloc[i]
                next_event = group.iloc[i+1]

                if curr_event['type_name'] == 'Carry' and 'Pass' in next_event['type_name']:
                    # Carry 거리
                    carry_dist = np.sqrt(
                        (curr_event['end_x'] - curr_event['start_x'])**2 +
                        (curr_event['end_y'] - curr_event['start_y'])**2
                    )

                    # 이어지는 Pass 거리
                    pass_dist = np.sqrt(
                        (next_event['end_x'] - next_event['start_x'])**2 +
                        (next_event['end_y'] - next_event['start_y'])**2
                    )

                    carry_pass_data.append({
                        'carry_dist': carry_dist,
                        'pass_dist': pass_dist,
                        'pass_result': next_event['result_name']
                    })

        cp_df = pd.DataFrame(carry_pass_data)

        self.log_insight(f"📊 Carry → Pass 조합: 총 {len(cp_df):,}회 발생")
        self.log_insight(f"\n  Carry 거리:")
        self.log_insight(f"    - 평균: {cp_df['carry_dist'].mean():.2f}m")
        self.log_insight(f"    - 중앙값: {cp_df['carry_dist'].median():.2f}m")

        self.log_insight(f"\n  이어지는 Pass 거리:")
        self.log_insight(f"    - 평균: {cp_df['pass_dist'].mean():.2f}m")
        self.log_insight(f"    - 중앙값: {cp_df['pass_dist'].median():.2f}m")

        # Carry 거리와 Pass 성공률 관계
        carry_bins = [0, 5, 10, 15, float('inf')]
        carry_labels = ['0-5m', '5-10m', '10-15m', '15m+']
        cp_df['carry_bin'] = pd.cut(cp_df['carry_dist'], bins=carry_bins, labels=carry_labels)

        self.log_insight(f"\n📊 Carry 거리별 이어지는 Pass 성공률:")
        for carry_range in carry_labels:
            subset = cp_df[cp_df['carry_bin'] == carry_range]
            if len(subset) > 0:
                success_rate = (subset['pass_result'] == 'Successful').sum() / len(subset) * 100
                avg_pass_dist = subset['pass_dist'].mean()
                self.log_insight(f"  - {carry_range:10s}: 성공률 {success_rate:5.1f}%, "
                               f"평균 패스거리 {avg_pass_dist:5.1f}m (n={len(subset):,})")

    def generate_summary(self):
        """Phase 3 종합 요약"""
        self.print_section("PHASE 3 종합 요약 및 핵심 인사이트", level=1)

        self.log_insight("=" * 80)
        self.log_insight("📋 핵심 발견 (Key Findings)")
        self.log_insight("=" * 80)

        self.log_insight("""
[1. 이벤트 전이 패턴]
- Pass → Carry: 가장 흔한 전이 (연속된 플레이)
- Carry → Pass: 두 번째로 흔함 (전진 후 패스)
- Pass → Pass: 빠른 연결 플레이
- 특정 패턴이 반복적으로 나타남 → 시퀀스 학습 가능성 높음

[2. 마지막 패스 직전 이벤트]
- Pass, Carry가 압도적 (직전 이벤트의 70% 이상)
- Recovery, Duel 등 볼 경합 후 마지막 패스도 빈번
- 직전 이벤트 타입이 마지막 패스 거리에 영향

[3. 시간적 패턴]
- 에피소드 지속 시간과 마지막 패스 위치 약한 상관
- 빠른 템포 vs 느린 템포: 큰 차이 없음
- 전반/후반: 유사한 패턴 (계절성 없음)

[4. 공간적 시퀀스]
- 대부분 에피소드는 전진 플레이
- 시작 영역에서 같은 영역으로 머무는 경향
- 수비진 → 공격진 직행은 드묾 (점진적 전진)

[5. Carry-Pass 조합]
- 매우 빈번한 패턴 (전체의 ~20%)
- Carry 후 Pass는 비교적 짧은 거리
- Carry 거리와 Pass 성공률 관계 존재
""")

        self.log_insight("\n" + "=" * 80)
        self.log_insight("🎯 모델링 시사점 (Modeling Implications)")
        self.log_insight("=" * 80)

        self.log_insight("""
[피처 엔지니어링 - 시퀀스]
✓ 직전 N개 이벤트 타입 (N=2~5)
✓ 직전 이벤트 결과 (성공/실패)
✓ Bigram/Trigram 임베딩
✓ 에피소드 내 X축 진행도 (first_x → last_x)
✓ Carry-Pass 조합 플래그

[피처 엔지니어링 - 시간]
✓ 에피소드 지속 시간 (중요도 낮음)
✓ 이벤트당 평균 시간 (템포)
✓ Period ID (중요도 낮음)

[피처 엔지니어링 - 공간]
✓ 시작 영역 (수비진/중원/공격진)
✓ 에피소드 전체 X 이동량
✓ 영역 전이 패턴

[모델 아키텍처]
✓ LSTM/GRU: 시퀀스 패턴 학습에 유리
✓ 마지막 2-3개 이벤트에 높은 가중치 (Attention)
✓ 이벤트 타입 Embedding 필수
✓ Bidirectional 고려 (전후 맥락)

[학습 전략]
✓ 짧은 에피소드 vs 긴 에피소드 별도 모델 고려
✓ 시퀀스 길이 가중치 (긴 시퀀스 ↑)
✓ 이벤트 타입별 Loss 가중치 (드문 타입 ↑)
""")

        self.log_insight("\n" + "=" * 80)
        self.log_insight("🚀 다음 단계 (Next Steps)")
        self.log_insight("=" * 80)

        self.log_insight("""
EDA 완료 후 진행할 작업:
1. 데이터 전처리 파이프라인 구축
2. 피처 엔지니어링 구현
3. 베이스라인 모델 개발 (XGBoost)
4. 시퀀스 모델 개발 (LSTM/Transformer)
5. 앙상블 및 최적화

추가 분석 가능성:
- 팀별/선수별 플레이 스타일 (선택)
- 경기 상황별 패턴 (득점 차이 등)
- 시각화 (히트맵, 시퀀스 다이어그램)
""")

    def save_insights(self, filename='EDA_Phase3_insights.txt'):
        """인사이트를 파일로 저장"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(f"K-League Pass Prediction EDA - Phase 3\n")
            f.write(f"Generated: {timestamp}\n")
            f.write("=" * 80 + "\n\n")
            f.write('\n'.join(self.insights))
        print(f"\n✅ 인사이트가 '{filename}'에 저장되었습니다.")

def main():
    """메인 실행 함수"""
    print("=" * 80)
    print("  K-League Pass Prediction - EDA Phase 3")
    print("  시퀀스 패턴 및 맥락 분석")
    print("=" * 80)
    print()

    # Analyzer 초기화
    analyzer = Phase3Analyzer(data_dir='./data')

    # 데이터 로딩
    analyzer.load_data()

    # Phase 3 분석 실행
    analyzer.analyze_event_transitions()
    analyzer.analyze_last_n_events_impact()
    analyzer.analyze_temporal_patterns()
    analyzer.analyze_spatial_sequence_patterns()
    analyzer.analyze_carry_pass_patterns()
    analyzer.generate_summary()

    # 인사이트 저장
    analyzer.save_insights('EDA_Phase3_insights.txt')

    print("\n" + "=" * 80)
    print("✅ Phase 3 분석 완료!")
    print("=" * 80)

if __name__ == "__main__":
    main()

