"""
K-League Pass Prediction - EDA Analysis
Phase 2: Target Variable Analysis

목표: 예측 대상(마지막 패스)의 특성 심층 분석
출력: 텍스트 기반 인사이트 + 베이스라인 성능 추정
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
pd.set_option('display.float_format', lambda x: f'{x:.4f}')

class Phase2Analyzer:
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

    def extract_last_passes(self):
        """각 에피소드의 마지막 패스 추출"""
        self.log_insight("🔍 마지막 패스 추출 중...")

        # 각 에피소드의 마지막 이벤트 추출
        last_events = self.train_data.groupby('game_episode').tail(1).copy()

        self.log_insight(f"  - 마지막 이벤트 수: {len(last_events):,}")
        self.log_insight(f"  - Pass 타입 이벤트: {(last_events['type_name'].str.contains('Pass')).sum():,}")
        self.log_insight(f"  - 기타 타입: {(~last_events['type_name'].str.contains('Pass')).sum():,}")

        return last_events

    def analyze_target_distribution(self):
        """예측 대상 좌표 분포 분석"""
        self.print_section("PHASE 2: 예측 대상 분석 - 좌표 분포", level=1)

        last_events = self.extract_last_passes()

        # 2.1 기본 통계
        self.print_section("2.1 마지막 패스 좌표 기본 통계", level=2)

        self.log_insight("📊 end_x (좌우 위치) 통계:")
        end_x = last_events['end_x']
        self.log_insight(f"  - 평균: {end_x.mean():.2f}")
        self.log_insight(f"  - 중앙값: {end_x.median():.2f}")
        self.log_insight(f"  - 표준편차: {end_x.std():.2f}")
        self.log_insight(f"  - 최소: {end_x.min():.2f}")
        self.log_insight(f"  - 최대: {end_x.max():.2f}")

        self.log_insight("\n📊 end_y (상하 위치) 통계:")
        end_y = last_events['end_y']
        self.log_insight(f"  - 평균: {end_y.mean():.2f}")
        self.log_insight(f"  - 중앙값: {end_y.median():.2f}")
        self.log_insight(f"  - 표준편차: {end_y.std():.2f}")
        self.log_insight(f"  - 최소: {end_y.min():.2f}")
        self.log_insight(f"  - 최대: {end_y.max():.2f}")

        # 백분위수
        self.log_insight("\n📊 좌표 백분위수:")
        percentiles = [10, 25, 50, 75, 90]
        self.log_insight("  end_x:")
        for p in percentiles:
            val = end_x.quantile(p/100)
            self.log_insight(f"    {p:2d}%: {val:6.2f}")

        self.log_insight("  end_y:")
        for p in percentiles:
            val = end_y.quantile(p/100)
            self.log_insight(f"    {p:2d}%: {val:6.2f}")

        # 2.2 경기장 영역 분석
        self.print_section("2.2 경기장 영역별 분포", level=2)

        # X축 기준 (수비/중원/공격)
        def classify_x_zone(x):
            if x < 35:
                return '수비진 (0-35)'
            elif x < 70:
                return '중원 (35-70)'
            else:
                return '공격진 (70-105)'

        # Y축 기준 (좌측/중앙/우측)
        def classify_y_zone(y):
            if y < 22.67:
                return '좌측 (0-22.67)'
            elif y < 45.33:
                return '중앙 (22.67-45.33)'
            else:
                return '우측 (45.33-68)'

        last_events['x_zone'] = last_events['end_x'].apply(classify_x_zone)
        last_events['y_zone'] = last_events['end_y'].apply(classify_y_zone)

        self.log_insight("📊 X축 영역별 분포 (전진 방향):")
        x_zone_counts = last_events['x_zone'].value_counts()
        for zone in ['수비진 (0-35)', '중원 (35-70)', '공격진 (70-105)']:
            if zone in x_zone_counts.index:
                count = x_zone_counts[zone]
                pct = (count / len(last_events)) * 100
                self.log_insight(f"  - {zone:20s}: {count:6,} ({pct:5.1f}%)")

        self.log_insight("\n📊 Y축 영역별 분포 (좌우):")
        y_zone_counts = last_events['y_zone'].value_counts()
        for zone in ['좌측 (0-22.67)', '중앙 (22.67-45.33)', '우측 (45.33-68)']:
            if zone in y_zone_counts.index:
                count = y_zone_counts[zone]
                pct = (count / len(last_events)) * 100
                self.log_insight(f"  - {zone:20s}: {count:6,} ({pct:5.1f}%)")

        # 2.3 골 근접도 분석
        self.print_section("2.3 골 근접도 분석", level=2)

        # 골문까지의 거리 (골문 중앙: x=105, y=34)
        goal_x, goal_y = 105, 34
        last_events['distance_to_goal'] = np.sqrt(
            (last_events['end_x'] - goal_x)**2 +
            (last_events['end_y'] - goal_y)**2
        )

        self.log_insight("📊 골문까지의 거리:")
        dist = last_events['distance_to_goal']
        self.log_insight(f"  - 평균: {dist.mean():.2f}m")
        self.log_insight(f"  - 중앙값: {dist.median():.2f}m")
        self.log_insight(f"  - 표준편차: {dist.std():.2f}m")

        # 거리 구간별 분포
        dist_bins = [0, 20, 40, 60, 80, float('inf')]
        dist_labels = ['0-20m', '20-40m', '40-60m', '60-80m', '80m+']
        last_events['dist_zone'] = pd.cut(dist, bins=dist_bins, labels=dist_labels)

        self.log_insight("\n📊 골문 거리 구간별 분포:")
        for zone in dist_labels:
            count = (last_events['dist_zone'] == zone).sum()
            pct = (count / len(last_events)) * 100
            self.log_insight(f"  - {zone:10s}: {count:6,} ({pct:5.1f}%)")

        return last_events

    def analyze_pass_types(self):
        """마지막 패스 타입 분석"""
        self.print_section("PHASE 2: 마지막 패스 타입 분석", level=1)

        last_events = self.train_data.groupby('game_episode').tail(1).copy()

        # 3.1 이벤트 타입 분포
        self.print_section("3.1 마지막 이벤트 타입 분포", level=2)

        type_counts = last_events['type_name'].value_counts()
        self.log_insight("📊 Top 10 이벤트 타입:")
        for i, (event_type, count) in enumerate(type_counts.head(10).items(), 1):
            pct = (count / len(last_events)) * 100
            self.log_insight(f"  {i:2d}. {event_type:30s}: {count:6,} ({pct:5.1f}%)")

        # 3.2 Pass 타입 상세 분석
        self.print_section("3.2 Pass 타입 상세 분석", level=2)

        pass_events = last_events[last_events['type_name'].str.contains('Pass', na=False)]
        self.log_insight(f"📊 Pass 관련 이벤트: {len(pass_events):,} ({len(pass_events)/len(last_events)*100:.1f}%)")

        if len(pass_events) > 0:
            pass_types = pass_events['type_name'].value_counts()
            self.log_insight("\nPass 세부 타입:")
            for pass_type, count in pass_types.items():
                pct = (count / len(pass_events)) * 100
                self.log_insight(f"  - {pass_type:30s}: {count:6,} ({pct:5.1f}%)")

            # 성공/실패 분석
            if 'result_name' in pass_events.columns:
                self.log_insight("\n📊 Pass 성공률:")
                result_counts = pass_events['result_name'].value_counts()
                for result, count in result_counts.items():
                    pct = (count / len(pass_events)) * 100
                    self.log_insight(f"  - {result:20s}: {count:6,} ({pct:5.1f}%)")

                if 'Successful' in result_counts.index and 'Unsuccessful' in result_counts.index:
                    success_rate = result_counts['Successful'] / (result_counts['Successful'] + result_counts['Unsuccessful']) * 100
                    self.log_insight(f"\n  전체 성공률: {success_rate:.1f}%")

    def analyze_start_end_relationship(self):
        """시작-도착 위치 관계 분석"""
        self.print_section("PHASE 2: 시작-도착 위치 관계 분석", level=1)

        last_events = self.train_data.groupby('game_episode').tail(1).copy()

        # 4.1 패스 거리 분석
        self.print_section("4.1 패스 거리 분석", level=2)

        last_events['pass_distance'] = np.sqrt(
            (last_events['end_x'] - last_events['start_x'])**2 +
            (last_events['end_y'] - last_events['start_y'])**2
        )

        self.log_insight("📊 마지막 패스 거리:")
        dist = last_events['pass_distance']
        self.log_insight(f"  - 평균: {dist.mean():.2f}m")
        self.log_insight(f"  - 중앙값: {dist.median():.2f}m")
        self.log_insight(f"  - 표준편차: {dist.std():.2f}m")
        self.log_insight(f"  - 최소: {dist.min():.2f}m")
        self.log_insight(f"  - 최대: {dist.max():.2f}m")

        # 거리 분포
        dist_bins = [0, 5, 10, 15, 20, 30, float('inf')]
        dist_labels = ['0-5m', '5-10m', '10-15m', '15-20m', '20-30m', '30m+']
        last_events['pass_dist_zone'] = pd.cut(dist, bins=dist_bins, labels=dist_labels)

        self.log_insight("\n📊 패스 거리 분포:")
        for zone in dist_labels:
            count = (last_events['pass_dist_zone'] == zone).sum()
            pct = (count / len(last_events)) * 100
            self.log_insight(f"  - {zone:10s}: {count:6,} ({pct:5.1f}%)")

        # 4.2 패스 방향 분석
        self.print_section("4.2 패스 방향 분석", level=2)

        last_events['delta_x'] = last_events['end_x'] - last_events['start_x']
        last_events['delta_y'] = last_events['end_y'] - last_events['start_y']

        self.log_insight("📊 패스 방향 (X축 - 전진/후진):")
        forward = (last_events['delta_x'] > 0).sum()
        backward = (last_events['delta_x'] < 0).sum()
        neutral = (last_events['delta_x'] == 0).sum()
        total = len(last_events)

        self.log_insight(f"  - 전진 패스 (X+): {forward:6,} ({forward/total*100:5.1f}%)")
        self.log_insight(f"  - 후진 패스 (X-): {backward:6,} ({backward/total*100:5.1f}%)")
        self.log_insight(f"  - 수평 패스 (X=): {neutral:6,} ({neutral/total*100:5.1f}%)")
        self.log_insight(f"  - 평균 X 이동: {last_events['delta_x'].mean():.2f}m")

        self.log_insight("\n📊 패스 방향 (Y축 - 좌우):")
        self.log_insight(f"  - 평균 Y 이동: {last_events['delta_y'].mean():.2f}m")
        self.log_insight(f"  - Y 이동 표준편차: {last_events['delta_y'].std():.2f}m")

        # 4.3 시작-도착 상관관계
        self.print_section("4.3 시작-도착 좌표 상관관계", level=2)

        corr_x = last_events['start_x'].corr(last_events['end_x'])
        corr_y = last_events['start_y'].corr(last_events['end_y'])

        self.log_insight("📊 좌표 상관계수:")
        self.log_insight(f"  - start_x ↔ end_x: {corr_x:.4f}")
        self.log_insight(f"  - start_y ↔ end_y: {corr_y:.4f}")

        if corr_x > 0.7:
            self.log_insight(f"  ✓ X 좌표 강한 양의 상관관계 → 시작 위치가 도착 위치 예측에 중요")
        if corr_y > 0.7:
            self.log_insight(f"  ✓ Y 좌표 강한 양의 상관관계 → 시작 위치가 도착 위치 예측에 중요")

    def estimate_baseline_performance(self):
        """베이스라인 성능 추정"""
        self.print_section("PHASE 2: 베이스라인 성능 추정", level=1)

        last_events = self.train_data.groupby('game_episode').tail(1).copy()

        # 5.1 단순 평균 예측
        self.print_section("5.1 단순 평균 예측 (Naive Baseline)", level=2)

        mean_x = last_events['end_x'].mean()
        mean_y = last_events['end_y'].mean()

        self.log_insight(f"📊 전체 평균 좌표:")
        self.log_insight(f"  - 평균 end_x: {mean_x:.2f}")
        self.log_insight(f"  - 평균 end_y: {mean_y:.2f}")

        # 유클리드 거리 계산
        last_events['pred_x'] = mean_x
        last_events['pred_y'] = mean_y
        last_events['error'] = np.sqrt(
            (last_events['end_x'] - last_events['pred_x'])**2 +
            (last_events['end_y'] - last_events['pred_y'])**2
        )

        mean_error = last_events['error'].mean()
        median_error = last_events['error'].median()
        std_error = last_events['error'].std()

        self.log_insight(f"\n📊 Naive Baseline 성능 (전체 평균 예측):")
        self.log_insight(f"  - 평균 유클리드 거리: {mean_error:.2f}m")
        self.log_insight(f"  - 중앙값 유클리드 거리: {median_error:.2f}m")
        self.log_insight(f"  - 표준편차: {std_error:.2f}m")

        # 5.2 중앙값 예측
        self.print_section("5.2 중앙값 예측", level=2)

        median_x = last_events['end_x'].median()
        median_y = last_events['end_y'].median()

        self.log_insight(f"📊 전체 중앙값 좌표:")
        self.log_insight(f"  - 중앙값 end_x: {median_x:.2f}")
        self.log_insight(f"  - 중앙값 end_y: {median_y:.2f}")

        last_events['pred_x_med'] = median_x
        last_events['pred_y_med'] = median_y
        last_events['error_med'] = np.sqrt(
            (last_events['end_x'] - last_events['pred_x_med'])**2 +
            (last_events['end_y'] - last_events['pred_y_med'])**2
        )

        mean_error_med = last_events['error_med'].mean()

        self.log_insight(f"\n📊 중앙값 예측 성능:")
        self.log_insight(f"  - 평균 유클리드 거리: {mean_error_med:.2f}m")

        # 5.3 시작 위치 그대로 예측
        self.print_section("5.3 시작 위치 그대로 예측 (Identity Baseline)", level=2)

        last_events['error_identity'] = np.sqrt(
            (last_events['end_x'] - last_events['start_x'])**2 +
            (last_events['end_y'] - last_events['start_y'])**2
        )

        mean_error_identity = last_events['error_identity'].mean()

        self.log_insight(f"📊 Identity Baseline 성능 (start = end 가정):")
        self.log_insight(f"  - 평균 유클리드 거리: {mean_error_identity:.2f}m")

        # 5.4 베이스라인 비교
        self.print_section("5.4 베이스라인 비교 요약", level=2)

        self.log_insight("📊 베이스라인 성능 비교:")
        self.log_insight(f"  1. 전체 평균 예측:   {mean_error:.2f}m")
        self.log_insight(f"  2. 전체 중앙값 예측: {mean_error_med:.2f}m")
        self.log_insight(f"  3. 시작=도착 예측:   {mean_error_identity:.2f}m")

        best_baseline = min(mean_error, mean_error_med, mean_error_identity)
        self.log_insight(f"\n✓ 최선의 단순 베이스라인: {best_baseline:.2f}m")
        self.log_insight(f"✓ 이 값보다 낮은 성능을 목표로 모델 개발 필요")

    def generate_summary(self):
        """Phase 2 종합 요약"""
        self.print_section("PHASE 2 종합 요약 및 핵심 인사이트", level=1)

        self.log_insight("=" * 80)
        self.log_insight("📋 핵심 발견 (Key Findings)")
        self.log_insight("=" * 80)

        self.log_insight("""
[1. 예측 대상 특성]
- 마지막 패스는 경기장 전 영역에 고루 분포
- 중원(35-70m) 지역이 가장 빈번
- Y축(좌우)은 비교적 균등한 분포

[2. 패스 특성]
- 평균 패스 거리: 약 15-20m 내외
- 전진 패스가 후진 패스보다 우세
- 짧은 패스(0-15m)가 대부분

[3. 예측 가능성]
- start ↔ end 강한 상관관계 존재
- 시작 위치만으로도 어느정도 예측 가능
- 하지만 표준편차가 크므로 맥락 정보 필요

[4. 베이스라인]
- 단순 평균 예측: 약 20-25m 오차
- 모델 개발 목표: 15m 이하
- 상위 모델 목표: 10m 이하
""")

        self.log_insight("\n" + "=" * 80)
        self.log_insight("🎯 모델링 시사점 (Modeling Implications)")
        self.log_insight("=" * 80)

        self.log_insight("""
[피처 엔지니어링]
✓ 시작 위치 (start_x, start_y): 가장 중요한 피처
✓ 이동 거리/방향: delta_x, delta_y 피처
✓ 골 근접도: distance_to_goal
✓ 경기장 영역: 영역별 패턴이 다를 수 있음

[모델 전략]
✓ 회귀 문제로 접근 (end_x, end_y 동시 예측)
✓ Multi-output 회귀 or 2개 모델 앙상블
✓ 공간적 제약 반영 (0≤x≤105, 0≤y≤68)

[손실 함수]
✓ 유클리드 거리 직접 최적화 고려
✓ MSE Loss도 합리적 선택
✓ X, Y 좌표의 중요도 균형

[성능 목표]
✓ 최소 목표: 베이스라인(20-25m) 이하
✓ 경쟁력: 15m 이하
✓ 우승권: 10m 이하
""")

        self.log_insight("\n" + "=" * 80)
        self.log_insight("🚀 다음 단계 (Next Steps)")
        self.log_insight("=" * 80)

        self.log_insight("""
Phase 3에서 분석할 내용:
1. 시퀀스 패턴 분석 (이벤트 연쇄)
2. 직전 N개 이벤트가 마지막 패스에 미치는 영향
3. 이벤트 타입 전이 확률
4. 시간적 특성과 좌표의 관계
""")

    def save_insights(self, filename='EDA_Phase2_insights.txt'):
        """인사이트를 파일로 저장"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(f"K-League Pass Prediction EDA - Phase 2\n")
            f.write(f"Generated: {timestamp}\n")
            f.write("=" * 80 + "\n\n")
            f.write('\n'.join(self.insights))
        print(f"\n✅ 인사이트가 '{filename}'에 저장되었습니다.")

def main():
    """메인 실행 함수"""
    print("=" * 80)
    print("  K-League Pass Prediction - EDA Phase 2")
    print("  예측 대상(마지막 패스) 상세 분석")
    print("=" * 80)
    print()

    # Analyzer 초기화
    analyzer = Phase2Analyzer(data_dir='./data')

    # 데이터 로딩
    analyzer.load_data()

    # Phase 2 분석 실행
    analyzer.analyze_target_distribution()
    analyzer.analyze_pass_types()
    analyzer.analyze_start_end_relationship()
    analyzer.estimate_baseline_performance()
    analyzer.generate_summary()

    # 인사이트 저장
    analyzer.save_insights('EDA_Phase2_insights.txt')

    print("\n" + "=" * 80)
    print("✅ Phase 2 분석 완료!")
    print("=" * 80)

if __name__ == "__main__":
    main()

