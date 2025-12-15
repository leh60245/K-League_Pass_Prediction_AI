"""
Feature Configuration Manager
피처 설정을 JSON으로 관리하여 전처리와 모델 학습 간 일관성 유지
"""

import json
import os
from typing import List, Dict, Any
from datetime import datetime


class FeatureConfig:
    """피처 설정 관리 클래스"""

    def __init__(self, config_path: str = 'feature_config.json'):
        self.config_path = config_path
        self.config = self._load_or_create_default()

    def _load_or_create_default(self) -> Dict[str, Any]:
        """설정 파일 로드 또는 기본값 생성"""
        if os.path.exists(self.config_path):
            with open(self.config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        else:
            return self._create_default_config()

    def _create_default_config(self) -> Dict[str, Any]:
        """기본 설정 생성"""
        return {
            "version": "1.0",
            "created_at": datetime.now().isoformat(),
            "feature_columns": [],
            "target_columns": ["end_x", "end_y"],
            "categorical_features": [],
            "numerical_features": [],
            "feature_groups": {},
            "preprocessing_params": {}
        }

    def save(self):
        """설정 파일 저장"""
        self.config['updated_at'] = datetime.now().isoformat()
        with open(self.config_path, 'w', encoding='utf-8') as f:
            json.dump(self.config, f, indent=2, ensure_ascii=False)
        print(f"✅ 피처 설정 저장: {self.config_path}")

    def update_from_preprocessor(self, preprocessor):
        """전처리기로부터 피처 정보 업데이트"""
        feature_cols = preprocessor.get_feature_columns()

        # 피처 타입 자동 감지
        categorical = []
        numerical = []

        for col in feature_cols:
            if 'encoded' in col or 'zone' in col or col in ['is_home', 'period_id']:
                categorical.append(col)
            else:
                numerical.append(col)

        # 피처 그룹 분류
        feature_groups = {
            "basic_spatial": [
                "start_x", "start_y", "delta_x", "delta_y", "distance",
                "start_x_norm", "start_y_norm"
            ],
            "goal_related": [
                "distance_to_goal_start", "distance_to_goal_end",
                "goal_approach", "shooting_angle"
            ],
            "zone_features": [
                "start_x_zone", "start_y_zone", "start_x_zone_fine",
                "in_penalty_area", "in_final_third"
            ],
            "velocity_acceleration": [
                "velocity", "velocity_x", "velocity_y", "acceleration"
            ],
            "pressure": [
                "event_density", "local_pressure", "weighted_pressure"
            ],
            "space_creation": [
                "distance_change_rate", "vertical_spread", "attack_width"
            ],
            "direction": [
                "direction_consistency", "pass_angle_change",
                "horizontal_vertical_ratio"
            ],
            "tempo": [
                "tempo", "tempo_change", "match_phase"
            ],
            "path_efficiency": [
                "path_efficiency", "forward_momentum"
            ],
            "positioning": [
                "dist_from_team_center", "final_third_time_ratio"
            ],
            "history": [
                "avg_velocity_3", "goal_approach_trend"
            ],
            "episode_info": [
                "episode_length", "event_order", "x_progression",
                "x_total_progression", "relative_time"
            ],
            "event_type": [
                "type_name_encoded"
            ],
            "previous_events": [
                "prev_type_name_encoded", "prev_start_x", "prev_start_y",
                "prev_end_x", "prev_end_y", "prev2_type_name_encoded"
            ],
            "match_info": [
                "period_id", "is_home"
            ],
            "result": [
                "result_name_encoded", "prev_result_name_encoded"
            ]
        }

        # 실제 존재하는 피처만 필터링
        filtered_groups = {}
        for group_name, features in feature_groups.items():
            existing = [f for f in features if f in feature_cols]
            if existing:
                filtered_groups[group_name] = existing

        self.config.update({
            "version": "2.0",
            "feature_columns": feature_cols,
            "categorical_features": categorical,
            "numerical_features": numerical,
            "feature_groups": filtered_groups,
            "n_features": len(feature_cols),
            "n_categorical": len(categorical),
            "n_numerical": len(numerical)
        })

        return self

    def get_feature_columns(self) -> List[str]:
        """피처 컬럼 리스트 반환"""
        return self.config.get('feature_columns', [])

    def get_target_columns(self) -> List[str]:
        """타겟 컬럼 리스트 반환"""
        return self.config.get('target_columns', ['end_x', 'end_y'])

    def get_categorical_features(self) -> List[str]:
        """범주형 피처 리스트 반환"""
        return self.config.get('categorical_features', [])

    def get_numerical_features(self) -> List[str]:
        """수치형 피처 리스트 반환"""
        return self.config.get('numerical_features', [])

    def get_feature_group(self, group_name: str) -> List[str]:
        """특정 그룹의 피처 리스트 반환"""
        return self.config.get('feature_groups', {}).get(group_name, [])

    def print_summary(self):
        """피처 설정 요약 출력"""
        print("=" * 80)
        print("  피처 설정 요약")
        print("=" * 80)
        print(f"버전: {self.config.get('version', 'N/A')}")
        print(f"총 피처 개수: {self.config.get('n_features', 0)}")
        print(f"  - 범주형: {self.config.get('n_categorical', 0)}")
        print(f"  - 수치형: {self.config.get('n_numerical', 0)}")
        print(f"타겟 컬럼: {', '.join(self.get_target_columns())}")
        print(f"\n피처 그룹: {len(self.config.get('feature_groups', {}))}")
        for group_name, features in self.config.get('feature_groups', {}).items():
            print(f"  - {group_name}: {len(features)}개")
        print("=" * 80)


def create_feature_config_from_data(data_path: str, preprocessor_path: str = None):
    """
    데이터로부터 피처 설정 자동 생성

    실무 패턴: 데이터 타입을 자동으로 감지하여 설정 생성
    """
    import pandas as pd
    import pickle

    # 데이터 로딩
    print(f"📊 데이터 로딩: {data_path}")
    data = pd.read_csv(data_path)

    # Preprocessor가 있으면 사용
    if preprocessor_path and os.path.exists(preprocessor_path):
        print(f"🔧 Preprocessor 로딩: {preprocessor_path}")
        try:
            # DataPreprocessor 클래스 import
            from preprocessing import DataPreprocessor
            preprocessor = DataPreprocessor(data_dir='./data')
            preprocessor.load_preprocessor(preprocessor_path)

            config = FeatureConfig()
            config.update_from_preprocessor(preprocessor)
        except Exception as e:
            print(f"⚠️  Preprocessor 로딩 실패: {e}")
            print("📊 데이터로부터 자동 감지로 전환")
            preprocessor_path = None

    if not preprocessor_path:
        # 데이터로부터 자동 감지
        print("🔍 데이터로부터 피처 자동 감지")
        config = FeatureConfig()

        # 타겟 제외
        target_cols = ['end_x', 'end_y']
        exclude_cols = target_cols + ['game_episode', 'game_id', 'episode_id',
                                       'action_id', 'player_id', 'team_id']

        # 피처 컬럼 추출
        feature_cols = [col for col in data.columns
                       if col not in exclude_cols and not data[col].isna().all()]

        # 타입별 분류
        categorical = []
        numerical = []

        for col in feature_cols:
            if data[col].dtype in ['object', 'category']:
                categorical.append(col)
            elif 'encoded' in col or data[col].nunique() < 50:
                categorical.append(col)
            else:
                numerical.append(col)

        config.config.update({
            "feature_columns": feature_cols,
            "categorical_features": categorical,
            "numerical_features": numerical,
            "n_features": len(feature_cols),
            "n_categorical": len(categorical),
            "n_numerical": len(numerical)
        })

    config.save()
    config.print_summary()

    return config


if __name__ == "__main__":
    # 예제: processed_train_data로부터 설정 생성
    config = create_feature_config_from_data(
        data_path='processed_train_data.csv',
        preprocessor_path='preprocessor.pkl'
    )

    print(f"\n✅ 피처 설정 파일 생성 완료: feature_config.json")
    print(f"✅ 모델 학습 시 이 파일을 읽어서 사용하세요!")

