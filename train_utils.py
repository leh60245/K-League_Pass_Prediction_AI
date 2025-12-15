"""
모델 학습 유틸리티
피처 설정 자동 로딩 및 데이터 준비 헬퍼 함수들

실무 패턴: 공통 로직을 함수로 분리하여 재사용성 향상
"""

import pandas as pd
import numpy as np
import os
from feature_config import FeatureConfig


def load_data_and_features(data_path='processed_train_data.csv',
                           config_path='feature_config.json',
                           verbose=True):
    """
    데이터와 피처 설정을 함께 로딩

    Returns:
        data (DataFrame): 전처리된 데이터
        feature_cols (list): 피처 컬럼 리스트
        target_cols (list): 타겟 컬럼 리스트
        config (FeatureConfig): 피처 설정 객체
    """
    if verbose:
        print("=" * 80)
        print("  데이터 및 피처 설정 로딩")
        print("=" * 80)

    # 1. 데이터 로딩
    if verbose:
        print(f"\n📊 데이터 로딩: {data_path}")

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"데이터 파일을 찾을 수 없습니다: {data_path}")

    data = pd.read_csv(data_path)
    if verbose:
        print(f"✅ 데이터 로딩 완료: {data.shape}")

    # 2. 피처 설정 로딩
    if verbose:
        print(f"\n🔧 피처 설정 로딩: {config_path}")

    if not os.path.exists(config_path):
        if verbose:
            print(f"⚠️  설정 파일이 없습니다. 자동 생성 중...")

        from feature_config import create_feature_config_from_data
        config = create_feature_config_from_data(
            data_path=data_path,
            preprocessor_path='preprocessor.pkl'
        )
    else:
        config = FeatureConfig(config_path)
        if verbose:
            print(f"✅ 피처 설정 로딩 완료")

    # 3. 피처/타겟 추출
    feature_cols = config.get_feature_columns()
    target_cols = config.get_target_columns()

    # 실제 데이터에 존재하는 피처만 사용
    available_features = [col for col in feature_cols if col in data.columns]
    missing_features = [col for col in feature_cols if col not in data.columns]

    if verbose:
        print(f"\n📊 피처 정보:")
        print(f"  - 설정 파일 피처: {len(feature_cols)}개")
        print(f"  - 사용 가능 피처: {len(available_features)}개")
        if missing_features:
            print(f"  - 누락된 피처: {len(missing_features)}개")
            if len(missing_features) <= 5:
                for f in missing_features:
                    print(f"    · {f}")
        print(f"  - 타겟: {', '.join(target_cols)}")

    if verbose:
        print("\n" + "=" * 80)

    return data, available_features, target_cols, config


def prepare_train_val_split(data, feature_cols, target_cols,
                            val_ratio=0.2, random_seed=42, verbose=True):
    """
    Train/Validation 분할 (게임 기반)

    Returns:
        X_train, y_train, X_val, y_val
    """
    if verbose:
        print("📊 Train/Val Split (게임 기반)...")

    # 게임 ID 기반 분할
    games = data['game_id'].unique()
    np.random.seed(random_seed)
    np.random.shuffle(games)

    n_val_games = int(len(games) * val_ratio)
    val_games = games[:n_val_games]

    val_mask = data['game_id'].isin(val_games)
    train_mask = ~val_mask

    # 피처/타겟 추출
    X_train = data.loc[train_mask, feature_cols].fillna(0).values
    y_train = data.loc[train_mask, target_cols].values
    X_val = data.loc[val_mask, feature_cols].fillna(0).values
    y_val = data.loc[val_mask, target_cols].values

    if verbose:
        print(f"  - Train: {len(games) - n_val_games} 게임, {X_train.shape[0]:,} 에피소드")
        print(f"  - Val: {n_val_games} 게임, {X_val.shape[0]:,} 에피소드")
        print(f"  - 피처: {X_train.shape[1]}개\n")

    return X_train, y_train, X_val, y_val


def euclidean_distance(y_true, y_pred):
    """
    유클리드 거리 계산 (평균)

    Args:
        y_true: (N, 2) - 실제 좌표 [x, y]
        y_pred: (N, 2) - 예측 좌표 [x, y]

    Returns:
        float: 평균 유클리드 거리 (m)
    """
    if y_true.ndim == 1:
        y_true = y_true.reshape(-1, 2)
    if y_pred.ndim == 1:
        y_pred = y_pred.reshape(-1, 2)

    distances = np.sqrt(
        (y_true[:, 0] - y_pred[:, 0])**2 +
        (y_true[:, 1] - y_pred[:, 1])**2
    )
    return np.mean(distances)


def print_performance_summary(train_eucl, val_eucl, baseline_eucl=20.37,
                             target_eucl=18.0, verbose=True):
    """
    성능 요약 출력

    Args:
        train_eucl: Train 유클리드 거리
        val_eucl: Validation 유클리드 거리
        baseline_eucl: 베이스라인 성능 (기본값: 20.37m)
        target_eucl: 목표 성능 (기본값: 18m)
    """
    if not verbose:
        return

    improvement = baseline_eucl - val_eucl
    improvement_pct = (improvement / baseline_eucl) * 100

    print("\n" + "=" * 80)
    print("  성능 요약")
    print("=" * 80)

    print(f"\n📊 유클리드 거리:")
    print(f"  - Train: {train_eucl:.2f}m")
    print(f"  - Val: {val_eucl:.2f}m")

    print(f"\n📊 베이스라인 대비:")
    print(f"  - 베이스라인: {baseline_eucl:.2f}m")
    print(f"  - 개선: {improvement:.2f}m ({improvement_pct:+.1f}%)")

    if val_eucl < baseline_eucl:
        print(f"  ✅ 베이스라인보다 {improvement:.2f}m 개선!")
    else:
        print(f"  ⚠️  베이스라인보다 {-improvement:.2f}m 나쁨")

    print(f"\n📊 목표 달성:")
    print(f"  - 목표: < {target_eucl:.2f}m")
    print(f"  - 현재: {val_eucl:.2f}m")

    if val_eucl < target_eucl:
        print(f"  🎯 목표 달성! ({val_eucl:.2f}m < {target_eucl:.2f}m)")
    else:
        gap = val_eucl - target_eucl
        print(f"  ⏰ 목표 미달성 (목표까지 {gap:.2f}m 남음)")

    print("=" * 80)


def save_submission(predictions, output_path='submission.csv', verbose=True):
    """
    제출 파일 생성

    Args:
        predictions: (N, 2) - 예측 좌표 [x, y]
        output_path: 저장 경로
    """
    submission = pd.DataFrame({
        'end_x': predictions[:, 0],
        'end_y': predictions[:, 1]
    })

    submission.to_csv(output_path, index=False)

    if verbose:
        print(f"✅ 제출 파일 저장: {output_path}")
        print(f"  - 행 개수: {len(submission):,}")
        print(f"  - 컬럼: {list(submission.columns)}")


def get_feature_group_importance(model_x, model_y, feature_cols, config, top_n=5):
    """
    피처 그룹별 중요도 계산

    Args:
        model_x: X 좌표 예측 모델
        model_y: Y 좌표 예측 모델
        feature_cols: 피처 컬럼 리스트
        config: FeatureConfig 객체
        top_n: 각 그룹에서 표시할 상위 N개

    Returns:
        dict: 그룹별 중요도
    """
    # 피처별 중요도
    importance_x = model_x.feature_importances_
    importance_y = model_y.feature_importances_
    importance_avg = (importance_x + importance_y) / 2

    # 피처 -> 중요도 매핑
    feature_importance = dict(zip(feature_cols, importance_avg))

    # 그룹별 집계
    group_importance = {}
    feature_groups = config.config.get('feature_groups', {})

    for group_name, group_features in feature_groups.items():
        # 이 그룹에 속한 피처들의 중요도 합
        group_total = sum(feature_importance.get(f, 0) for f in group_features
                         if f in feature_cols)

        # 개별 피처 중요도
        feature_details = []
        for f in group_features:
            if f in feature_cols:
                importance = feature_importance[f]
                feature_details.append((f, importance))

        # 중요도 순 정렬
        feature_details.sort(key=lambda x: x[1], reverse=True)

        group_importance[group_name] = {
            'total': group_total,
            'features': feature_details[:top_n]
        }

    return group_importance


def print_feature_group_importance(group_importance, verbose=True):
    """피처 그룹별 중요도 출력"""
    if not verbose:
        return

    print("\n" + "=" * 80)
    print("  피처 그룹별 중요도")
    print("=" * 80)

    # 총 중요도 순으로 정렬
    sorted_groups = sorted(group_importance.items(),
                          key=lambda x: x[1]['total'],
                          reverse=True)

    for group_name, info in sorted_groups:
        print(f"\n📊 {group_name} (총 {info['total']:.4f})")
        for feature, importance in info['features']:
            print(f"  · {feature:30s}: {importance:.4f}")

    print("\n" + "=" * 80)


# 실무 팁: 이렇게 유틸리티 함수를 만들어두면
# 1. 코드 재사용성 향상
# 2. 유지보수 용이
# 3. 일관성 유지
# 4. 테스트 용이

if __name__ == "__main__":
    # 테스트
    print("🧪 유틸리티 함수 테스트\n")

    # 데이터 로딩 테스트
    data, features, targets, config = load_data_and_features()

    print(f"\n✅ 테스트 완료!")
    print(f"  - 데이터: {data.shape}")
    print(f"  - 피처: {len(features)}개")
    print(f"  - 타겟: {len(targets)}개")

