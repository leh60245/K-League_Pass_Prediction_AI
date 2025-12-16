"""
V3 전처리 테스트
"""
import sys
import os

print("=" * 80)
print("V3 전처리 테스트 시작")
print("=" * 80)

# 경로 확인
print(f"\n현재 디렉토리: {os.getcwd()}")
print(f"data 폴더 존재: {os.path.exists('./data')}")
print(f"test_index.csv 존재: {os.path.exists('./data/test_index.csv')}")

# 모듈 임포트 테스트
try:
    from preprocessing_v3 import DataPreprocessorV3
    print("\n✅ preprocessing_v3 모듈 임포트 성공")
except Exception as e:
    print(f"\n❌ preprocessing_v3 모듈 임포트 실패: {e}")
    sys.exit(1)

# Preprocessor 초기화 테스트
try:
    preprocessor = DataPreprocessorV3(data_dir='./data', K=20)
    print("✅ DataPreprocessorV3 초기화 성공")
except Exception as e:
    print(f"❌ DataPreprocessorV3 초기화 실패: {e}")
    sys.exit(1)

# 데이터 로딩 테스트
try:
    print("\n데이터 로딩 중...")
    data = preprocessor.load_data(verbose=True)
    print(f"✅ 데이터 로딩 성공: {data.shape}")
    print(f"   Train 에피소드: {data[data['is_train']==1]['game_episode'].nunique()}")
    print(f"   Test 에피소드: {data[data['is_train']==0]['game_episode'].nunique()}")
except Exception as e:
    print(f"❌ 데이터 로딩 실패: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 80)
print("✅ 테스트 성공!")
print("=" * 80)

