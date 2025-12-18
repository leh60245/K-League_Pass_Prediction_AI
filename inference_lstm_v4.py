"""
K-League Pass Prediction - PyTorch LSTM/GRU 추론 파이프라인

학습된 LSTM/GRU 모델로 Test 데이터 예측 및 제출 파일 생성

작성일: 2025-12-18
"""

import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from train_lstm_v4 import SoccerDatasetV4, SoccerRNN
import warnings
from datetime import datetime

warnings.filterwarnings('ignore')


def load_model(checkpoint_path, device):
    """학습된 모델 로딩"""
    print(f"📦 모델 로딩 중: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    # 하이퍼파라미터
    hyperparams = checkpoint['hyperparameters']

    # 모델 생성
    model = SoccerRNN(
        num_numerical_features=checkpoint['num_numerical_features'],
        categorical_vocab_sizes=checkpoint['vocab_sizes'],
        embedding_dims=checkpoint['embedding_dims'],
        hidden_dim=hyperparams['hidden_dim'],
        num_layers=hyperparams['num_layers'],
        dropout=hyperparams['dropout'],
        use_lstm=hyperparams['use_lstm']
    ).to(device)

    # State dict 로딩
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"✅ 모델 로딩 완료 (Val Loss: {checkpoint['val_loss']:.4f}m)")

    return model, checkpoint


def predict(model, dataloader, device):
    """추론"""
    model.eval()
    predictions = []

    with torch.no_grad():
        for batch in dataloader:
            if len(batch) == 2:  # Test data (no target)
                num_feat, cat_feat = batch
            else:  # Val data (with target)
                num_feat, cat_feat, _ = batch

            num_feat = num_feat.to(device)
            cat_feat = cat_feat.to(device)

            output = model(num_feat, cat_feat)

            # 정규화 해제 (0~1 → 실제 좌표)
            output[:, 0] *= 105.0  # target_x
            output[:, 1] *= 68.0   # target_y

            predictions.append(output.cpu().numpy())

    return np.vstack(predictions)


def main():
    print("=" * 80)
    print("  PyTorch LSTM/GRU - Test 추론")
    print("  V4 Wide Format 데이터")
    print("=" * 80)
    print()

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔧 Using device: {device}\n")

    # 1. Test 데이터 로딩
    print("📊 Test 데이터 로딩...")
    test_data = pd.read_csv('processed_test_data_v4.csv')
    print(f"Test Shape: {test_data.shape}")
    print()

    # 2. 모델 로딩
    model, checkpoint = load_model('lstm_model_v4_best.pth', device)
    print()

    # 3. Test Dataset 생성
    print("📦 Test Dataset 생성 중...")

    # Test 데이터는 target이 없으므로 임시로 NaN 추가
    if 'target_x' not in test_data.columns:
        test_data['target_x'] = np.nan
        test_data['target_y'] = np.nan

    test_dataset = SoccerDatasetV4(test_data, K=checkpoint['hyperparameters']['K'], is_train=False)
    print()

    # DataLoader
    test_loader = DataLoader(
        test_dataset,
        batch_size=128,
        shuffle=False,
        num_workers=0
    )

    # 4. 추론
    print("🔮 추론 중...")
    predictions = predict(model, test_loader, device)
    print(f"✅ 추론 완료: {predictions.shape}")
    print()

    # 5. 제출 파일 생성
    print("💾 제출 파일 생성 중...")
    submission = pd.DataFrame({
        'game_episode': test_data['game_episode'].values,
        'end_x': predictions[:, 0],
        'end_y': predictions[:, 1]
    })

    # 파일 저장
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f'submission_lstm_v4_{timestamp}.csv'
    submission.to_csv(filename, index=False)

    print(f"✅ 제출 파일 저장: {filename}")
    print()

    # 6. 예측 통계
    print("=" * 80)
    print("  예측 통계")
    print("=" * 80)
    print(f"\nend_x 통계:")
    print(f"   - 최소: {predictions[:, 0].min():.2f}")
    print(f"   - 최대: {predictions[:, 0].max():.2f}")
    print(f"   - 평균: {predictions[:, 0].mean():.2f}")
    print(f"   - 표준편차: {predictions[:, 0].std():.2f}")

    print(f"\nend_y 통계:")
    print(f"   - 최소: {predictions[:, 1].min():.2f}")
    print(f"   - 최대: {predictions[:, 1].max():.2f}")
    print(f"   - 평균: {predictions[:, 1].mean():.2f}")
    print(f"   - 표준편차: {predictions[:, 1].std():.2f}")

    print("\n" + "=" * 80)
    print("다음 단계:")
    print("   1. Kaggle/Dacon에 제출")
    print("   2. LightGBM vs LSTM 성능 비교")
    print("   3. 앙상블 고려 (LightGBM + LSTM)")
    print("=" * 80)


if __name__ == "__main__":
    main()

