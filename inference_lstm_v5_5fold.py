"""
K-League Pass Prediction - LSTM V5 5-Fold Ensemble Inference + TTA

- 5개 Fold 모델의 평균 예측
- Test Time Augmentation (원본 + 좌우 반전 등)
- 최종 제출 파일 생성

작성일: 2025-12-19
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import warnings
import re
from tqdm import tqdm

warnings.filterwarnings('ignore')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🔧 Using device: {device}")


class SoccerDatasetV5(Dataset):
    """Test용 Dataset"""

    def __init__(self, data, K=20, numerical_features=None, categorical_features=None):
        self.data = data.reset_index(drop=True)
        self.K = K

        exclude_cols = ['game_episode', 'game_id']
        feature_data = data.drop(columns=[c for c in exclude_cols if c in data.columns])

        if numerical_features is None or categorical_features is None:
            self.numerical_features, self.categorical_features = self._classify_columns(feature_data.columns)
        else:
            self.numerical_features = numerical_features
            self.categorical_features = categorical_features

        self.numerical_tensor = self._prepare_numerical_features(feature_data)
        self.categorical_tensor = self._prepare_categorical_features(feature_data)
        self.padding_mask = self._create_padding_mask()

    def _classify_columns(self, columns):
        pattern = re.compile(r'^(.+)_(\d+)$')
        feature_names = set()

        for col in columns:
            match = pattern.match(col)
            if match:
                feature_names.add(match.group(1))

        categorical_keywords = ['type_id', 'res_id', 'team_id_enc', 'is_home', 'is_last', 'period_id']
        categorical_features = []
        numerical_features = []

        for feat in sorted(feature_names):
            if any(keyword in feat for keyword in categorical_keywords):
                categorical_features.append(feat)
            else:
                numerical_features.append(feat)

        return numerical_features, categorical_features

    def _prepare_numerical_features(self, data):
        tensors = []
        x_coord_keywords = ['start_x', 'end_x', 'dx']
        y_coord_keywords = ['start_y', 'end_y', 'dy']

        for feat_name in self.numerical_features:
            cols = [f"{feat_name}_{i}" for i in range(self.K)]
            cols = [c for c in cols if c in data.columns]
            if not cols:
                continue

            feat_data = data[cols].values.astype(np.float32)
            if feat_data.shape[1] < self.K:
                padding = np.zeros((feat_data.shape[0], self.K - feat_data.shape[1]), dtype=np.float32)
                feat_data = np.concatenate([feat_data, padding], axis=1)

            if any(kw in feat_name for kw in x_coord_keywords):
                feat_data = feat_data / 105.0
            elif any(kw in feat_name for kw in y_coord_keywords):
                feat_data = feat_data / 68.0
            elif 'speed' in feat_name.lower():
                feat_data = np.clip(feat_data / 30.0, 0, 1)
            elif 'angle' in feat_name.lower() or 'direction' in feat_name.lower():
                feat_data = feat_data / np.pi
            elif 'time' in feat_name.lower():
                max_time = np.nanmax(feat_data) if not np.all(np.isnan(feat_data)) else 1.0
                if max_time > 0:
                    feat_data = feat_data / max_time

            feat_data = np.nan_to_num(feat_data, nan=0.0)
            tensors.append(feat_data)

        result = np.stack(tensors, axis=-1) if tensors else np.zeros((len(data), self.K, 0), dtype=np.float32)
        return torch.from_numpy(result)

    def _prepare_categorical_features(self, data):
        tensors = []
        for feat_name in self.categorical_features:
            cols = [f"{feat_name}_{i}" for i in range(self.K)]
            cols = [c for c in cols if c in data.columns]
            if not cols:
                continue

            feat_data = data[cols].values.astype(np.float32)
            if feat_data.shape[1] < self.K:
                padding = np.zeros((feat_data.shape[0], self.K - feat_data.shape[1]), dtype=np.float32)
                feat_data = np.concatenate([feat_data, padding], axis=1)

            feat_data = np.nan_to_num(feat_data, nan=0.0)
            tensors.append(feat_data)

        result = np.stack(tensors, axis=-1) if tensors else np.zeros((len(data), self.K, 0), dtype=np.float32)
        return torch.from_numpy(result).long()

    def _create_padding_mask(self):
        mask = (self.numerical_tensor.sum(dim=-1) == 0)
        return mask

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.numerical_tensor[idx], self.categorical_tensor[idx], self.padding_mask[idx]


class SoccerRNNWithAttention(nn.Module):
    """V5 Attention Model"""

    def __init__(self, num_numerical_features, categorical_vocab_sizes, embedding_dims,
                 hidden_dim=256, num_layers=2, dropout=0.3, use_lstm=False,
                 bidirectional=True, num_heads=8):
        super(SoccerRNNWithAttention, self).__init__()

        self.embeddings = nn.ModuleDict()
        total_embedding_dim = 0

        for feat_name, vocab_size in categorical_vocab_sizes.items():
            emb_dim = embedding_dims[feat_name]
            self.embeddings[feat_name] = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
            total_embedding_dim += emb_dim

        input_dim = num_numerical_features + total_embedding_dim
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        self.input_norm = nn.LayerNorm(hidden_dim)

        rnn_hidden = hidden_dim
        if use_lstm:
            self.rnn = nn.LSTM(hidden_dim, rnn_hidden, num_layers=num_layers,
                              dropout=dropout if num_layers > 1 else 0,
                              batch_first=True, bidirectional=bidirectional)
        else:
            self.rnn = nn.GRU(hidden_dim, rnn_hidden, num_layers=num_layers,
                             dropout=dropout if num_layers > 1 else 0,
                             batch_first=True, bidirectional=bidirectional)

        rnn_output_dim = rnn_hidden * 2 if bidirectional else rnn_hidden

        self.attention = nn.MultiheadAttention(embed_dim=rnn_output_dim, num_heads=num_heads,
                                              dropout=dropout, batch_first=True)
        self.attention_norm = nn.LayerNorm(rnn_output_dim)

        self.fc = nn.Sequential(
            nn.Linear(rnn_output_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim // 2),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 2)
        )

    def forward(self, num_feat, cat_feat, padding_mask):
        batch_size, seq_len, _ = num_feat.shape

        embedded = []
        for i, feat_name in enumerate(self.embeddings.keys()):
            emb = self.embeddings[feat_name](cat_feat[:, :, i])
            embedded.append(emb)

        if embedded:
            embedded = torch.cat(embedded, dim=-1)
            x = torch.cat([num_feat, embedded], dim=-1)
        else:
            x = num_feat

        x_proj = self.input_projection(x)
        x_proj = self.input_norm(x_proj)

        rnn_out, _ = self.rnn(x_proj)

        attn_out, _ = self.attention(rnn_out, rnn_out, rnn_out, key_padding_mask=padding_mask)
        attn_out = self.attention_norm(attn_out + rnn_out)

        valid_lengths = (~padding_mask).sum(dim=1) - 1
        valid_lengths = valid_lengths.clamp(min=0)
        batch_indices = torch.arange(batch_size, device=attn_out.device)
        last_hidden = attn_out[batch_indices, valid_lengths]

        output = self.fc(last_hidden)
        return output


def load_fold_model(fold_idx):
    """Fold 모델 로드"""
    checkpoint = torch.load(f'lstm_model_v5_fold{fold_idx}.pth', map_location=device)

    vocab_sizes = checkpoint['vocab_sizes']
    embedding_dims = checkpoint['embedding_dims']
    num_numerical_features = checkpoint['num_numerical_features']
    hyperparams = checkpoint['hyperparameters']

    model = SoccerRNNWithAttention(
        num_numerical_features=num_numerical_features,
        categorical_vocab_sizes=vocab_sizes,
        embedding_dims=embedding_dims,
        hidden_dim=hyperparams['hidden_dim'],
        num_layers=hyperparams['num_layers'],
        dropout=hyperparams['dropout'],
        use_lstm=hyperparams['use_lstm'],
        bidirectional=hyperparams['bidirectional'],
        num_heads=hyperparams['num_heads']
    ).to(device)

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    return model, checkpoint


def predict_with_model(model, dataloader):
    """단일 모델로 예측"""
    predictions = []

    with torch.no_grad():
        for num_feat, cat_feat, padding_mask in tqdm(dataloader, desc="Predicting", leave=False):
            num_feat = num_feat.to(device)
            cat_feat = cat_feat.to(device)
            padding_mask = padding_mask.to(device)

            output = model(num_feat, cat_feat, padding_mask)

            # 좌표 복원
            output[:, 0] *= 105.0
            output[:, 1] *= 68.0

            predictions.append(output.cpu().numpy())

    return np.vstack(predictions)


def apply_tta(test_data, test_dataset):
    """Test Time Augmentation (좌우 반전)"""
    # 좌우 반전: y 좌표 반전 (68 - y)
    # 복사본 생성
    augmented_data = test_data.copy()

    # y 좌표 컬럼 찾기
    y_cols = [col for col in augmented_data.columns if 'start_y_' in col or 'end_y_' in col]

    for col in y_cols:
        # NaN이 아닌 값만 반전
        mask = ~augmented_data[col].isna()
        augmented_data.loc[mask, col] = 68.0 - augmented_data.loc[mask, col]

    # Dataset 생성
    aug_dataset = SoccerDatasetV5(
        augmented_data,
        K=test_dataset.K,
        numerical_features=test_dataset.numerical_features,
        categorical_features=test_dataset.categorical_features
    )

    return aug_dataset


def main():
    print("=" * 80)
    print("  LSTM V5 - 5-Fold Ensemble Inference + TTA")
    print("=" * 80)
    print()

    # Test 데이터 로딩
    print("📊 Test 데이터 로딩...")
    test_data = pd.read_csv('processed_test_data_v4.csv')
    print(f"Test 데이터 Shape: {test_data.shape}")
    print()

    # 첫 번째 모델 로드하여 메타 정보 추출
    print("📦 첫 번째 Fold 모델 로딩 (메타 정보 추출)...")
    _, checkpoint = load_fold_model(0)

    numerical_features = checkpoint['numerical_features']
    categorical_features = checkpoint['categorical_features']
    K = checkpoint['hyperparameters']['K']

    # Test Dataset 생성
    print("📦 Test Dataset 생성...")
    test_dataset = SoccerDatasetV5(
        test_data,
        K=K,
        numerical_features=numerical_features,
        categorical_features=categorical_features
    )
    print()

    test_loader = DataLoader(
        test_dataset,
        batch_size=128,
        shuffle=False,
        num_workers=0
    )

    # 5-Fold 예측
    print("🔮 5-Fold 모델 예측 시작...")
    all_predictions = []

    for fold_idx in range(5):
        print(f"\n{'='*60}")
        print(f"  Fold {fold_idx + 1}/5 예측")
        print(f"{'='*60}")

        model, checkpoint = load_fold_model(fold_idx)
        print(f"Val Loss: {checkpoint['val_loss']:.4f}m")

        # 원본 예측
        print("  📍 원본 데이터 예측...")
        pred_original = predict_with_model(model, test_loader)

        # TTA: 좌우 반전
        print("  📍 TTA (좌우 반전) 예측...")
        aug_dataset = apply_tta(test_data, test_dataset)
        aug_loader = DataLoader(aug_dataset, batch_size=128, shuffle=False, num_workers=0)
        pred_flipped = predict_with_model(model, aug_loader)

        # 반전 예측 결과도 다시 반전 (원래 좌표계로 복원)
        pred_flipped[:, 1] = 68.0 - pred_flipped[:, 1]

        # 원본 + TTA 평균
        pred_tta = (pred_original + pred_flipped) / 2.0

        all_predictions.append(pred_tta)

        print(f"  ✅ Fold {fold_idx + 1} 완료")

    # 5개 Fold 평균
    print("\n" + "=" * 80)
    print("  5-Fold 예측 평균 계산...")
    print("=" * 80)

    final_predictions = np.mean(all_predictions, axis=0)
    print(f"✅ 최종 예측 Shape: {final_predictions.shape}")
    print()

    # Submission 생성
    print("📝 Submission 파일 생성...")
    submission = pd.DataFrame({
        'game_episode': test_data['game_episode'],
        'end_x': final_predictions[:, 0],
        'end_y': final_predictions[:, 1]
    })

    submission_file = 'submission_lstm_v5_5fold_tta.csv'
    submission.to_csv(submission_file, index=False)

    print(f"✅ Submission 저장: {submission_file}")
    print()

    # 예측 통계
    print("=" * 80)
    print("  예측 통계")
    print("=" * 80)
    print(f"\nend_x 통계:")
    print(f"   - 최소: {submission['end_x'].min():.2f}")
    print(f"   - 최대: {submission['end_x'].max():.2f}")
    print(f"   - 평균: {submission['end_x'].mean():.2f}")
    print(f"   - 표준편차: {submission['end_x'].std():.2f}")

    print(f"\nend_y 통계:")
    print(f"   - 최소: {submission['end_y'].min():.2f}")
    print(f"   - 최대: {submission['end_y'].max():.2f}")
    print(f"   - 평균: {submission['end_y'].mean():.2f}")
    print(f"   - 표준편차: {submission['end_y'].std():.2f}")

    # 개별 Fold 간 분산도 계산
    predictions_std = np.std(all_predictions, axis=0)
    avg_std_x = np.mean(predictions_std[:, 0])
    avg_std_y = np.mean(predictions_std[:, 1])

    print(f"\n📊 Fold 간 예측 불일치 (표준편차):")
    print(f"   - X 좌표: {avg_std_x:.4f}m")
    print(f"   - Y 좌표: {avg_std_y:.4f}m")

    if avg_std_x < 1.0 and avg_std_y < 1.0:
        print("   ✅ Fold 간 예측이 안정적입니다!")
    else:
        print("   ⚠️ Fold 간 예측 차이가 큽니다. 모델 일반화 성능 확인 필요.")

    print("\n" + "=" * 80)
    print(f"🎉 추론 완료! {submission_file}을 제출하세요.")
    print("=" * 80)

    print("\n📌 기대 성능:")
    print("   - LightGBM V4: 14.138m")
    print("   - LSTM V5 (5-Fold + TTA): ???m")
    print("\n   👉 리더보드 제출 후 성능을 확인하세요!")
    print("=" * 80)


if __name__ == "__main__":
    main()

