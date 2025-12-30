"""
K-League Pass Prediction - LSTM V5 Inference (Attention Model)

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
    """Test용 Dataset (V5)"""

    def __init__(self, data, K=20, numerical_features=None, categorical_features=None):
        self.data = data.reset_index(drop=True)
        self.K = K

        # 메타 정보 제외
        exclude_cols = ['game_episode', 'game_id']
        feature_data = data.drop(columns=[c for c in exclude_cols if c in data.columns])

        # 피처 리스트가 제공되지 않으면 자동 분류
        if numerical_features is None or categorical_features is None:
            self.numerical_features, self.categorical_features = self._classify_columns(feature_data.columns)
        else:
            self.numerical_features = numerical_features
            self.categorical_features = categorical_features

        # 3D 텐서 변환
        self.numerical_tensor = self._prepare_numerical_features(feature_data)
        self.categorical_tensor = self._prepare_categorical_features(feature_data)
        self.padding_mask = self._create_padding_mask()

        print(f"✅ Test Dataset 준비 완료:")
        print(f"   - 샘플 수: {len(self.data)}")
        print(f"   - 수치형: {self.numerical_tensor.shape}")
        print(f"   - 범주형: {self.categorical_tensor.shape}")
        print(f"   - Padding Mask: {self.padding_mask.shape}")

    def _classify_columns(self, columns):
        """컬럼 분류"""
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
        """수치형 피처 변환 + 정규화"""
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

            # 정규화
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
        """범주형 피처 변환"""
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
        """Padding Mask 생성"""
        mask = (self.numerical_tensor.sum(dim=-1) == 0)
        return mask

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.numerical_tensor[idx], self.categorical_tensor[idx], self.padding_mask[idx]


class SoccerRNNWithAttention(nn.Module):
    """V5 Attention Model"""

    def __init__(self,
                 num_numerical_features,
                 categorical_vocab_sizes,
                 embedding_dims,
                 hidden_dim=256,
                 num_layers=2,
                 dropout=0.3,
                 use_lstm=False,
                 bidirectional=True,
                 num_heads=8):
        super(SoccerRNNWithAttention, self).__init__()

        self.num_numerical_features = num_numerical_features
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.use_lstm = use_lstm
        self.bidirectional = bidirectional
        self.num_heads = num_heads

        # Embeddings
        self.embeddings = nn.ModuleDict()
        total_embedding_dim = 0

        for feat_name, vocab_size in categorical_vocab_sizes.items():
            emb_dim = embedding_dims[feat_name]
            self.embeddings[feat_name] = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
            total_embedding_dim += emb_dim

        input_dim = num_numerical_features + total_embedding_dim

        # Layers
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        self.input_norm = nn.LayerNorm(hidden_dim)

        rnn_hidden = hidden_dim
        if use_lstm:
            self.rnn = nn.LSTM(
                hidden_dim, rnn_hidden,
                num_layers=num_layers,
                dropout=dropout if num_layers > 1 else 0,
                batch_first=True,
                bidirectional=bidirectional
            )
        else:
            self.rnn = nn.GRU(
                hidden_dim, rnn_hidden,
                num_layers=num_layers,
                dropout=dropout if num_layers > 1 else 0,
                batch_first=True,
                bidirectional=bidirectional
            )

        rnn_output_dim = rnn_hidden * 2 if bidirectional else rnn_hidden

        self.attention = nn.MultiheadAttention(
            embed_dim=rnn_output_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

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

        # Embedding
        embedded = []
        for i, feat_name in enumerate(self.embeddings.keys()):
            emb = self.embeddings[feat_name](cat_feat[:, :, i])
            embedded.append(emb)

        if embedded:
            embedded = torch.cat(embedded, dim=-1)
            x = torch.cat([num_feat, embedded], dim=-1)
        else:
            x = num_feat

        # Projection
        x_proj = self.input_projection(x)
        x_proj = self.input_norm(x_proj)

        # RNN
        rnn_out, _ = self.rnn(x_proj)

        # Attention
        attn_out, _ = self.attention(
            rnn_out, rnn_out, rnn_out,
            key_padding_mask=padding_mask
        )

        # Residual + Norm
        attn_out = self.attention_norm(attn_out + rnn_out)

        # Last valid hidden state
        valid_lengths = (~padding_mask).sum(dim=1) - 1
        valid_lengths = valid_lengths.clamp(min=0)

        batch_indices = torch.arange(batch_size, device=attn_out.device)
        last_hidden = attn_out[batch_indices, valid_lengths]

        # Output
        output = self.fc(last_hidden)

        return output


def main():
    print("=" * 80)
    print("  LSTM V5 (Attention) - Inference")
    print("=" * 80)
    print()

    # 1. 체크포인트 로드
    print("📦 모델 체크포인트 로딩...")
    checkpoint = torch.load('lstm_model_v5_attention_best.pth', map_location=device)

    vocab_sizes = checkpoint['vocab_sizes']
    embedding_dims = checkpoint['embedding_dims']
    num_numerical_features = checkpoint['num_numerical_features']
    categorical_features = checkpoint['categorical_features']
    numerical_features = checkpoint['numerical_features']
    hyperparams = checkpoint['hyperparameters']

    print(f"✅ 체크포인트 정보:")
    print(f"   - Epoch: {checkpoint['epoch']}")
    print(f"   - Val Loss: {checkpoint['val_loss']:.4f}m")
    print(f"   - Hidden Dim: {hyperparams['hidden_dim']}")
    print(f"   - Num Layers: {hyperparams['num_layers']}")
    print(f"   - Bidirectional: {hyperparams['bidirectional']}")
    print(f"   - Attention Heads: {hyperparams['num_heads']}")
    print()

    # 2. 모델 생성
    print("🏗️ 모델 생성 중...")
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

    print(f"✅ 모델 로드 완료")
    print()

    # 3. Test 데이터 로딩
    print("📊 Test 데이터 로딩...")
    test_data = pd.read_csv('processed_test_data_v4.csv')
    print(f"Test 데이터 Shape: {test_data.shape}")
    print()

    # 4. Test Dataset 생성
    print("📦 Test Dataset 생성...")
    test_dataset = SoccerDatasetV5(
        test_data,
        K=hyperparams['K'],
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

    # 5. 추론
    print("🔮 추론 시작...")
    predictions = []

    with torch.no_grad():
        for num_feat, cat_feat, padding_mask in tqdm(test_loader, desc="Predicting"):
            num_feat = num_feat.to(device)
            cat_feat = cat_feat.to(device)
            padding_mask = padding_mask.to(device)

            output = model(num_feat, cat_feat, padding_mask)

            # 좌표 복원 (0~1 → 실제 좌표)
            output[:, 0] *= 105.0
            output[:, 1] *= 68.0

            predictions.append(output.cpu().numpy())

    predictions = np.vstack(predictions)
    print(f"✅ 추론 완료: {predictions.shape}")
    print()

    # 6. Submission 생성
    print("📝 Submission 파일 생성...")
    submission = pd.DataFrame({
        'game_episode': test_data['game_episode'],
        'end_x': predictions[:, 0],
        'end_y': predictions[:, 1]
    })

    submission_file = 'submission_lstm_v5_attention.csv'
    submission.to_csv(submission_file, index=False)

    print(f"✅ Submission 저장: {submission_file}")
    print()

    # 7. 예측 통계
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

    print("\n" + "=" * 80)
    print(f"🎉 추론 완료! {submission_file}을 제출하세요.")
    print("=" * 80)


if __name__ == "__main__":
    main()

