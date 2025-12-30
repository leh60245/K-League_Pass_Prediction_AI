"""
K-League Pass Prediction - Transformer with Attention

🎯 핵심 개선:
1. Self-Attention으로 중요한 이벤트 자동 선택
2. Positional Encoding으로 시간 정보 명시적 반영
3. Multi-Head Attention으로 다양한 패턴 학습

작성일: 2025-12-18
"""

import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    """위치 정보를 인코딩 (시간 순서 정보)"""

    def __init__(self, d_model, max_len=20, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Positional encoding 생성
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, d_model)
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TransformerPassPredictor(nn.Module):
    """
    Transformer 기반 패스 예측 모델

    Architecture:
    1. Input Embedding (Numerical + Categorical)
    2. Positional Encoding
    3. Transformer Encoder (Multi-Head Attention)
    4. Global Pooling (마지막 시점 + attention-weighted average)
    5. Output Head
    """

    def __init__(self,
                 num_numerical_features,
                 categorical_vocab_sizes,
                 embedding_dims,
                 d_model=128,
                 nhead=4,
                 num_layers=3,
                 dim_feedforward=512,
                 dropout=0.3):
        """
        Args:
            d_model: Transformer hidden dimension
            nhead: Multi-head attention의 head 수
            num_layers: Transformer encoder layer 수
            dim_feedforward: FFN hidden dimension
        """
        super().__init__()

        self.d_model = d_model

        # Embeddings
        self.embeddings = nn.ModuleDict()
        total_embedding_dim = 0

        for feat_name, vocab_size in categorical_vocab_sizes.items():
            emb_dim = embedding_dims[feat_name]
            self.embeddings[feat_name] = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
            total_embedding_dim += emb_dim

        input_dim = num_numerical_features + total_embedding_dim

        # Input Projection
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Positional Encoding
        self.pos_encoder = PositionalEncoding(d_model, max_len=20, dropout=dropout)

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation='gelu'  # GELU는 Transformer에서 더 효과적
        )

        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(d_model)
        )

        # Attention Pooling (중요한 시점에 가중치)
        self.attention_weight = nn.Linear(d_model, 1)

        # Output Head
        self.fc = nn.Sequential(
            nn.Linear(d_model * 2, d_model),  # *2: last + pooled
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 2)  # (target_x, target_y)
        )

    def forward(self, num_feat, cat_feat, src_key_padding_mask=None):
        """
        Args:
            num_feat: (batch, seq_len, num_features)
            cat_feat: (batch, seq_len, cat_features)
            src_key_padding_mask: (batch, seq_len) - True for padding positions

        Returns:
            (batch, 2)
        """
        batch_size, seq_len, _ = num_feat.shape

        # Embedding
        embedded = []
        for i, feat_name in enumerate(self.embeddings.keys()):
            emb = self.embeddings[feat_name](cat_feat[:, :, i])
            embedded.append(emb)

        # Concatenate
        if embedded:
            embedded = torch.cat(embedded, dim=-1)
            x = torch.cat([num_feat, embedded], dim=-1)
        else:
            x = num_feat

        # Input Projection
        x = self.input_projection(x)  # (batch, seq_len, d_model)

        # Positional Encoding
        x = self.pos_encoder(x)

        # Transformer Encoder
        # src_key_padding_mask: True면 해당 위치를 attention에서 무시
        encoder_output = self.transformer_encoder(
            x,
            src_key_padding_mask=src_key_padding_mask
        )  # (batch, seq_len, d_model)

        # 1. 마지막 시점 (LSTM처럼)
        last_output = encoder_output[:, -1, :]  # (batch, d_model)

        # 2. Attention-weighted pooling (중요한 이벤트에 가중치)
        attention_scores = self.attention_weight(encoder_output)  # (batch, seq_len, 1)

        if src_key_padding_mask is not None:
            # 패딩 위치는 attention에서 제외
            attention_scores = attention_scores.masked_fill(
                src_key_padding_mask.unsqueeze(-1), float('-inf')
            )

        attention_weights = torch.softmax(attention_scores, dim=1)  # (batch, seq_len, 1)
        pooled_output = torch.sum(encoder_output * attention_weights, dim=1)  # (batch, d_model)

        # 3. Concatenate (마지막 + 가중 평균)
        combined = torch.cat([last_output, pooled_output], dim=-1)  # (batch, d_model*2)

        # Output
        output = self.fc(combined)  # (batch, 2)

        return output


class SoccerTransformerDataset(torch.utils.data.Dataset):
    """
    Transformer용 Dataset
    - 패딩 마스크 생성
    """

    def __init__(self, sequences, targets, max_seq_len=20):
        self.sequences = sequences
        self.targets = targets
        self.max_seq_len = max_seq_len

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]  # (seq_len, features)
        target = self.targets[idx]  # (2,)

        # 패딩
        seq_len = len(seq)
        if seq_len < self.max_seq_len:
            padding = torch.zeros(self.max_seq_len - seq_len, seq.shape[1])
            seq = torch.cat([padding, seq], dim=0)

            # 패딩 마스크 (True = 패딩)
            mask = torch.cat([
                torch.ones(self.max_seq_len - seq_len, dtype=torch.bool),
                torch.zeros(seq_len, dtype=torch.bool)
            ])
        else:
            seq = seq[-self.max_seq_len:]  # 마지막 20개만
            mask = torch.zeros(self.max_seq_len, dtype=torch.bool)

        return seq, target, mask


# Loss 함수 - MSE + Huber 조합
class CombinedLoss(nn.Module):
    """
    MSE + Huber Loss 조합
    - MSE: 전반적인 정확도
    - Huber: 이상치에 robust
    """

    def __init__(self, mse_weight=0.5, huber_weight=0.5, delta=1.0):
        super().__init__()
        self.mse_weight = mse_weight
        self.huber_weight = huber_weight
        self.mse = nn.MSELoss()
        self.huber = nn.HuberLoss(delta=delta)

    def forward(self, pred, target):
        # 정규화된 좌표 그대로 계산
        mse_loss = self.mse(pred, target)
        huber_loss = self.huber(pred, target)

        return self.mse_weight * mse_loss + self.huber_weight * huber_loss


def evaluate_euclidean(model, dataloader, device):
    """유클리드 거리로 평가"""
    model.eval()
    total_dist = 0.0
    count = 0

    with torch.no_grad():
        for seq, target, mask in dataloader:
            seq = seq.to(device)
            target = target.to(device)
            mask = mask.to(device)

            output = model(seq[:, :, :-6], seq[:, :, -6:].long(), mask)

            # 실제 좌표로 복원
            output_real = output.clone()
            output_real[:, 0] *= 105.0
            output_real[:, 1] *= 68.0

            target_real = target.clone()
            target_real[:, 0] *= 105.0
            target_real[:, 1] *= 68.0

            dist = torch.sqrt(torch.sum((output_real - target_real) ** 2, dim=1))
            total_dist += dist.sum().item()
            count += len(seq)

    return total_dist / count


# 사용 예시
"""
# 1. 데이터 로딩 (Long format)
with open('train_sequences_long.pkl', 'rb') as f:
    train_data = pickle.load(f)

# 2. 모델 생성
model = TransformerPassPredictor(
    num_numerical_features=12,
    categorical_vocab_sizes={'type_id': 50, 'result_id': 20, ...},
    embedding_dims={'type_id': 16, 'result_id': 8, ...},
    d_model=128,
    nhead=4,
    num_layers=3,
    dropout=0.3
)

# 3. 학습
criterion = CombinedLoss(mse_weight=0.5, huber_weight=0.5)
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)
"""

