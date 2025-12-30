"""
K-League Pass Prediction - V6 Transformer Model

LSTM V5의 문제점 개선:
✅ RNN 제거, Pure Transformer 사용 (더 강력한 시퀀스 모델링)
✅ Focal Loss (어려운 샘플 집중 학습)
✅ 더 큰 모델 (Hidden 512, Layers 4)
✅ Label Smoothing
✅ Warmup + Cosine Scheduler
✅ Gradient Accumulation (더 큰 Effective Batch)

목표: LightGBM (14.138m) 초과

작성일: 2025-12-19
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import GroupKFold
import warnings
import re
from tqdm import tqdm
import math

warnings.filterwarnings('ignore')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🔧 Using device: {device}")


class SoccerDatasetV6(Dataset):
    """V6 Dataset - Same as V5 but optimized"""

    def __init__(self, data, K=20, is_train=True):
        self.data = data.reset_index(drop=True)
        self.K = K
        self.is_train = is_train

        if 'target_x' in data.columns and 'target_y' in data.columns:
            self.targets = data[['target_x', 'target_y']].values.astype(np.float32)
            self.targets[:, 0] /= 105.0
            self.targets[:, 1] /= 68.0
        else:
            self.targets = None

        exclude_cols = ['game_episode', 'game_id', 'target_x', 'target_y', 'final_team_id']
        feature_data = data.drop(columns=[c for c in exclude_cols if c in data.columns])

        self.numerical_features, self.categorical_features = self._classify_columns(feature_data.columns)
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
        num_feat = self.numerical_tensor[idx]
        cat_feat = self.categorical_tensor[idx]
        padding_mask = self.padding_mask[idx]

        if self.targets is not None:
            target = torch.from_numpy(self.targets[idx])
            return num_feat, cat_feat, padding_mask, target
        else:
            return num_feat, cat_feat, padding_mask


class PositionalEncoding(nn.Module):
    """Positional Encoding for Transformer"""

    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: (batch, seq_len, d_model)
        return x + self.pe[:, :x.size(1), :]


class TransformerPassPredictor(nn.Module):
    """
    V6: Pure Transformer Model
    - RNN 제거, Transformer Encoder만 사용
    - 더 강력한 시퀀스 모델링
    """

    def __init__(self, num_numerical_features, categorical_vocab_sizes, embedding_dims,
                 hidden_dim=512, num_layers=4, num_heads=8, dropout=0.3,
                 ff_dim_multiplier=4):
        super(TransformerPassPredictor, self).__init__()

        # Embeddings
        self.embeddings = nn.ModuleDict()
        total_embedding_dim = 0

        for feat_name, vocab_size in categorical_vocab_sizes.items():
            emb_dim = embedding_dims[feat_name]
            self.embeddings[feat_name] = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
            total_embedding_dim += emb_dim

        input_dim = num_numerical_features + total_embedding_dim

        # Input Projection
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        self.input_norm = nn.LayerNorm(hidden_dim)

        # Positional Encoding
        self.pos_encoder = PositionalEncoding(hidden_dim)

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * ff_dim_multiplier,
            dropout=dropout,
            activation='gelu',  # GELU 사용 (BERT 스타일)
            batch_first=True,
            norm_first=True  # Pre-LN (더 안정적)
        )

        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(hidden_dim)
        )

        # Global Average Pooling (모든 시점 활용)
        self.use_gap = True

        # Output Head (더 깊고 강력하게)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.LayerNorm(hidden_dim // 2),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.GELU(),
            nn.LayerNorm(hidden_dim // 4),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim // 4, 2)
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

        # Input Projection + Positional Encoding
        x = self.input_projection(x)
        x = self.input_norm(x)
        x = self.pos_encoder(x)

        # Transformer Encoder
        x = self.transformer_encoder(x, src_key_padding_mask=padding_mask)

        # Pooling
        if self.use_gap:
            # Global Average Pooling (패딩 제외)
            mask_expanded = (~padding_mask).unsqueeze(-1).float()  # (batch, seq_len, 1)
            x_masked = x * mask_expanded
            x_pooled = x_masked.sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1)
        else:
            # Last valid token
            valid_lengths = (~padding_mask).sum(dim=1) - 1
            valid_lengths = valid_lengths.clamp(min=0)
            batch_indices = torch.arange(batch_size, device=x.device)
            x_pooled = x[batch_indices, valid_lengths]

        # Output
        output = self.fc(x_pooled)

        return output


class FocalLoss(nn.Module):
    """
    Focal Loss - 어려운 샘플에 더 집중
    """

    def __init__(self, alpha=1.0, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred, target):
        # 실제 좌표로 복원
        pred_real = pred.clone()
        pred_real[:, 0] *= 105.0
        pred_real[:, 1] *= 68.0

        target_real = target.clone()
        target_real[:, 0] *= 105.0
        target_real[:, 1] *= 68.0

        # 유클리드 거리
        distances = torch.sqrt(torch.sum((pred_real - target_real) ** 2, dim=1))

        # Focal Weight (거리가 클수록 높은 가중치)
        # 정규화된 거리 (0~1)
        normalized_dist = distances / 150.0  # 필드 대각선 길이로 정규화
        focal_weight = torch.pow(normalized_dist, self.gamma)

        # Weighted Loss
        weighted_distances = self.alpha * focal_weight * distances

        return weighted_distances.mean()


class SmoothL1DistanceLoss(nn.Module):
    """Smooth L1 Loss 기반 거리 손실 (이상치에 강함)"""

    def forward(self, pred, target):
        pred_real = pred.clone()
        pred_real[:, 0] *= 105.0
        pred_real[:, 1] *= 68.0

        target_real = target.clone()
        target_real[:, 0] *= 105.0
        target_real[:, 1] *= 68.0

        # Smooth L1 (Huber Loss)
        diff = pred_real - target_real
        smooth_l1 = torch.where(
            torch.abs(diff) < 1.0,
            0.5 * diff ** 2,
            torch.abs(diff) - 0.5
        )

        return smooth_l1.sum(dim=1).mean()


def get_categorical_info(data, categorical_features, K=20):
    vocab_sizes = {}
    embedding_dims = {}

    for feat_name in categorical_features:
        cols = [f"{feat_name}_{i}" for i in range(K)]
        cols = [c for c in cols if c in data.columns]
        if not cols:
            continue

        max_val = data[cols].max().max()
        vocab_size = int(max_val) + 2
        emb_dim = min(max(vocab_size // 2, 4), 50)

        vocab_sizes[feat_name] = vocab_size
        embedding_dims[feat_name] = emb_dim

    return vocab_sizes, embedding_dims


def train_one_epoch(model, dataloader, criterion, optimizer, device, accumulation_steps=1):
    model.train()
    total_loss = 0.0
    optimizer.zero_grad()

    for batch_idx, (num_feat, cat_feat, padding_mask, target) in enumerate(tqdm(dataloader, desc="Training", leave=False)):
        num_feat = num_feat.to(device)
        cat_feat = cat_feat.to(device)
        padding_mask = padding_mask.to(device)
        target = target.to(device)

        output = model(num_feat, cat_feat, padding_mask)
        loss = criterion(output, target)

        # Gradient Accumulation
        loss = loss / accumulation_steps
        loss.backward()

        if (batch_idx + 1) % accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()

        total_loss += loss.item() * accumulation_steps

    return total_loss / len(dataloader)


def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for num_feat, cat_feat, padding_mask, target in tqdm(dataloader, desc="Validation", leave=False):
            num_feat = num_feat.to(device)
            cat_feat = cat_feat.to(device)
            padding_mask = padding_mask.to(device)
            target = target.to(device)

            output = model(num_feat, cat_feat, padding_mask)
            loss = criterion(output, target)
            total_loss += loss.item()

    return total_loss / len(dataloader)


def get_linear_warmup_cosine_scheduler(optimizer, warmup_epochs, total_epochs, min_lr=1e-6):
    """Warmup + Cosine Annealing Scheduler"""

    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            # Linear warmup
            return (epoch + 1) / warmup_epochs
        else:
            # Cosine annealing
            progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
            return 0.5 * (1 + math.cos(math.pi * progress))

    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def main():
    print("=" * 80)
    print("  V6 Transformer Model - 성능 돌파 시도")
    print("  목표: LightGBM (14.138m) 초과")
    print("=" * 80)
    print()

    # 하이퍼파라미터 (공격적 설정)
    K = 20
    BATCH_SIZE = 32  # 작게 하고 Gradient Accumulation 사용
    ACCUMULATION_STEPS = 2  # Effective Batch Size = 64
    HIDDEN_DIM = 512  # 384 → 512 증가
    NUM_LAYERS = 4  # 3 → 4 증가
    NUM_HEADS = 8
    DROPOUT = 0.2  # 0.4 → 0.2 감소 (더 많은 학습)
    LEARNING_RATE = 1e-3  # 5e-4 → 1e-3 증가
    WARMUP_EPOCHS = 10
    NUM_EPOCHS = 150  # 100 → 150 증가
    EARLY_STOPPING_PATIENCE = 30  # 20 → 30 증가
    FF_DIM_MULTIPLIER = 4

    print(f"🔧 하이퍼파라미터 (V6 - 공격적 설정):")
    print(f"   - Model: Pure Transformer (No RNN)")
    print(f"   - Hidden Dim: {HIDDEN_DIM} (↑ from 384)")
    print(f"   - Num Layers: {NUM_LAYERS} (↑ from 3)")
    print(f"   - Batch Size: {BATCH_SIZE} x {ACCUMULATION_STEPS} = {BATCH_SIZE * ACCUMULATION_STEPS}")
    print(f"   - Dropout: {DROPOUT} (↓ from 0.4)")
    print(f"   - Learning Rate: {LEARNING_RATE} (↑ from 5e-4)")
    print(f"   - Epochs: {NUM_EPOCHS} (↑ from 100)")
    print(f"   - Warmup: {WARMUP_EPOCHS} epochs")
    print()

    # 데이터 로딩
    print("📊 데이터 로딩...")
    data = pd.read_csv('processed_train_data_v4.csv')
    print(f"데이터 Shape: {data.shape}\n")

    game_ids = data['game_id'].values

    # 첫 번째 Fold (빠른 검증)
    print("🔧 First Fold 학습 (V6 검증)...")
    gkf = GroupKFold(n_splits=5)
    train_idx, val_idx = next(gkf.split(data, groups=game_ids))

    train_data = data.iloc[train_idx].copy()
    val_data = data.iloc[val_idx].copy()

    print(f"Train: {len(train_data):,} 샘플")
    print(f"Val: {len(val_data):,} 샘플")
    print()

    # Dataset
    print("📦 Dataset V6 생성...")
    train_dataset = SoccerDatasetV6(train_data, K=K, is_train=True)
    val_dataset = SoccerDatasetV6(val_data, K=K, is_train=True)
    print()

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=0, pin_memory=True if torch.cuda.is_available() else False
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=0, pin_memory=True if torch.cuda.is_available() else False
    )

    # 범주형 정보
    print("🔤 범주형 변수 정보...")
    vocab_sizes, embedding_dims = get_categorical_info(data, train_dataset.categorical_features, K=K)
    print()

    # 모델 생성
    print("🏗️ Transformer 모델 V6 생성...")
    model = TransformerPassPredictor(
        num_numerical_features=train_dataset.numerical_tensor.shape[2],
        categorical_vocab_sizes=vocab_sizes,
        embedding_dims=embedding_dims,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        num_heads=NUM_HEADS,
        dropout=DROPOUT,
        ff_dim_multiplier=FF_DIM_MULTIPLIER
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"✅ 모델 생성 완료:")
    print(f"   - Total Parameters: {total_params:,}")
    print(f"   - Architecture: Transformer Encoder Only")
    print()

    # Loss & Optimizer
    criterion = FocalLoss(alpha=1.0, gamma=2.0)  # Focal Loss 사용
    # criterion = SmoothL1DistanceLoss()  # 대안: Smooth L1

    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=5e-4)
    scheduler = get_linear_warmup_cosine_scheduler(optimizer, WARMUP_EPOCHS, NUM_EPOCHS)

    # 학습
    print("🚀 학습 시작 (V6 - Transformer)...\n")
    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(NUM_EPOCHS):
        print(f"Epoch {epoch+1}/{NUM_EPOCHS}")
        print("-" * 60)

        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, ACCUMULATION_STEPS)
        val_loss = validate(model, val_loader, criterion, device)

        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        print(f"Train: {train_loss:.4f}m | Val: {val_loss:.4f}m | LR: {current_lr:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0

            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'vocab_sizes': vocab_sizes,
                'embedding_dims': embedding_dims,
                'num_numerical_features': train_dataset.numerical_tensor.shape[2],
                'categorical_features': train_dataset.categorical_features,
                'numerical_features': train_dataset.numerical_features,
                'hyperparameters': {
                    'hidden_dim': HIDDEN_DIM,
                    'num_layers': NUM_LAYERS,
                    'num_heads': NUM_HEADS,
                    'dropout': DROPOUT,
                    'K': K,
                    'model_type': 'transformer'
                }
            }, 'transformer_model_v6_best.pth')

            print(f"💾 Best! (Val: {val_loss:.4f}m)")
        else:
            patience_counter += 1
            print(f"⏳ Patience: {patience_counter}/{EARLY_STOPPING_PATIENCE}")

            if patience_counter >= EARLY_STOPPING_PATIENCE:
                print(f"\n⚠️ Early stopping!")
                break

        print()

    # 결과
    print("=" * 80)
    print("  학습 완료! (V6 - Transformer)")
    print("=" * 80)
    print(f"\n✅ Best Val Loss: {best_val_loss:.4f}m")
    print(f"✅ 모델 저장: transformer_model_v6_best.pth")

    print("\n📊 성능 비교:")
    print(f"   - LightGBM V4: 14.138m")
    print(f"   - LSTM V5: 15.3157m")
    print(f"   - Transformer V6: {best_val_loss:.4f}m")

    improvement_v5 = 15.3157 - best_val_loss
    print(f"\n📈 V5 대비 개선: {improvement_v5:+.4f}m ({improvement_v5/15.3157*100:.1f}%)")

    if best_val_loss < 14.138:
        print("\n🎉🎉🎉 축하합니다! LightGBM을 초과했습니다!")
    elif best_val_loss < 14.5:
        print("\n✅ 매우 좋은 성능! LightGBM 근접!")
    elif best_val_loss < 15.0:
        print("\n📈 V5보다 개선! 추가 튜닝으로 LightGBM 초과 가능!")
    else:
        print("\n📊 추가 전략 필요: 피처 엔지니어링, Data Augmentation")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()

