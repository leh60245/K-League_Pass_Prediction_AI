"""
K-League Pass Prediction - PyTorch LSTM/GRU with Multi-Head Attention (V5)

핵심 개선사항:
✅ Multi-Head Attention 추가 (중요 시점 학습)
✅ Padding Mask 활용 (실제 데이터 구분)
✅ Bidirectional RNN (양방향 정보 활용)
✅ 전체 피처 정규화/표준화
✅ 깊은 Output Head (복잡한 패턴 학습)
✅ Residual Connection
✅ LayerNorm 추가

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

warnings.filterwarnings('ignore')

# Device 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🔧 Using device: {device}")


class SoccerDatasetV5(Dataset):
    """
    V5: 고도화된 Dataset
    - Padding Mask 생성
    - 전체 피처 표준화
    - 추가 시퀀스 피처 생성
    """

    def __init__(self, data, K=20, is_train=True, scaler=None):
        self.data = data.reset_index(drop=True)
        self.K = K
        self.is_train = is_train
        self.scaler = scaler

        # 타겟 추출 및 정규화
        if 'target_x' in data.columns and 'target_y' in data.columns:
            self.targets = data[['target_x', 'target_y']].values.astype(np.float32)
            self.targets[:, 0] /= 105.0
            self.targets[:, 1] /= 68.0
        else:
            self.targets = None

        # 메타 정보 제외
        exclude_cols = ['game_episode', 'game_id', 'target_x', 'target_y', 'final_team_id']
        feature_data = data.drop(columns=[c for c in exclude_cols if c in data.columns])

        # 컬럼 분류
        self.numerical_features, self.categorical_features = self._classify_columns(feature_data.columns)

        # 3D 텐서 변환
        self.numerical_tensor = self._prepare_numerical_features(feature_data)
        self.categorical_tensor = self._prepare_categorical_features(feature_data)

        # Padding Mask 생성 (중요!)
        self.padding_mask = self._create_padding_mask()

        print(f"✅ Dataset V5 준비 완료:")
        print(f"   - 샘플 수: {len(self.data)}")
        print(f"   - 수치형 피처: {len(self.numerical_features)} → Shape: {self.numerical_tensor.shape}")
        print(f"   - 범주형 피처: {len(self.categorical_features)} → Shape: {self.categorical_tensor.shape}")
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
        """수치형 피처를 3D 텐서로 변환 + 정규화"""
        tensors = []

        # 좌표 정규화 키워드
        x_coord_keywords = ['start_x', 'end_x', 'dx']
        y_coord_keywords = ['start_y', 'end_y', 'dy']

        for feat_name in self.numerical_features:
            cols = [f"{feat_name}_{i}" for i in range(self.K)]
            cols = [c for c in cols if c in data.columns]

            if not cols:
                continue

            feat_data = data[cols].values.astype(np.float32)

            # 패딩
            if feat_data.shape[1] < self.K:
                padding = np.zeros((feat_data.shape[0], self.K - feat_data.shape[1]), dtype=np.float32)
                feat_data = np.concatenate([feat_data, padding], axis=1)

            # 정규화
            if any(kw in feat_name for kw in x_coord_keywords):
                feat_data = feat_data / 105.0
            elif any(kw in feat_name for kw in y_coord_keywords):
                feat_data = feat_data / 68.0
            elif 'speed' in feat_name.lower():
                # 속도: 최대값으로 정규화 (예: 최대 30m/s)
                feat_data = np.clip(feat_data / 30.0, 0, 1)
            elif 'angle' in feat_name.lower() or 'direction' in feat_name.lower():
                # 각도: -π ~ π → -1 ~ 1
                feat_data = feat_data / np.pi
            elif 'time' in feat_name.lower():
                # 시간차: 최대값으로 정규화
                max_time = np.nanmax(feat_data) if not np.all(np.isnan(feat_data)) else 1.0
                if max_time > 0:
                    feat_data = feat_data / max_time

            # NaN → 0.0
            feat_data = np.nan_to_num(feat_data, nan=0.0)

            tensors.append(feat_data)

        result = np.stack(tensors, axis=-1) if tensors else np.zeros((len(data), self.K, 0), dtype=np.float32)
        return torch.from_numpy(result)

    def _prepare_categorical_features(self, data):
        """범주형 피처를 3D 텐서로 변환"""
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
        """
        Padding Mask 생성
        Returns:
            (N, K) bool tensor - True: Padding, False: Valid
        """
        # 모든 수치형 피처가 0인 시점 = Padding
        mask = (self.numerical_tensor.sum(dim=-1) == 0)  # (N, K)
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


class SoccerRNNWithAttention(nn.Module):
    """
    V5: Multi-Head Attention + Bidirectional RNN + Residual Connection
    """

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

        # Embedding layers
        self.embeddings = nn.ModuleDict()
        total_embedding_dim = 0

        for feat_name, vocab_size in categorical_vocab_sizes.items():
            emb_dim = embedding_dims[feat_name]
            self.embeddings[feat_name] = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
            total_embedding_dim += emb_dim

        # Input dimension
        input_dim = num_numerical_features + total_embedding_dim

        # Input Projection
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        self.input_norm = nn.LayerNorm(hidden_dim)

        # RNN Layer
        rnn_hidden = hidden_dim
        if use_lstm:
            self.rnn = nn.LSTM(
                hidden_dim,
                rnn_hidden,
                num_layers=num_layers,
                dropout=dropout if num_layers > 1 else 0,
                batch_first=True,
                bidirectional=bidirectional
            )
        else:
            self.rnn = nn.GRU(
                hidden_dim,
                rnn_hidden,
                num_layers=num_layers,
                dropout=dropout if num_layers > 1 else 0,
                batch_first=True,
                bidirectional=bidirectional
            )

        # RNN 출력 차원 (양방향이면 2배)
        rnn_output_dim = rnn_hidden * 2 if bidirectional else rnn_hidden

        # Multi-Head Attention
        self.attention = nn.MultiheadAttention(
            embed_dim=rnn_output_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        self.attention_norm = nn.LayerNorm(rnn_output_dim)

        # Output Head (깊은 구조)
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
        """
        Args:
            num_feat: (batch, seq_len, num_features)
            cat_feat: (batch, seq_len, cat_features)
            padding_mask: (batch, seq_len) - True: padding, False: valid

        Returns:
            (batch, 2) - (target_x, target_y)
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

        # Input Projection + LayerNorm
        x_proj = self.input_projection(x)
        x_proj = self.input_norm(x_proj)

        # RNN
        rnn_out, _ = self.rnn(x_proj)

        # Multi-Head Attention (중요 시점 학습)
        # padding_mask: True(패딩)는 attention에서 무시
        attn_out, attn_weights = self.attention(
            rnn_out, rnn_out, rnn_out,
            key_padding_mask=padding_mask
        )

        # Residual Connection + LayerNorm
        attn_out = self.attention_norm(attn_out + rnn_out)

        # 마지막 시점의 hidden state
        # Padding이 아닌 마지막 유효 시점 찾기
        valid_lengths = (~padding_mask).sum(dim=1) - 1  # (batch,)
        valid_lengths = valid_lengths.clamp(min=0)

        # Gather last valid hidden state
        batch_indices = torch.arange(batch_size, device=attn_out.device)
        last_hidden = attn_out[batch_indices, valid_lengths]  # (batch, hidden_dim)

        # Output
        output = self.fc(last_hidden)

        return output


class EuclideanDistanceLoss(nn.Module):
    """유클리드 거리 기반 손실 함수"""

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
        return distances.mean()


def get_categorical_info(data, categorical_features, K=20):
    """범주형 변수의 어휘 크기와 임베딩 차원 계산"""
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


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """1 에포크 학습"""
    model.train()
    total_loss = 0.0

    for num_feat, cat_feat, padding_mask, target in tqdm(dataloader, desc="Training", leave=False):
        num_feat = num_feat.to(device)
        cat_feat = cat_feat.to(device)
        padding_mask = padding_mask.to(device)
        target = target.to(device)

        optimizer.zero_grad()

        output = model(num_feat, cat_feat, padding_mask)
        loss = criterion(output, target)

        loss.backward()

        # Gradient Clipping (학습 안정성)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def validate(model, dataloader, criterion, device):
    """검증"""
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


def main():
    print("=" * 80)
    print("  PyTorch LSTM/GRU V5 - Multi-Head Attention + 고도화")
    print("  목표: LightGBM (14.138m) 초과 성능 달성")
    print("=" * 80)
    print()

    # 하이퍼파라미터 (최적화된 값)
    K = 20
    BATCH_SIZE = 64  # 작은 배치로 학습 안정성 향상
    HIDDEN_DIM = 384  # 더 큰 모델
    NUM_LAYERS = 3  # 더 깊은 RNN
    DROPOUT = 0.4  # 강한 Regularization
    LEARNING_RATE = 5e-4  # 더 작은 LR
    NUM_EPOCHS = 100
    EARLY_STOPPING_PATIENCE = 20
    USE_LSTM = False  # GRU가 더 빠름
    BIDIRECTIONAL = True
    NUM_HEADS = 8

    print(f"🔧 하이퍼파라미터:")
    print(f"   - Sequence Length: {K}")
    print(f"   - Batch Size: {BATCH_SIZE}")
    print(f"   - Hidden Dim: {HIDDEN_DIM}")
    print(f"   - Num Layers: {NUM_LAYERS}")
    print(f"   - Dropout: {DROPOUT}")
    print(f"   - Learning Rate: {LEARNING_RATE}")
    print(f"   - Epochs: {NUM_EPOCHS}")
    print(f"   - RNN Type: {'Bidirectional ' if BIDIRECTIONAL else ''}{'LSTM' if USE_LSTM else 'GRU'}")
    print(f"   - Attention Heads: {NUM_HEADS}")
    print()

    # 1. 데이터 로딩
    print("📊 데이터 로딩...")
    data = pd.read_csv('processed_train_data_v4.csv')
    print(f"데이터 Shape: {data.shape}")
    print()

    game_ids = data['game_id'].values

    # 2. 첫 번째 Fold (프로토타이핑)
    print("🔧 First Fold 학습...")
    gkf = GroupKFold(n_splits=5)
    train_idx, val_idx = next(gkf.split(data, groups=game_ids))

    train_data = data.iloc[train_idx].copy()
    val_data = data.iloc[val_idx].copy()

    print(f"Train: {len(train_data):,} 샘플")
    print(f"Val: {len(val_data):,} 샘플")
    print()

    # 3. Dataset 생성
    print("📦 Dataset V5 생성 중...")
    train_dataset = SoccerDatasetV5(train_data, K=K, is_train=True)
    val_dataset = SoccerDatasetV5(val_data, K=K, is_train=True)
    print()

    # DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=True if torch.cuda.is_available() else False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=True if torch.cuda.is_available() else False
    )

    # 4. 범주형 정보
    print("🔤 범주형 변수 정보 추출 중...")
    vocab_sizes, embedding_dims = get_categorical_info(
        data, train_dataset.categorical_features, K=K
    )

    print("범주형 변수:")
    for feat_name in vocab_sizes.keys():
        print(f"   - {feat_name:20s}: Vocab={vocab_sizes[feat_name]:3d}, Emb_Dim={embedding_dims[feat_name]:2d}")
    print()

    # 5. 모델 생성
    print("🏗️ 모델 V5 생성 중...")
    model = SoccerRNNWithAttention(
        num_numerical_features=train_dataset.numerical_tensor.shape[2],
        categorical_vocab_sizes=vocab_sizes,
        embedding_dims=embedding_dims,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
        use_lstm=USE_LSTM,
        bidirectional=BIDIRECTIONAL,
        num_heads=NUM_HEADS
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"✅ 모델 V5 생성 완료:")
    print(f"   - Total Parameters: {total_params:,}")
    print(f"   - Trainable Parameters: {trainable_params:,}")
    print(f"   - Attention Heads: {NUM_HEADS}")
    print(f"   - Bidirectional: {BIDIRECTIONAL}")
    print()

    # 6. Loss & Optimizer
    criterion = EuclideanDistanceLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-3)

    # Cosine Annealing with Warm Restarts
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-6
    )

    # 7. 학습 루프
    print("🚀 학습 시작 (V5 - Attention Model)...\n")
    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(NUM_EPOCHS):
        print(f"Epoch {epoch+1}/{NUM_EPOCHS}")
        print("-" * 60)

        # Train
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)

        # Validate
        val_loss = validate(model, val_loader, criterion, device)

        # Learning Rate Scheduler
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        print(f"Train Loss: {train_loss:.4f}m | Val Loss: {val_loss:.4f}m | LR: {current_lr:.6f}")

        # Early Stopping & Model Saving
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0

            # 모델 저장
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_loss': val_loss,
                'vocab_sizes': vocab_sizes,
                'embedding_dims': embedding_dims,
                'num_numerical_features': train_dataset.numerical_tensor.shape[2],
                'categorical_features': train_dataset.categorical_features,
                'numerical_features': train_dataset.numerical_features,
                'hyperparameters': {
                    'hidden_dim': HIDDEN_DIM,
                    'num_layers': NUM_LAYERS,
                    'dropout': DROPOUT,
                    'use_lstm': USE_LSTM,
                    'bidirectional': BIDIRECTIONAL,
                    'num_heads': NUM_HEADS,
                    'K': K
                }
            }, 'lstm_model_v5_attention_best.pth')

            print(f"💾 Best model saved! (Val Loss: {val_loss:.4f}m)")
        else:
            patience_counter += 1
            print(f"⏳ Patience: {patience_counter}/{EARLY_STOPPING_PATIENCE}")

            if patience_counter >= EARLY_STOPPING_PATIENCE:
                print(f"\n⚠️ Early stopping triggered!")
                break

        print()

    # 8. 최종 결과
    print("=" * 80)
    print("  학습 완료! (V5 - Attention Model)")
    print("=" * 80)
    print(f"\n✅ Best Validation Loss: {best_val_loss:.4f}m")
    print(f"✅ 모델 저장: lstm_model_v5_attention_best.pth")

    print("\n📊 성능 비교:")
    print(f"   - LightGBM V4 (5-Fold): 14.138m")
    print(f"   - LSTM/GRU V4 (Baseline): 15.649m")
    print(f"   - LSTM/GRU V5 (Attention): {best_val_loss:.4f}m")

    improvement = 15.649 - best_val_loss
    print(f"\n📈 개선폭: {improvement:.4f}m ({improvement/15.649*100:.1f}%)")

    if best_val_loss < 14.138:
        print("\n🎉🎉🎉 축하합니다! LightGBM을 초과했습니다!")
    elif best_val_loss < 15.0:
        print("\n✅ 좋은 성능! 추가 튜닝으로 LightGBM 초과 가능합니다.")
    else:
        print("\n📈 다음 단계: 5-Fold CV, TTA, Data Augmentation")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()

