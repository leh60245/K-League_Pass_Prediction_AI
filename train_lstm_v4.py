"""
K-League Pass Prediction - PyTorch LSTM/GRU 학습 파이프라인

V4 Wide Format 데이터를 3D 시퀀스 텐서로 변환하여 딥러닝 모델 학습
✅ 데이터 정규화 (좌표 스케일링)
✅ Input Projection Layer
✅ NaN 처리 (패딩 → 0 변환)
✅ Embedding for Categorical Features

작성일: 2025-12-18
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


class SoccerDatasetV4(Dataset):
    """
    V4 Wide Format 데이터를 3D 시퀀스 텐서로 변환하는 Dataset

    Features:
    - Wide format → (Batch, SeqLen=20, Features) 3D tensor
    - NaN → 0.0 변환
    - 좌표 정규화 (start_x/end_x → /105, start_y/end_y → /68)
    - Categorical/Numerical 자동 분류
    """

    def __init__(self, data, K=20, is_train=True):
        """
        Args:
            data: pd.DataFrame (Wide format V4 data)
            K: 시퀀스 길이 (기본 20)
            is_train: Train/Val 구분
        """
        self.data = data.reset_index(drop=True)
        self.K = K
        self.is_train = is_train

        # 타겟 추출
        if 'target_x' in data.columns and 'target_y' in data.columns:
            self.targets = data[['target_x', 'target_y']].values.astype(np.float32)
            # 타겟도 정규화
            self.targets[:, 0] /= 105.0  # target_x
            self.targets[:, 1] /= 68.0   # target_y
        else:
            self.targets = None

        # 메타 정보 제외 (game_episode, game_id, target_x, target_y, final_team_id)
        exclude_cols = ['game_episode', 'game_id', 'target_x', 'target_y', 'final_team_id']
        feature_data = data.drop(columns=[c for c in exclude_cols if c in data.columns])

        # 컬럼 분류 (자동)
        self.numerical_features, self.categorical_features = self._classify_columns(feature_data.columns)

        # 3D 텐서로 변환 + 정규화
        self.numerical_tensor = self._prepare_numerical_features(feature_data)
        self.categorical_tensor = self._prepare_categorical_features(feature_data)

        print(f"✅ Dataset 준비 완료:")
        print(f"   - 샘플 수: {len(self.data)}")
        print(f"   - 수치형 피처: {len(self.numerical_features)} → Shape: {self.numerical_tensor.shape}")
        print(f"   - 범주형 피처: {len(self.categorical_features)} → Shape: {self.categorical_tensor.shape}")

    def _classify_columns(self, columns):
        """컬럼명에서 _{index} 패턴 추출하여 수치형/범주형 분류"""
        pattern = re.compile(r'^(.+)_(\d+)$')

        # 고유 feature 이름 추출
        feature_names = set()
        for col in columns:
            match = pattern.match(col)
            if match:
                feature_names.add(match.group(1))

        # 범주형 키워드 (Embedding 사용 대상)
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

        # 좌표 관련 컬럼 식별 (정규화 대상)
        x_coord_keywords = ['start_x', 'end_x', 'dx']  # X 좌표 관련 (105로 나눔)
        y_coord_keywords = ['start_y', 'end_y', 'dy']  # Y 좌표 관련 (68로 나눔)

        for feat_name in self.numerical_features:
            # 각 시점별 컬럼 (0~19)
            cols = [f"{feat_name}_{i}" for i in range(self.K)]
            cols = [c for c in cols if c in data.columns]

            if not cols:
                continue

            # 데이터 추출
            feat_data = data[cols].values.astype(np.float32)

            # 시퀀스 길이가 K보다 짧은 경우 오른쪽에 0 패딩
            if feat_data.shape[1] < self.K:
                padding = np.zeros((feat_data.shape[0], self.K - feat_data.shape[1]), dtype=np.float32)
                feat_data = np.concatenate([feat_data, padding], axis=1)

            # 정규화 적용
            if any(kw in feat_name for kw in x_coord_keywords):
                # X 좌표 관련: 105로 나눔
                feat_data = feat_data / 105.0
            elif any(kw in feat_name for kw in y_coord_keywords):
                # Y 좌표 관련: 68로 나눔
                feat_data = feat_data / 68.0
            # 나머지 수치형은 그대로 (이미 적절한 범위이거나 비율)

            # NaN → 0.0
            feat_data = np.nan_to_num(feat_data, nan=0.0)

            tensors.append(feat_data)

        # (N, K, num_features)
        result = np.stack(tensors, axis=-1) if tensors else np.zeros((len(data), self.K, 0), dtype=np.float32)
        return torch.from_numpy(result)

    def _prepare_categorical_features(self, data):
        """범주형 피처를 3D 텐서로 변환 (정수 인코딩 유지)"""
        tensors = []

        for feat_name in self.categorical_features:
            cols = [f"{feat_name}_{i}" for i in range(self.K)]
            cols = [c for c in cols if c in data.columns]

            if not cols:
                continue

            feat_data = data[cols].values.astype(np.float32)

            # 시퀀스 길이가 K보다 짧은 경우 오른쪽에 0 패딩
            if feat_data.shape[1] < self.K:
                padding = np.zeros((feat_data.shape[0], self.K - feat_data.shape[1]), dtype=np.float32)
                feat_data = np.concatenate([feat_data, padding], axis=1)

            # NaN → 0 (Unknown 범주)
            feat_data = np.nan_to_num(feat_data, nan=0.0)

            tensors.append(feat_data)

        # (N, K, cat_features)
        result = np.stack(tensors, axis=-1) if tensors else np.zeros((len(data), self.K, 0), dtype=np.float32)
        return torch.from_numpy(result).long()  # Long tensor for embedding

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        num_feat = self.numerical_tensor[idx]  # (K, num_features)
        cat_feat = self.categorical_tensor[idx]  # (K, cat_features)

        if self.targets is not None:
            target = torch.from_numpy(self.targets[idx])
            return num_feat, cat_feat, target
        else:
            return num_feat, cat_feat


class SoccerRNN(nn.Module):
    """
    Embedding + Input Projection + GRU/LSTM 기반 패스 예측 모델

    Architecture:
    1. Categorical Embedding
    2. Concatenate (Numerical + Embedded)
    3. Input Projection (Linear)
    4. GRU/LSTM
    5. Output Head (마지막 hidden state → target_x, target_y)
    """

    def __init__(self,
                 num_numerical_features,
                 categorical_vocab_sizes,
                 embedding_dims,
                 hidden_dim=256,
                 num_layers=2,
                 dropout=0.3,
                 use_lstm=False):
        """
        Args:
            num_numerical_features: 수치형 피처 개수
            categorical_vocab_sizes: 범주형 변수별 어휘 크기 (dict)
            embedding_dims: 범주형 변수별 임베딩 차원 (dict)
            hidden_dim: RNN hidden dimension
            num_layers: RNN 레이어 수
            dropout: Dropout 비율
            use_lstm: True면 LSTM, False면 GRU
        """
        super(SoccerRNN, self).__init__()

        self.num_numerical_features = num_numerical_features
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.use_lstm = use_lstm

        # Embedding layers
        self.embeddings = nn.ModuleDict()
        total_embedding_dim = 0

        for feat_name, vocab_size in categorical_vocab_sizes.items():
            emb_dim = embedding_dims[feat_name]
            self.embeddings[feat_name] = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
            total_embedding_dim += emb_dim

        # Input dimension
        input_dim = num_numerical_features + total_embedding_dim

        # Input Projection Layer (핵심 개선사항)
        self.input_projection = nn.Linear(input_dim, hidden_dim)

        # RNN Layer
        if use_lstm:
            self.rnn = nn.LSTM(
                hidden_dim,
                hidden_dim,
                num_layers=num_layers,
                dropout=dropout if num_layers > 1 else 0,
                batch_first=True
            )
        else:
            self.rnn = nn.GRU(
                hidden_dim,
                hidden_dim,
                num_layers=num_layers,
                dropout=dropout if num_layers > 1 else 0,
                batch_first=True
            )

        # Output Head
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 2)  # (target_x, target_y)
        )

    def forward(self, num_feat, cat_feat):
        """
        Args:
            num_feat: (batch, seq_len, num_features)
            cat_feat: (batch, seq_len, cat_features)

        Returns:
            (batch, 2) - (target_x, target_y)
        """
        batch_size, seq_len, _ = num_feat.shape

        # Embedding
        embedded = []
        for i, feat_name in enumerate(self.embeddings.keys()):
            emb = self.embeddings[feat_name](cat_feat[:, :, i])  # (batch, seq_len, emb_dim)
            embedded.append(emb)

        # Concatenate
        if embedded:
            embedded = torch.cat(embedded, dim=-1)  # (batch, seq_len, total_emb_dim)
            x = torch.cat([num_feat, embedded], dim=-1)  # (batch, seq_len, input_dim)
        else:
            x = num_feat

        # Input Projection
        x = self.input_projection(x)  # (batch, seq_len, hidden_dim)

        # RNN
        rnn_out, _ = self.rnn(x)  # (batch, seq_len, hidden_dim)

        # 마지막 시점의 hidden state
        last_hidden = rnn_out[:, -1, :]  # (batch, hidden_dim)

        # Output
        output = self.fc(last_hidden)  # (batch, 2)

        return output


class EuclideanDistanceLoss(nn.Module):
    """유클리드 거리 기반 손실 함수 (평가지표와 일치)"""

    def forward(self, pred, target):
        """
        Args:
            pred: (batch, 2) - (pred_x, pred_y) [0~1 normalized]
            target: (batch, 2) - (target_x, target_y) [0~1 normalized]

        Returns:
            평균 유클리드 거리 (실제 미터 단위)
        """
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

        # 최대값 (어휘 크기)
        max_val = data[cols].max().max()
        vocab_size = int(max_val) + 2  # 0: padding, 1~max_val: 실제 값

        # 임베딩 차원 (휴리스틱: min(vocab_size // 2, 50))
        emb_dim = min(max(vocab_size // 2, 4), 50)

        vocab_sizes[feat_name] = vocab_size
        embedding_dims[feat_name] = emb_dim

    return vocab_sizes, embedding_dims


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """1 에포크 학습"""
    model.train()
    total_loss = 0.0

    for num_feat, cat_feat, target in tqdm(dataloader, desc="Training", leave=False):
        num_feat = num_feat.to(device)
        cat_feat = cat_feat.to(device)
        target = target.to(device)

        optimizer.zero_grad()

        output = model(num_feat, cat_feat)
        loss = criterion(output, target)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def validate(model, dataloader, criterion, device):
    """검증"""
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for num_feat, cat_feat, target in tqdm(dataloader, desc="Validation", leave=False):
            num_feat = num_feat.to(device)
            cat_feat = cat_feat.to(device)
            target = target.to(device)

            output = model(num_feat, cat_feat)
            loss = criterion(output, target)

            total_loss += loss.item()

    return total_loss / len(dataloader)


def main():
    print("=" * 80)
    print("  PyTorch LSTM/GRU - V4 Wide Format 학습")
    print("  딥러닝 기반 시퀀스 모델링")
    print("=" * 80)
    print()

    # 하이퍼파라미터
    K = 20
    BATCH_SIZE = 128
    HIDDEN_DIM = 256
    NUM_LAYERS = 2
    DROPOUT = 0.3
    LEARNING_RATE = 1e-3
    NUM_EPOCHS = 50
    EARLY_STOPPING_PATIENCE = 10
    USE_LSTM = False  # False: GRU, True: LSTM

    print(f"🔧 하이퍼파라미터:")
    print(f"   - Sequence Length: {K}")
    print(f"   - Batch Size: {BATCH_SIZE}")
    print(f"   - Hidden Dim: {HIDDEN_DIM}")
    print(f"   - Num Layers: {NUM_LAYERS}")
    print(f"   - Dropout: {DROPOUT}")
    print(f"   - Learning Rate: {LEARNING_RATE}")
    print(f"   - Epochs: {NUM_EPOCHS}")
    print(f"   - RNN Type: {'LSTM' if USE_LSTM else 'GRU'}")
    print()

    # 1. 데이터 로딩
    print("📊 데이터 로딩...")
    data = pd.read_csv('processed_train_data_v4.csv')
    print(f"데이터 Shape: {data.shape}")
    print()

    # game_id 추출 (GroupKFold용)
    game_ids = data['game_id'].values

    # 2. 첫 번째 Fold만 사용 (프로토타이핑)
    print("🔧 First Fold 학습 (빠른 프로토타이핑)...")
    gkf = GroupKFold(n_splits=5)
    train_idx, val_idx = next(gkf.split(data, groups=game_ids))

    train_data = data.iloc[train_idx].copy()
    val_data = data.iloc[val_idx].copy()

    print(f"Train: {len(train_data):,} 샘플")
    print(f"Val: {len(val_data):,} 샘플")
    print()

    # 3. Dataset 생성
    print("📦 Dataset 생성 중...")
    train_dataset = SoccerDatasetV4(train_data, K=K, is_train=True)
    val_dataset = SoccerDatasetV4(val_data, K=K, is_train=True)
    print()

    # DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,  # Windows 호환
        pin_memory=True if torch.cuda.is_available() else False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=True if torch.cuda.is_available() else False
    )

    # 4. 범주형 정보 추출
    print("🔤 범주형 변수 정보 추출 중...")
    vocab_sizes, embedding_dims = get_categorical_info(
        data, train_dataset.categorical_features, K=K
    )

    print("범주형 변수:")
    for feat_name in vocab_sizes.keys():
        print(f"   - {feat_name:20s}: Vocab={vocab_sizes[feat_name]:3d}, Emb_Dim={embedding_dims[feat_name]:2d}")
    print()

    # 5. 모델 생성
    print("🏗️ 모델 생성 중...")
    model = SoccerRNN(
        num_numerical_features=train_dataset.numerical_tensor.shape[2],
        categorical_vocab_sizes=vocab_sizes,
        embedding_dims=embedding_dims,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
        use_lstm=USE_LSTM
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"✅ 모델 생성 완료:")
    print(f"   - Total Parameters: {total_params:,}")
    print(f"   - Trainable Parameters: {trainable_params:,}")
    print()

    # 6. Loss & Optimizer
    criterion = EuclideanDistanceLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )

    # 7. 학습 루프
    print("🚀 학습 시작...\n")
    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(NUM_EPOCHS):
        print(f"Epoch {epoch+1}/{NUM_EPOCHS}")
        print("-" * 60)

        # Train
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)

        # Validate
        val_loss = validate(model, val_loader, criterion, device)

        print(f"Train Loss: {train_loss:.4f}m | Val Loss: {val_loss:.4f}m")

        # Learning Rate Scheduler
        scheduler.step(val_loss)

        # Early Stopping & Model Saving
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0

            # 모델 저장
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
                    'dropout': DROPOUT,
                    'use_lstm': USE_LSTM,
                    'K': K
                }
            }, 'lstm_model_v4_best.pth')

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
    print("  학습 완료!")
    print("=" * 80)
    print(f"\n✅ Best Validation Loss: {best_val_loss:.4f}m")
    print(f"✅ 모델 저장: lstm_model_v4_best.pth")

    print("\n📊 성능 비교:")
    print(f"   - LightGBM V4 (5-Fold): ~1.5m")
    print(f"   - LSTM/GRU V4 (Fold 1): {best_val_loss:.4f}m")

    if best_val_loss < 1.5:
        print("\n🎉 매우 우수한 성능! 딥러닝이 트리 모델보다 효과적입니다!")
    elif best_val_loss < 2.0:
        print("\n✅ 좋은 성능! 추가 튜닝으로 개선 가능합니다.")
    else:
        print("\n📈 하이퍼파라미터 튜닝 필요 (Hidden Dim, Learning Rate 등)")

    print("\n" + "=" * 80)
    print("다음 단계:")
    print("   1. inference_lstm_v4.py 작성 (Test 추론)")
    print("   2. 5-Fold 전체 학습 (lstm_model_v4_5fold.py)")
    print("   3. LightGBM vs LSTM 앙상블")
    print("=" * 80)


if __name__ == "__main__":
    main()

