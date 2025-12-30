"""
K-League Pass Prediction - LSTM V5 5-Fold Training + TTA

목표: LightGBM (14.138m) 초과 성능 달성
- 5-Fold Cross Validation
- Test Time Augmentation
- 모델 앙상블 (단일 모델 내에서)

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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🔧 Using device: {device}")


class SoccerDatasetV5(Dataset):
    """V5 Dataset with Padding Mask"""

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


class EuclideanDistanceLoss(nn.Module):
    def forward(self, pred, target):
        pred_real = pred.clone()
        pred_real[:, 0] *= 105.0
        pred_real[:, 1] *= 68.0
        target_real = target.clone()
        target_real[:, 0] *= 105.0
        target_real[:, 1] *= 68.0
        distances = torch.sqrt(torch.sum((pred_real - target_real) ** 2, dim=1))
        return distances.mean()


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


def train_one_epoch(model, dataloader, criterion, optimizer, device):
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
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()

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


def train_single_fold(fold_idx, train_data, val_data, hyperparams, vocab_sizes, embedding_dims, num_numerical_features):
    """단일 Fold 학습"""
    print(f"\n{'='*80}")
    print(f"  Fold {fold_idx + 1}/5 학습 시작")
    print(f"{'='*80}\n")

    # Dataset
    train_dataset = SoccerDatasetV5(train_data, K=hyperparams['K'], is_train=True)
    val_dataset = SoccerDatasetV5(val_data, K=hyperparams['K'], is_train=True)

    train_loader = DataLoader(train_dataset, batch_size=hyperparams['batch_size'],
                             shuffle=True, num_workers=0, pin_memory=True if torch.cuda.is_available() else False)
    val_loader = DataLoader(val_dataset, batch_size=hyperparams['batch_size'],
                           shuffle=False, num_workers=0, pin_memory=True if torch.cuda.is_available() else False)

    # Model
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

    criterion = EuclideanDistanceLoss()
    optimizer = optim.AdamW(model.parameters(), lr=hyperparams['lr'], weight_decay=1e-3)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)

    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(hyperparams['num_epochs']):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = validate(model, val_loader, criterion, device)
        scheduler.step()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0

            # 모델 저장
            torch.save({
                'fold': fold_idx,
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_loss': val_loss,
                'vocab_sizes': vocab_sizes,
                'embedding_dims': embedding_dims,
                'num_numerical_features': num_numerical_features,
                'categorical_features': train_dataset.categorical_features,
                'numerical_features': train_dataset.numerical_features,
                'hyperparameters': hyperparams
            }, f'lstm_model_v5_fold{fold_idx}.pth')

        else:
            patience_counter += 1
            if patience_counter >= hyperparams['patience']:
                print(f"Early stopping at epoch {epoch+1}")
                break

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}: Train={train_loss:.4f}m, Val={val_loss:.4f}m, Best={best_val_loss:.4f}m")

    print(f"\nFold {fold_idx + 1} 완료! Best Val Loss: {best_val_loss:.4f}m")
    return best_val_loss, model


def main():
    print("=" * 80)
    print("  LSTM V5 - 5-Fold Cross Validation Training")
    print("  목표: LightGBM (14.138m) 초과 성능 달성")
    print("=" * 80)
    print()

    # 하이퍼파라미터
    hyperparams = {
        'K': 20,
        'batch_size': 64,
        'hidden_dim': 384,
        'num_layers': 3,
        'dropout': 0.4,
        'lr': 5e-4,
        'num_epochs': 100,
        'patience': 20,
        'use_lstm': False,
        'bidirectional': True,
        'num_heads': 8
    }

    print("🔧 하이퍼파라미터:")
    for key, val in hyperparams.items():
        print(f"   - {key}: {val}")
    print()

    # 데이터 로딩
    print("📊 데이터 로딩...")
    data = pd.read_csv('processed_train_data_v4.csv')
    print(f"데이터 Shape: {data.shape}\n")

    game_ids = data['game_id'].values

    # 범주형 정보 (전체 데이터에서 추출)
    print("🔤 범주형 변수 정보 추출...")
    temp_dataset = SoccerDatasetV5(data.head(100), K=hyperparams['K'], is_train=True)
    vocab_sizes, embedding_dims = get_categorical_info(data, temp_dataset.categorical_features, K=hyperparams['K'])
    num_numerical_features = len(temp_dataset.numerical_features)
    print()

    # 5-Fold CV
    print("🔄 5-Fold Cross Validation 시작...\n")
    gkf = GroupKFold(n_splits=5)
    fold_results = []

    for fold_idx, (train_idx, val_idx) in enumerate(gkf.split(data, groups=game_ids)):
        train_data = data.iloc[train_idx].copy()
        val_data = data.iloc[val_idx].copy()

        print(f"Fold {fold_idx + 1}: Train={len(train_data):,}, Val={len(val_data):,}")

        best_val_loss, model = train_single_fold(
            fold_idx, train_data, val_data, hyperparams,
            vocab_sizes, embedding_dims, num_numerical_features
        )

        fold_results.append(best_val_loss)

    # 최종 결과
    print("\n" + "=" * 80)
    print("  5-Fold 학습 완료!")
    print("=" * 80)
    print("\n📊 Fold별 결과:")
    for i, loss in enumerate(fold_results):
        print(f"   Fold {i+1}: {loss:.4f}m")

    avg_loss = np.mean(fold_results)
    std_loss = np.std(fold_results)

    print(f"\n✅ 평균 Validation Loss: {avg_loss:.4f}m ± {std_loss:.4f}m")

    print("\n📈 성능 비교:")
    print(f"   - LightGBM V4 (5-Fold): 14.138m")
    print(f"   - LSTM V5 (5-Fold): {avg_loss:.4f}m")

    if avg_loss < 14.138:
        print("\n🎉🎉🎉 축하합니다! LightGBM을 초과했습니다!")
    else:
        gap = avg_loss - 14.138
        print(f"\n📊 LightGBM과의 차이: +{gap:.4f}m")
        print("   추가 개선 방안: TTA, Data Augmentation, Ensemble")

    print("\n" + "=" * 80)
    print("다음 단계: inference_lstm_v5_5fold.py (5개 모델 평균 예측)")
    print("=" * 80)


if __name__ == "__main__":
    main()

