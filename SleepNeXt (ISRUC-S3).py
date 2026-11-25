# ════════════════════════════════════════════════════════════════════════
# SleepNeXt — ISRUC-Sleep Subgroup III Training
# 10-fold Subject-Independent Cross-Validation (LOSO style)
# Phase 1: 8ch (6 EEG + 2 EOG) | Phase 2: 11ch (all channels)
# ════════════════════════════════════════════════════════════════════════
import os, gc, time, json, pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import KFold
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, classification_report,
                             cohen_kappa_score, confusion_matrix)
from timm.layers import DropPath
from tqdm import tqdm

# ═══════════════════════════════════════════════════════════════════════
# GLOBAL SETTINGS
# ═══════════════════════════════════════════════════════════════════════
SAVE        = '/kaggle/working'
DEVICE      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
STAGE_NAMES = ['Wake', 'N1', 'N2', 'N3', 'REM']
os.environ['PYTORCH_ALLOC_CONF'] = 'expandable_segments:True'

# ── Set phase here ──────────────────────────────────────────────────────
# Phase 1: 8 channels (6 EEG + 2 EOG)
# Phase 2: 11 channels (all)
PHASE_TAG = 'phase2_11ch'   # ← changed to 'phase2_11ch' for phase 2
N_CHANNELS = 11             # ← changed to 11 for phase 2
# ────────────────────────────────────────────────────────────────────────

N_FOLDS = 10   # 10 subjects → 10-fold, each fold leaves 1 subject out


# ═══════════════════════════════════════════════════════════════════════
# EEG AUGMENTATION
# ═══════════════════════════════════════════════════════════════════════
def eeg_augment(x, n_time_masks=2, time_mask_width=150,
                noise_prob=0.5, noise_std=0.05,
                ch_drop_prob=0.3, amp_scale_prob=0.5,
                amp_scale_range=(0.8, 1.2)):
    B, C, T = x.shape
    for _ in range(n_time_masks):
        w = np.random.randint(1, time_mask_width + 1)
        if T - w > 0:
            t0 = np.random.randint(0, T - w)
            x[:, :, t0:t0 + w] = 0.0
    if np.random.rand() < noise_prob:
        x = x + torch.randn_like(x) * noise_std
    if np.random.rand() < ch_drop_prob:
        x[:, np.random.randint(0, C), :] = 0.0
    if np.random.rand() < amp_scale_prob:
        lo, hi = amp_scale_range
        x = x * (lo + np.random.rand() * (hi - lo))
    return x


# ═══════════════════════════════════════════════════════════════════════
# 1D COORDINATE ATTENTION
# ═══════════════════════════════════════════════════════════════════════
class CoordinateAttention1D(nn.Module):
    def __init__(self, in_channels, reduction=8):
        super().__init__()
        hidden = max(8, in_channels // reduction)
        self.shared_conv = nn.Conv1d(in_channels, hidden, 1, bias=False)
        self.bn          = nn.BatchNorm1d(hidden)
        self.act         = nn.ReLU(inplace=True)
        self.conv_time   = nn.Conv1d(hidden, in_channels, 1, bias=False)
        self.conv_chan   = nn.Conv1d(hidden, in_channels, 1, bias=False)

    def forward(self, x):
        B, C, T = x.shape
        x_chan = F.adaptive_avg_pool1d(x, 1)
        x_time = x.mean(dim=1, keepdim=True).permute(0, 2, 1)
        f_chan = self.act(self.bn(self.shared_conv(x_chan)))
        x_time_expanded = x_time.expand(-1, -1, C).permute(0, 2, 1)
        f_time = self.act(self.bn(self.shared_conv(x_time_expanded)))
        att_chan = torch.sigmoid(self.conv_chan(f_chan))
        att_time = torch.sigmoid(self.conv_time(f_time))
        return x * att_chan * att_time


# ═══════════════════════════════════════════════════════════════════════
# MODEL COMPONENTS
# ═══════════════════════════════════════════════════════════════════════
class LayerNorm1D(nn.Module):
    def __init__(self, num_channels, eps=1e-6,
                 data_format="channels_first"):
        super().__init__()
        self.weight       = nn.Parameter(torch.ones(num_channels))
        self.bias         = nn.Parameter(torch.zeros(num_channels))
        self.eps          = eps
        self.data_format  = data_format
        self.num_channels = num_channels

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, (self.num_channels,),
                                self.weight, self.bias, self.eps)
        u = x.mean(dim=1, keepdim=True)
        s = (x - u).pow(2).mean(dim=1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        return self.weight[None, :, None] * x + self.bias[None, :, None]


class GRN1D(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, dim))
        self.beta  = nn.Parameter(torch.zeros(1, 1, dim))

    def forward(self, x):
        gx = torch.norm(x, p=2, dim=1, keepdim=True)
        nx = gx / (gx.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (x * nx) + self.beta + x


class ConvNeXt1DBlock(nn.Module):
    def __init__(self, dim, drop_path=0.0):
        super().__init__()
        self.dwconv    = nn.Conv1d(dim, dim, 7, padding=3,
                                   groups=dim, bias=False)
        self.norm      = LayerNorm1D(dim, data_format="channels_last")
        self.pwconv1   = nn.Linear(dim, 4 * dim)
        self.act       = nn.GELU()
        self.grn       = GRN1D(4 * dim)
        self.pwconv2   = nn.Linear(4 * dim, dim)
        self.drop_path = (DropPath(drop_path) if drop_path > 0.
                          else nn.Identity())

    def forward(self, x):
        residual = x
        x = self.dwconv(x).permute(0, 2, 1)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x).permute(0, 2, 1)
        return residual + self.drop_path(x)


class DualFrequencyStem1D(nn.Module):
    def __init__(self, in_chans=8, out_chans=64):
        super().__init__()
        half = out_chans // 2
        self.branch_fine   = nn.Conv1d(in_chans, half, 5,
                                       padding=2,  bias=False)
        self.branch_coarse = nn.Conv1d(in_chans, half, 51,
                                       padding=25, bias=False)
        self.norm = LayerNorm1D(out_chans, data_format="channels_first")

    def forward(self, x):
        fine   = self.branch_fine(x)
        coarse = self.branch_coarse(x)
        return self.norm(torch.cat([fine, coarse], dim=1))


class SleepEncoder1D(nn.Module):
    def __init__(self, in_chans=8, stem_chans=64,
                 depths=(2, 2, 6), dims=(64, 128, 256),
                 drop_path_rate=0.15, use_ca=True):
        super().__init__()
        self.use_ca = use_ca
        self.stem   = DualFrequencyStem1D(in_chans, stem_chans)
        total_blocks = sum(depths)
        dp_rates    = [x.item() for x in
                       torch.linspace(0, drop_path_rate, total_blocks)]

        self.stages      = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        cur = 0
        for i in range(len(depths)):
            self.stages.append(nn.Sequential(*[
                ConvNeXt1DBlock(dims[i], dp_rates[cur + j])
                for j in range(depths[i])
            ]))
            cur += depths[i]
            if i < len(depths) - 1:
                self.downsamples.append(nn.Sequential(
                    LayerNorm1D(dims[i], data_format="channels_first"),
                    nn.Conv1d(dims[i], dims[i + 1], 4, stride=4)
                ))

        self.ca      = CoordinateAttention1D(dims[-1]) if use_ca else nn.Identity()
        self.out_dim = dims[-1]

    def forward(self, x):
        x = self.stem(x)
        for i, stage in enumerate(self.stages):
            x = stage(x)
            if i < len(self.downsamples):
                x = self.downsamples[i](x)
        x = self.ca(x)
        return x.mean(dim=-1)


class SleepNeXt(nn.Module):
    """
    1D-ConvNeXtV2 + Coordinate Attention + BiLSTM
    in_chans must match the number of channels in the loaded data
    (8 for Phase 1, 11 for Phase 2)
    """
    def __init__(self, in_chans=8, num_classes=5,
                 stem_chans=64, depths=(2, 2, 6), dims=(64, 128, 256),
                 drop_path_rate=0.15, dropout=0.4,
                 lstm_hidden=128, lstm_layers=2, use_ca=True):
        super().__init__()
        self.encoder = SleepEncoder1D(
            in_chans=in_chans, stem_chans=stem_chans,
            depths=depths, dims=dims,
            drop_path_rate=drop_path_rate, use_ca=use_ca)
        feat_dim = self.encoder.out_dim
        self.bilstm = nn.LSTM(
            input_size    = feat_dim,
            hidden_size   = lstm_hidden,
            num_layers    = lstm_layers,
            batch_first   = True,
            bidirectional = True,
            dropout       = dropout if lstm_layers > 1 else 0.0)
        self.norm       = nn.LayerNorm(lstm_hidden * 2)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden * 2, num_classes))
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv1d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # x: (B, L, C, T)
        B, L, C, T = x.shape
        x          = x.reshape(B * L, C, T)
        feats      = self.encoder(x)
        feats      = feats.reshape(B, L, -1)
        seq_out, _ = self.bilstm(feats)
        seq_out    = self.norm(seq_out)
        return self.classifier(seq_out)


# ═══════════════════════════════════════════════════════════════════════
# LOSS
# ═══════════════════════════════════════════════════════════════════════
class FocalLossWithSmoothing(nn.Module):
    def __init__(self, num_classes=5, gamma=2.0,
                 smoothing=0.1, weight=None):
        super().__init__()
        self.num_classes = num_classes
        self.gamma       = gamma
        self.smoothing   = smoothing
        self.weight      = weight

    def forward(self, logits, targets):
        log_probs = F.log_softmax(logits, dim=1)
        probs     = torch.exp(log_probs)
        with torch.no_grad():
            smooth_t = torch.full_like(
                logits, self.smoothing / (self.num_classes - 1))
            smooth_t.scatter_(1, targets.unsqueeze(1),
                              1.0 - self.smoothing)
        p_t  = (probs * smooth_t).sum(dim=1, keepdim=True)
        loss = -(((1 - p_t) ** self.gamma) *
                 (smooth_t * log_probs)).sum(dim=1)
        if self.weight is not None:
            loss = loss * self.weight[targets]
        return loss.mean()


# ═══════════════════════════════════════════════════════════════════════
# DATASET
# ═══════════════════════════════════════════════════════════════════════
class SleepSequenceDataset(Dataset):
    def __init__(self, X, y, subject_ids, recording_indices,
                 seq_len=20, stride=10):
        self.X        = X
        self.y        = y
        self.seq_len  = seq_len
        self.sequences = self._build_sequences(
            subject_ids, recording_indices, seq_len, stride)

    def _build_sequences(self, subject_ids, indices, seq_len, stride):
        sequences = []
        # For ISRUC-S3: 1 recording per subject, so subject_id == recording_id
        # Sequences must not cross subject boundaries
        unique_subj = np.unique(subject_ids[indices])
        for s in unique_subj:
            subj_idx = np.where(
                (subject_ids == s) &
                np.isin(np.arange(len(subject_ids)), indices)
            )[0]
            n = len(subj_idx)
            if n < seq_len:
                continue
            start = 0
            while start + seq_len <= n:
                sequences.append(subj_idx[start:start + seq_len])
                start += stride
            # Catch the tail if not already covered
            if start > stride and (start - stride + seq_len) < n:
                sequences.append(subj_idx[n - seq_len:n])
        return sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq_indices = self.sequences[idx]
        epochs = np.stack([
            self.X[i].astype(np.float32) for i in seq_indices])
        labels = np.array([int(self.y[i]) for i in seq_indices],
                          dtype=np.int64)
        return (torch.from_numpy(epochs),
                torch.from_numpy(labels))


# ═══════════════════════════════════════════════════════════════════════
# SCHEDULER & EVAL
# ═══════════════════════════════════════════════════════════════════════
def get_scheduler(optimizer, warmup_epochs, total_epochs):
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = ((epoch - warmup_epochs) /
                    max(1, total_epochs - warmup_epochs))
        return 0.5 * (1.0 + np.cos(np.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def evaluate_full(model, loader, criterion, n_channels):
    model.eval()
    total_loss, total_n   = 0.0, 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            B, L   = yb.shape
            logits = model(xb)
            loss   = criterion(
                logits.reshape(B * L, 5), yb.reshape(B * L))
            total_loss += loss.item() * B * L
            total_n    += B * L
            all_preds.extend(
                logits.reshape(B * L, 5).argmax(1).cpu().numpy())
            all_labels.extend(yb.reshape(B * L).cpu().numpy())
    return (total_loss / total_n,
            np.array(all_preds),
            np.array(all_labels))


# ═══════════════════════════════════════════════════════════════════════
# SAVE DETAILED PER-FOLD RESULTS
# ═══════════════════════════════════════════════════════════════════════
def save_fold_results(model, test_loader, criterion,
                      fold_idx, phase_tag, tag, results_path,
                      best_acc, best_mf1, n_channels):
    _, preds, labels = evaluate_full(
        model, test_loader, criterion, n_channels)

    acc           = accuracy_score(labels, preds)
    mf1           = f1_score(labels, preds, average='macro',    zero_division=0)
    weighted_f1   = f1_score(labels, preds, average='weighted', zero_division=0)
    kappa         = cohen_kappa_score(labels, preds)
    per_class_f1  = f1_score(labels, preds, average=None, zero_division=0)
    per_class_p   = precision_score(labels, preds, average=None, zero_division=0)
    per_class_r   = recall_score(labels, preds, average=None, zero_division=0)
    cm            = confusion_matrix(labels, preds, labels=[0, 1, 2, 3, 4])

    results = {
        'fold'               : fold_idx + 1,
        'phase'              : phase_tag,
        'model'              : tag,
        'accuracy'           : float(acc),
        'macro_f1'           : float(mf1),
        'weighted_f1'        : float(weighted_f1),
        'kappa'              : float(kappa),
        'per_class_f1'       : per_class_f1.tolist(),
        'per_class_precision': per_class_p.tolist(),
        'per_class_recall'   : per_class_r.tolist(),
        'confusion_matrix'   : cm.tolist(),
        'predictions'        : preds.tolist(),
        'labels'             : labels.tolist(),
        'class_names'        : STAGE_NAMES,
        'n_test_epochs'      : len(labels),
        'best_acc'           : float(best_acc),
        'best_mf1'           : float(best_mf1),
    }

    with open(results_path, 'wb') as f:
        pickle.dump(results, f)

    print(classification_report(labels, preds,
          target_names=STAGE_NAMES, digits=4, zero_division=0))
    print(f"Cohen's Kappa : {kappa:.4f}")
    print(f"Accuracy      : {acc * 100:.2f}%")
    print(f"Macro F1      : {mf1 * 100:.2f}%")
    print(f"\n[RESULTS SAVED] → {results_path}")
    print(f"Download this file for cross-fold aggregation.")


# ═══════════════════════════════════════════════════════════════════════
# TRAINING — ONE FOLD
# ═══════════════════════════════════════════════════════════════════════
def train_one_fold(fold_idx, X, y, subject_ids,
                   phase_tag=PHASE_TAG,
                   n_channels=N_CHANNELS,
                   total_epochs=30, chunk_epochs=15,
                   seq_len=20, train_stride=10,
                   batch_size=8, accum_steps=4,
                   lr=1e-3, warmup_epochs=5,
                   weight_decay=0.05, drop_path_rate=0.15,
                   dropout=0.4, lstm_hidden=128, lstm_layers=2,
                   model_name='SleepNeXt'):

    tag          = f'{model_name}_isruc_s3_{phase_tag}'
    resume_path  = f'{SAVE}/fold_{fold_idx + 1}_{tag}_resume.pth'
    best_path    = f'{SAVE}/fold_{fold_idx + 1}_{tag}_best.pth'
    results_path = f'{SAVE}/fold_{fold_idx + 1}_{tag}_results.pkl'

    print(f"\n{'═' * 70}")
    print(f"  {tag} — FOLD {fold_idx + 1}/{N_FOLDS}")
    print(f"{'═' * 70}")
    torch.cuda.empty_cache(); gc.collect()

    # ── 10-fold subject-independent split ──────────────────────────────
    # ISRUC-S3: 10 subjects (ids 0–9), each fold holds out 1 subject
    unique_subj = np.unique(subject_ids)          # array([0,1,...,9])
    kf          = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    splits      = list(kf.split(unique_subj))
    train_subj  = unique_subj[splits[fold_idx][0]]
    test_subj   = unique_subj[splits[fold_idx][1]]

    train_idx = np.where(np.isin(subject_ids, train_subj))[0]
    test_idx  = np.where(np.isin(subject_ids, test_subj))[0]

    print(f"Train subjects: {train_subj.tolist()} ({len(train_subj)} subjects)")
    print(f"Test  subjects: {test_subj.tolist()}  ({len(test_subj)} subject)")
    print(f"Train epochs  : {len(train_idx):,} | Test epochs: {len(test_idx):,}")

    # ── Sanity check ───────────────────────────────────────────────────
    overlap = set(subject_ids[train_idx]) & set(subject_ids[test_idx])
    assert len(overlap) == 0, f"LEAKAGE! Subjects in both: {overlap}"
    print(f"✅ No subject leakage (verified)")

    # ── Class weights from training set ────────────────────────────────
    counts  = np.bincount(y[train_idx], minlength=5)
    weights = len(train_idx) / (5.0 * counts.clip(min=1))
    print("Class weights: " +
          " | ".join(f"{n}:{w:.3f}" for n, w in zip(STAGE_NAMES, weights)))

    # ── Datasets ────────────────────────────────────────────────────────
    train_ds = SleepSequenceDataset(X, y, subject_ids, train_idx,
                                    seq_len=seq_len, stride=train_stride)
    test_ds  = SleepSequenceDataset(X, y, subject_ids, test_idx,
                                    seq_len=seq_len, stride=seq_len)
    print(f"Train sequences: {len(train_ds):,} | "
          f"Test sequences : {len(test_ds):,}")

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True,  num_workers=2,
                              pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size,
                              shuffle=False, num_workers=2,
                              pin_memory=True)

    # ── Model ───────────────────────────────────────────────────────────
    model = SleepNeXt(
        in_chans       = n_channels,   # 8 (phase1) or 11 (phase2)
        num_classes    = 5,
        stem_chans     = 64,
        depths         = (2, 2, 6),
        dims           = (64, 128, 256),
        drop_path_rate = drop_path_rate,
        dropout        = dropout,
        lstm_hidden    = lstm_hidden,
        lstm_layers    = lstm_layers,
        use_ca         = True
    ).to(DEVICE)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    criterion = FocalLossWithSmoothing(
        num_classes=5, gamma=2.0, smoothing=0.1,
        weight=torch.FloatTensor(weights).to(DEVICE))
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = get_scheduler(optimizer, warmup_epochs, total_epochs)

    # ── Resume if checkpoint exists ─────────────────────────────────────
    start_epoch, best_mf1, best_acc = 0, 0.0, 0.0
    if os.path.exists(resume_path):
        ck = torch.load(resume_path, map_location=DEVICE,
                        weights_only=False)
        model.load_state_dict(ck['model_state_dict'])
        optimizer.load_state_dict(ck['optimizer_state_dict'])
        scheduler.load_state_dict(ck['scheduler_state_dict'])
        start_epoch = ck['epoch']
        best_mf1    = ck['best_mf1']
        best_acc    = ck['best_acc']
        print(f"[RESUME] Epoch {start_epoch} | "
              f"Best MF1: {best_mf1 * 100:.2f}%")

    if start_epoch >= total_epochs:
        print("✅ Fold already complete. Generating results file...")
        model.load_state_dict(
            torch.load(best_path, map_location=DEVICE, weights_only=False))
        save_fold_results(model, test_loader, criterion,
                          fold_idx, phase_tag, tag, results_path,
                          best_acc, best_mf1, n_channels)
        return best_acc, best_mf1

    target_epoch = min(start_epoch + chunk_epochs, total_epochs)
    print(f"Running epochs {start_epoch + 1} → {target_epoch} "
          f"(budget: {total_epochs})\n")

    n_train_batches = len(train_loader)

    for epoch in range(start_epoch, target_epoch):
        t0 = time.time()

        # ── TRAIN ────────────────────────────────────────────────────────
        model.train()
        tr_loss, tr_n       = 0.0, 0
        tr_preds, tr_labels = [], []
        train_bar = tqdm(train_loader,
                         desc=f"Ep {epoch + 1:02d}/{total_epochs} [Train]",
                         total=n_train_batches, ncols=80,
                         leave=False, colour='green')
        optimizer.zero_grad()

        for step, (xb, yb) in enumerate(train_bar):
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            B, L   = yb.shape

            # Augment: flatten to (B*L, C, T), augment, reshape back
            xb = xb.reshape(B * L, n_channels, 3000)
            xb = eeg_augment(xb)
            xb = xb.reshape(B, L, n_channels, 3000)

            logits = model(xb)
            loss   = criterion(
                logits.reshape(B * L, 5), yb.reshape(B * L))
            loss = loss / accum_steps
            loss.backward()

            if (step + 1) % accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()

            tr_loss += loss.item() * accum_steps * B * L
            tr_n    += B * L
            tr_preds.extend(
                logits.reshape(B * L, 5).argmax(1).detach().cpu().numpy())
            tr_labels.extend(yb.reshape(B * L).cpu().numpy())
            train_bar.set_postfix(
                loss=f"{tr_loss / tr_n:.4f}",
                acc =f"{accuracy_score(tr_labels, tr_preds) * 100:.1f}%")

        # Flush remaining gradients
        if n_train_batches % accum_steps != 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()

        scheduler.step()
        tr_loss /= tr_n
        tr_acc   = accuracy_score(tr_labels, tr_preds) * 100
        tr_mf1   = f1_score(tr_labels, tr_preds,
                            average='macro', zero_division=0) * 100

        # ── EVAL ──────────────────────────────────────────────────────────
        va_loss, all_preds, all_labels_val = evaluate_full(
            model, test_loader, criterion, n_channels)
        va_acc = accuracy_score(all_labels_val, all_preds) * 100
        va_mf1 = f1_score(all_labels_val, all_preds,
                          average='macro', zero_division=0) * 100
        kappa  = cohen_kappa_score(all_labels_val, all_preds)
        gap    = tr_acc - va_acc
        lr_now = optimizer.param_groups[0]['lr']
        ep_t   = time.time() - t0

        flag = ''
        if va_mf1 > best_mf1 * 100:
            best_mf1 = va_mf1 / 100
            best_acc = va_acc / 100
            torch.save(model.state_dict(), best_path)
            flag = ' ★'

        print(f"Ep {epoch + 1:02d}/{total_epochs} lr={lr_now:.2e} | "
              f"Tr: loss={tr_loss:.4f} acc={tr_acc:.1f}% mf1={tr_mf1:.1f}% | "
              f"Va: loss={va_loss:.4f} acc={va_acc:.1f}% mf1={va_mf1:.1f}% "
              f"kap={kappa:.3f} gap={gap:.1f}% [{ep_t:.0f}s]{flag}")

        # Save resume checkpoint every epoch
        torch.save({
            'epoch'               : epoch + 1,
            'model_state_dict'    : model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_mf1'            : best_mf1,
            'best_acc'            : best_acc,
        }, resume_path)

    # ── Fold complete → save detailed results ────────────────────────────
    if target_epoch == total_epochs:
        print(f"\n[SAVED] → {resume_path}")
        print("\n" + "═" * 70)
        print(f"FOLD {fold_idx + 1} COMPLETE — {tag}")
        print("═" * 70)
        model.load_state_dict(
            torch.load(best_path, map_location=DEVICE, weights_only=False))
        save_fold_results(model, test_loader, criterion,
                          fold_idx, phase_tag, tag, results_path,
                          best_acc, best_mf1, n_channels)

    del model, train_loader, test_loader, train_ds, test_ds
    torch.cuda.empty_cache(); gc.collect()
    return best_acc, best_mf1


# ═══════════════════════════════════════════════════════════════════════
# CROSS-FOLD AGGREGATION UTILITY
# ═══════════════════════════════════════════════════════════════════════
def aggregate_results(phase_tag=PHASE_TAG, model_name='SleepNeXt',
                      n_folds=N_FOLDS):
    """
    Load all per-fold result pkl files and print aggregated metrics.
    Run this after all folds are complete.
    """
    tag = f'{model_name}_isruc_s3_{phase_tag}'
    all_acc, all_mf1, all_kappa = [], [], []
    all_preds_total, all_labels_total = [], []
    per_class_f1_all = []

    for fold_idx in range(n_folds):
        results_path = f'{SAVE}/fold_{fold_idx + 1}_{tag}_results.pkl'
        if not os.path.exists(results_path):
            print(f"  [MISSING] fold {fold_idx + 1}")
            continue
        with open(results_path, 'rb') as f:
            r = pickle.load(f)
        all_acc.append(r['accuracy'])
        all_mf1.append(r['macro_f1'])
        all_kappa.append(r['kappa'])
        per_class_f1_all.append(r['per_class_f1'])
        all_preds_total.extend(r['predictions'])
        all_labels_total.extend(r['labels'])
        print(f"  Fold {fold_idx + 1:2d}: "
              f"Acc={r['accuracy']*100:.2f}%  "
              f"MF1={r['macro_f1']*100:.2f}%  "
              f"Kappa={r['kappa']:.4f}")

    if not all_acc:
        print("No completed folds found.")
        return

    all_preds_total  = np.array(all_preds_total)
    all_labels_total = np.array(all_labels_total)
    per_class_f1_all = np.array(per_class_f1_all)  # (n_folds, 5)

    print(f"\n{'═'*60}")
    print(f"ISRUC-S3 {phase_tag} — {len(all_acc)}/{n_folds} folds complete")
    print(f"{'═'*60}")
    print(f"Accuracy  : {np.mean(all_acc)*100:.2f}% ± {np.std(all_acc)*100:.2f}%")
    print(f"Macro F1  : {np.mean(all_mf1)*100:.2f}% ± {np.std(all_mf1)*100:.2f}%")
    print(f"Kappa     : {np.mean(all_kappa):.4f} ± {np.std(all_kappa):.4f}")
    print(f"\nPer-class F1 (mean ± std across folds):")
    for c, nm in enumerate(STAGE_NAMES):
        vals = per_class_f1_all[:, c]
        print(f"  {nm:5s}: {vals.mean()*100:.2f}% ± {vals.std()*100:.2f}%")
    print(f"\nOverall (pooled all test epochs):")
    print(classification_report(all_labels_total, all_preds_total,
          target_names=STAGE_NAMES, digits=4, zero_division=0))


# ═══════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════
print(f"✅ SleepNeXt-ISRUC-S3 pipeline loaded")
print(f"   Device     : {DEVICE}")
print(f"   Phase      : {PHASE_TAG} ({N_CHANNELS} channels)")
print(f"   N_FOLDS    : {N_FOLDS} (subject-independent LOSO)")

suffix = f'isruc_{PHASE_TAG}'
X           = np.load(f'{SAVE}/X_raw_{suffix}.npy',       mmap_mode='r')
y           = np.load(f'{SAVE}/y_{suffix}.npy')
subject_ids = np.load(f'{SAVE}/subject_ids_{suffix}.npy')

print(f"   X shape    : {X.shape}")
print(f"   y shape    : {y.shape}")
print(f"   Subjects   : {np.unique(subject_ids).tolist()}")

# ── Run one fold (change fold_idx 0–9 to run all folds) ──────────────
train_one_fold(
    fold_idx       = 0,          # ← change to 0,1,...,9 across sessions
    X              = X,
    y              = y,
    subject_ids    = subject_ids,
    phase_tag      = PHASE_TAG,
    n_channels     = N_CHANNELS,
    total_epochs   = 35,
    chunk_epochs   = 35,         # epochs per Kaggle session (resume-safe)
    seq_len        = 20,
    train_stride   = 10,
    batch_size     = 8,
    accum_steps    = 4,
    lr             = 1e-3,
    warmup_epochs  = 5,
    weight_decay   = 0.05,
    dropout        = 0.4,
    drop_path_rate = 0.15,
    lstm_hidden    = 128,
    lstm_layers    = 2,
    model_name     = 'SleepNeXt'
)