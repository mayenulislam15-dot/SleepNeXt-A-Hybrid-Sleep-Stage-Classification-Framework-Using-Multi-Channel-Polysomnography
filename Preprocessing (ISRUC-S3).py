# ════════════════════════════════════════════════════════════════════════
# ISRUC-Sleep Subgroup III — Multi-Channel Preprocessing
# Phase 1: 6 EEG + 2 EOG (8 channels)
# Phase 2: 6 EEG + 2 EOG + chin EMG + ECG + leg EMG (11 channels) — used by SleepNeXt
# ════════════════════════════════════════════════════════════════════════
import os
import numpy as np
import scipy.io as sio
from scipy.signal import decimate

MAT_BASE = '/kaggle/input/datasets/hoque56/isruc-sleep-cohort-iii/ISRUC-Sleep Cohort III/ISRUC-Sleep Cohort III'
LBL_BASE = '/kaggle/input/datasets/hoque56/isruc-sleep-cohort-iii/ISRUC-Sleep-III/ISRUC-Sleep-III'
SAVE     = '/kaggle/working'

# ── Phase 1: 6 EEG + 2 EOG (8 channels) ──
CHANNELS_PHASE1 = [
    'F3_A2',   # EEG frontal left
    'C3_A2',   # EEG central left
    'O1_A2',   # EEG occipital left
    'F4_A1',   # EEG frontal right
    'C4_A1',   # EEG central right
    'O2_A1',   # EEG occipital right
    'LOC_A2',  # EOG left
    'ROC_A1',  # EOG right
]
PHASE_TAG_PHASE1 = 'phase1_8ch'

# ── Phase 2: All 11 channels ──
CHANNELS_PHASE2 = [
    'F3_A2',   # EEG frontal left
    'C3_A2',   # EEG central left
    'O1_A2',   # EEG occipital left
    'F4_A1',   # EEG frontal right
    'C4_A1',   # EEG central right
    'O2_A1',   # EEG occipital right
    'LOC_A2',  # EOG left
    'ROC_A1',  # EOG right
    'X1',      # Chin EMG
    'X2',      # ECG
    'X3',      # Leg EMG
]
PHASE_TAG_PHASE2 = 'phase2_11ch'

#SleepNeXt uses 11 channels from ISRUC-S3 dataset.
#So, phase 2 is chosen for SleepNeXt over 8 channels' phase 1.

# ── Set which phase to run ──
CHANNELS  = CHANNELS_PHASE2     # ← changed to CHANNELS_PHASE2 for phase 2
PHASE_TAG = PHASE_TAG_PHASE2    # ← changed to PHASE_TAG_PHASE2 for phase 2

EXPERT    = 1        # use expert 1 labels (expert 2 also available)
DECIM     = 2        # 200 Hz → 100 Hz
N_SAMPLES = 3000     # 30s × 100Hz = 3000 samples per epoch

STAGE_MAP = {0: 0, 1: 1, 2: 2, 3: 3, 5: 4}  # ISRUC codes → 5 classes

def load_labels(path):
    with open(path) as f:
        vals = [ln.strip() for ln in f if ln.strip() != '']
    return np.array([int(float(v)) for v in vals], dtype=int)

def percentile_normalize_1d(epoch):
    """
    Scale one 1-D epoch to [-1, 1] via 5th-95th percentile.
    ISRUC is already in µV — NO ×1e6 conversion (unlike SleepEDF).
    Threshold is 1e-6 µV (not 1e-10 which was the SleepEDF Volt bug).
    """
    p05, p95 = np.percentile(epoch, 5), np.percentile(epoch, 95)
    rng = p95 - p05
    if rng < 1e-6:
        return np.zeros_like(epoch, dtype=np.float32)
    return (2.0 * (epoch - p05) / rng - 1.0).astype(np.float32)

def decimate_channel(sig_200hz, E, decim=2):
    """
    Decimate a (E, 6000) array from 200Hz to 100Hz.
    Reconstruct continuous signal first to avoid edge artifacts per epoch.
    """
    cont     = sig_200hz.reshape(-1)               # (E*6000,)
    cont100  = decimate(cont, decim, ftype='fir', zero_phase=True)
    return cont100.reshape(E, N_SAMPLES)            # (E, 3000)

# ── Verify channels exist in mat file ──
print(f"Verifying channels in subject 1...")
test_mat = sio.loadmat(
    os.path.join(MAT_BASE, 'subject1.mat'),
    squeeze_me=True, struct_as_record=False)
available = [k for k in test_mat.keys() if not k.startswith('_')]
print(f"Available keys: {available}")
print(f"Requested channels: {CHANNELS}")
missing = [c for c in CHANNELS if c not in available]
if missing:
    print(f"⚠️  MISSING channels: {missing}")
    print("Check channel name spelling against available keys above.")
else:
    print(f"✅ All {len(CHANNELS)} channels found")

# ── Main preprocessing loop ──
X_list, y_list, sid_list = [], [], []
n_channels = len(CHANNELS)

for n in range(1, 11):   # subjects 1–10
    mat_path = os.path.join(MAT_BASE, f'subject{n}.mat')
    d = sio.loadmat(mat_path, squeeze_me=True, struct_as_record=False)

    # Load all requested channels and decimate
    channels_decimated = []
    E_ref = None
    for ch in CHANNELS:
        sig = np.asarray(d[ch], dtype=np.float64)   # (E, 6000) at 200Hz
        if E_ref is None:
            E_ref = sig.shape[0]
        assert sig.shape[0] == E_ref, \
            f"subject{n}: channel {ch} has {sig.shape[0]} epochs, expected {E_ref}"
        sig100 = decimate_channel(sig, E_ref)         # (E, 3000) at 100Hz
        channels_decimated.append(sig100)

    # Stack channels: (n_channels, E, 3000) → (E, n_channels, 3000)
    sig_all = np.stack(channels_decimated, axis=0).transpose(1, 0, 2)
    # sig_all: (E, n_channels, 3000)

    # Load labels
    lbl_path = os.path.join(LBL_BASE, str(n), f'{n}_{EXPERT}.txt')
    labels   = load_labels(lbl_path)[:E_ref]
    assert len(labels) == E_ref, f"subject{n}: label mismatch"

    # Filter valid stages only (0,1,2,3,5 → skip anything else)
    valid_stages = set(STAGE_MAP.keys())

    for ep_idx in range(E_ref):
        lab = labels[ep_idx]
        if lab not in valid_stages:
            continue   # skip unknown/movement epochs

        # Normalize each channel independently
        ep = sig_all[ep_idx]                          # (n_channels, 3000)
        ep_norm = np.stack([
            percentile_normalize_1d(ep[c]) for c in range(n_channels)
        ], axis=0)                                    # (n_channels, 3000)

        X_list.append(ep_norm)
        y_list.append(STAGE_MAP[lab])
        sid_list.append(n - 1)   # subject id 0..9

    print(f"subject{n}: {E_ref} raw epochs → "
          f"{np.sum(np.isin(labels, list(valid_stages))):,} valid  "
          f"(running total: {len(y_list):,})")

# ── Stack and save ──
X           = np.stack(X_list).astype(np.float32)    # (N, n_channels, 3000)
y           = np.array(y_list, dtype=np.int64)
subject_ids = np.array(sid_list, dtype=np.int64)

# For ISRUC-S3: 1 recording per subject → recording_id = true_subject_id
# No need for separate true_subject_ids file

suffix = f'isruc_{PHASE_TAG}'
np.save(f'{SAVE}/X_raw_{suffix}.npy',         X)
np.save(f'{SAVE}/y_{suffix}.npy',             y)
np.save(f'{SAVE}/subject_ids_{suffix}.npy',   subject_ids)

print(f"\n{'='*60}")
print(f"✅ ISRUC-S3 preprocessing complete")
print(f"   Phase    : {PHASE_TAG}")
print(f"   Channels : {CHANNELS}")
print(f"   X shape  : {X.shape}")
print(f"   y shape  : {y.shape}")
print(f"   Subjects : {len(np.unique(subject_ids))}")
print(f"\nClass distribution:")
STAGE_NAMES = ['Wake', 'N1', 'N2', 'N3', 'REM']
for c, nm in enumerate(STAGE_NAMES):
    cnt = np.sum(y == c)
    print(f"  {nm:5s}: {cnt:6,}  ({100*cnt/len(y):5.1f}%)")
print(f"\nPer-subject epoch count:")
for s in range(10):
    cnt = np.sum(subject_ids == s)
    print(f"  Subject {s+1}: {cnt:4,} epochs")

# ── Amplitude sanity check ──
print(f"\nAmplitude check (channel 0 = {CHANNELS[0]}):")
for c, nm in enumerate(STAGE_NAMES):
    mask = y == c
    if mask.sum() == 0: continue
    ch0 = X[mask, 0, :]
    print(f"  {nm:5s}: mean={ch0.mean():.4f}  std={ch0.std():.4f}  "
          f"min={ch0.min():.3f}  max={ch0.max():.3f}")

# Verifying channels in subject 1...
# Available keys: ['F3_A2', 'C3_A2', 'O1_A2', 'F4_A1', 'C4_A1', 'O2_A1', 'ROC_A1', 'LOC_A2', 'X1', 'X2', 'X3']
# Requested channels: ['F3_A2', 'C3_A2', 'O1_A2', 'F4_A1', 'C4_A1', 'O2_A1', 'LOC_A2', 'ROC_A1', 'X1', 'X2', 'X3']
# ✅ All 11 channels found
# subject1: 924 raw epochs → 924 valid  (running total: 924)
# subject2: 911 raw epochs → 911 valid  (running total: 1,835)
# subject3: 794 raw epochs → 794 valid  (running total: 2,629)
# subject4: 764 raw epochs → 764 valid  (running total: 3,393)
# subject5: 914 raw epochs → 914 valid  (running total: 4,307)
# subject6: 823 raw epochs → 823 valid  (running total: 5,130)
# subject7: 784 raw epochs → 784 valid  (running total: 5,914)
# subject8: 970 raw epochs → 970 valid  (running total: 6,884)
# subject9: 939 raw epochs → 939 valid  (running total: 7,823)
# subject10: 766 raw epochs → 766 valid  (running total: 8,589)
# 
# ============================================================
# ✅ ISRUC-S3 preprocessing complete
#    Phase    : phase2_11ch
#    Channels : ['F3_A2', 'C3_A2', 'O1_A2', 'F4_A1', 'C4_A1', 'O2_A1', 'LOC_A2', 'ROC_A1', 'X1', 'X2', 'X3']
#    X shape  : (8589, 11, 3000)
#    y shape  : (8589,)
#    Subjects : 10
# 
# Class distribution:
#   Wake :  1,674  ( 19.5%)
#   N1   :  1,217  ( 14.2%)
#   N2   :  2,616  ( 30.5%)
#   N3   :  2,016  ( 23.5%)
#   REM  :  1,066  ( 12.4%)
# 
# Per-subject epoch count:
#   Subject 1:  924 epochs
#   Subject 2:  911 epochs
#   Subject 3:  794 epochs
#   Subject 4:  764 epochs
#   Subject 5:  914 epochs
#   Subject 6:  823 epochs
#   Subject 7:  784 epochs
#   Subject 8:  970 epochs
#   Subject 9:  939 epochs
#   Subject 10:  766 epochs
# 
# Amplitude check (channel 0 = F3_A2):
#   Wake : mean=0.0031  std=0.6460  min=-17.596  max=15.780
#   N1   : mean=-0.0027  std=0.6506  min=-16.992  max=17.304
#   N2   : mean=-0.0119  std=0.6751  min=-11.491  max=12.928
#   N3   : mean=0.0023  std=0.6257  min=-6.682  max=7.088
#   REM  : mean=0.0028  std=0.6237  min=-9.315  max=13.892