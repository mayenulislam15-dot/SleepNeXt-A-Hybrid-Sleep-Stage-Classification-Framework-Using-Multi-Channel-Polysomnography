# ═══════════════════════════════════════════════════════════════════════
# PREPROCESSING — Raw 1D Signal Only (no STFT)
# ═══════════════════════════════════════════════════════════════════════
import os, glob, gc
import numpy as np
import mne
from tqdm import tqdm

mne.set_log_level('WARNING')

BASE = '/kaggle/input/datasets/labibmh/sleepedf-78/sleep-edf-database-expanded/sleep-cassette'
SAVE = '/kaggle/working'
CHANNELS = ['EEG Fpz-Cz', 'EEG Pz-Oz', 'EOG horizontal']
SAMPLING_RATE, EPOCH_SEC, SAMPLES_PER_EPOCH, TRIM_EPOCHS = 100, 30, 3000, 60
STAGE_MAP = {
    'Sleep stage W': 0, 'Sleep stage 1': 1, 'Sleep stage 2': 2,
    'Sleep stage 3': 3, 'Sleep stage 4': 3, 'Sleep stage R': 4
}

# ── Pair files (SC files only) ──
psg_files = sorted(glob.glob(os.path.join(BASE, 'SC*PSG.edf')))
hyp_files = sorted(glob.glob(os.path.join(BASE, 'SC*Hypnogram.edf')))
psg_dict = {os.path.basename(f)[:6]: f for f in psg_files}
hyp_dict = {os.path.basename(f)[:6]: f for f in hyp_files}
paired = [(psg_dict[s], hyp_dict[s]) for s in sorted(set(psg_dict) & set(hyp_dict))]
print(f"Paired subjects: {len(paired)}")

# ── Normalization helper ──
def percentile_normalize(epoch_data):
    """Convert V → µV, then scale each channel to [-1, 1] using 5th–95th percentile."""
    epoch_uv = epoch_data * 1e6          # Volts → microvolts
    out = np.zeros_like(epoch_uv, dtype=np.float32)
    for c in range(epoch_uv.shape[0]):
        p05 = np.percentile(epoch_uv[c], 5)
        p95 = np.percentile(epoch_uv[c], 95)
        rng = p95 - p05
        out[c] = (2.0 * (epoch_uv[c] - p05) / rng - 1.0) if rng > 1e-6 else 0.0
    return out

def process_one_file(psg_path, hyp_path):
    raw = mne.io.read_raw_edf(psg_path, preload=True)
    if any(c not in raw.ch_names for c in CHANNELS):
        return None, None
    raw.pick_channels(CHANNELS).reorder_channels(CHANNELS)
    data = raw.get_data()

    ann, all_epochs = mne.read_annotations(hyp_path), []
    for entry in ann:
        desc = entry['description']
        n = int(round(entry['duration'] / EPOCH_SEC))
        for i in range(n):
            start = int((entry['onset'] + i * EPOCH_SEC) * SAMPLING_RATE)
            if start + SAMPLES_PER_EPOCH > data.shape[1]:
                break
            if desc in STAGE_MAP:
                all_epochs.append((start, STAGE_MAP[desc]))
            elif desc == 'Sleep stage ?':
                all_epochs.append((start, None))

    sleep_idx = [i for i, (_, l) in enumerate(all_epochs) if l is not None and l != 0]
    if not sleep_idx:
        return None, None

    t0 = max(0, sleep_idx[0] - TRIM_EPOCHS)
    t1 = min(len(all_epochs) - 1, sleep_idx[-1] + TRIM_EPOCHS)
    valid = [(s, l) for s, l in all_epochs[t0:t1+1] if l is not None]
    if not valid:
        return None, None

    epochs = np.array([data[:, s:s+SAMPLES_PER_EPOCH] for s, _ in valid], dtype=np.float32)
    labels = np.array([l for _, l in valid], dtype=np.int64)
    return epochs, labels

# ── Pass 1: Count total epochs ──
print("Pass 1: Counting epochs...")
total_epochs = 0
for psg, hyp in tqdm(paired):
    try:
        _, lbl = process_one_file(psg, hyp)
        if lbl is not None:
            total_epochs += len(lbl)
    except:
        pass
print(f"Total epochs: {total_epochs}")

# ── Pass 2: Extract, normalize, save ──
# Shape: (N_epochs, 3_channels, 3000_timepoints)
X_raw_mmap = np.lib.format.open_memmap(
    f'{SAVE}/X_raw.npy', mode='w+', dtype='float32',
    shape=(total_epochs, 3, 3000))
y_arr    = np.zeros(total_epochs, dtype=np.int64)
sids_arr = np.zeros(total_epochs, dtype=np.int64)

print("Pass 2: Extracting and saving...")
idx = 0
for subj_idx, (psg, hyp) in enumerate(tqdm(paired)):
    try:
        raw_eps, labels = process_one_file(psg, hyp)
        if raw_eps is None:
            continue
        n = len(labels)
        norm_raw = np.array([percentile_normalize(ep) for ep in raw_eps], dtype=np.float32)
        X_raw_mmap[idx:idx+n] = norm_raw
        y_arr[idx:idx+n]    = labels
        sids_arr[idx:idx+n] = subj_idx
        idx += n
    except:
        continue

np.save(f'{SAVE}/y.npy',           y_arr[:idx])
np.save(f'{SAVE}/subject_ids.npy', sids_arr[:idx])

# Trim if needed
if idx < total_epochs:
    X_final = np.lib.format.open_memmap(
        f'{SAVE}/X_raw_final.npy', mode='w+', dtype='float32',
        shape=(idx, 3, 3000))
    X_final[:] = X_raw_mmap[:idx]
    os.rename(f'{SAVE}/X_raw_final.npy', f'{SAVE}/X_raw.npy')

print(f"\n✅ DONE. Saved {idx} epochs → X_raw.npy shape: ({idx}, 3, 3000)")


X = np.load(f'{SAVE}/X_raw.npy', mmap_mode='r')
# Find and remove bad epochs before training
print("Finding bad epochs...")
bad_mask = np.zeros(len(X), dtype=bool)

for i in range(0, len(X), 1000):
    chunk = X[i:i+1000]
    # Bad if any channel has std > 50 (clearly unnormalized)
    chunk_std = chunk.reshape(len(chunk), 3, 3000).std(axis=2)
    bad = (chunk_std > 50).any(axis=1)
    bad_mask[i:i+len(chunk)] = bad

bad_idx = np.where(bad_mask)[0]
print(f"Bad epochs found: {len(bad_idx)}")
print(f"Bad epoch indices: {bad_idx}")

# Create clean index — all epochs except bad ones
clean_mask = ~bad_mask
clean_idx  = np.where(clean_mask)[0]
print(f"Clean epochs: {len(clean_idx):,} / {len(X):,}")

#Finding bad epochs...
#Bad epochs found: 75
#Bad epoch indices: [73632 73639 73649 73652 73656 73657 73659 73680 73683 73685 73741 73753
# 73768 73790 73793 73806 73807 73808 73832 73834 73844 73855 73893 73939
# 73953 73984 74012 74053 74055 74061 74070 74079 74092 74114 74131 74157
# 74164 74198 74321 74383 74386 74389 74418 74421 74424 74439 74462 74464
# 74471 74483 74549 74589 74599 74618 74637 74653 74719 74721 74797 74824
# 74845 74877 74880 74933 74957 75025 75057 75060 75090 75091 75131 75145
# 75147 75150 75155]
#Clean epochs: 195,395 / 195,470


import numpy as np
import os, re, glob

# Find the original EDF files to confirm the naming pattern
edf_dir = '/kaggle/input/datasets/labibmh/sleepedf-78/sleep-edf-database-expanded/sleep-cassette'  # your sleep-cassette path
psg_files = sorted(glob.glob(os.path.join(edf_dir, '*PSG.edf')))

print(f"Total PSG files: {len(psg_files)}")
print("\nFirst 10 filenames and their subject mapping:")
for i, f in enumerate(psg_files[:10]):
    name = os.path.basename(f)
    match = re.search(r'SC4(\d{2})(\d)', name)
    subj = int(match.group(1))
    night = int(match.group(2))
    print(f"  Recording {i:3d}: {name} → subject {subj}, night {night}")

#Total PSG files: 153

#First 10 filenames and their subject mapping:
#  Recording   0: SC4001E0-PSG.edf → subject 0, night 1
#  Recording   1: SC4002E0-PSG.edf → subject 0, night 2
#  Recording   2: SC4011E0-PSG.edf → subject 1, night 1
#  Recording   3: SC4012E0-PSG.edf → subject 1, night 2
#  Recording   4: SC4021E0-PSG.edf → subject 2, night 1
#  Recording   5: SC4022E0-PSG.edf → subject 2, night 2
#  Recording   6: SC4031E0-PSG.edf → subject 3, night 1
#  Recording   7: SC4032E0-PSG.edf → subject 3, night 2
#  Recording   8: SC4041E0-PSG.edf → subject 4, night 1
#  Recording   9: SC4042E0-PSG.edf → subject 4, night 2


import numpy as np
import os, re, glob

SAVE    = '/kaggle/working'
edf_dir = '/kaggle/input/datasets/labibmh/sleepedf-78/sleep-edf-database-expanded/sleep-cassette'   # ← your sleep-cassette folder path

# ── Load existing arrays ──
y    = np.load(f'{SAVE}/y.npy')
sids = np.load(f'{SAVE}/subject_ids.npy')   # current recording IDs (0-152)

# ── Get the sorted PSG files (same order as preprocessing) ──
psg_files = sorted(glob.glob(os.path.join(edf_dir, '*PSG.edf')))
print(f"Total PSG files: {len(psg_files)}")

# ── Build recording_id → true_subject_id map ──
rec_to_subj = {}
for rec_id, f in enumerate(psg_files):
    name  = os.path.basename(f)
    match = re.search(r'SC4(\d{2})(\d)', name)
    subj  = int(match.group(1))   # the 'ss' digits = TRUE subject
    rec_to_subj[rec_id] = subj

# ── Verify the mapping around the known missing nights ──
print("\nChecking subjects 13, 36, 52 (known missing nights):")
for rec_id in range(len(psg_files)):
    name = os.path.basename(psg_files[rec_id])
    match = re.search(r'SC4(\d{2})(\d)', name)
    subj, night = int(match.group(1)), int(match.group(2))
    if subj in [12, 13, 14, 35, 36, 37, 51, 52, 53]:
        print(f"  Recording {rec_id:3d}: {name} → subject {subj}, night {night}")

# ── Create true_subject_ids array (per epoch) ──
true_subject_ids = np.array([rec_to_subj[rec] for rec in sids])

# ── Save ──
np.save(f'{SAVE}/true_subject_ids.npy', true_subject_ids)

print(f"\n✅ Saved true_subject_ids.npy")
print(f"Unique recordings (old): {len(np.unique(sids))}")
print(f"Unique TRUE subjects    : {len(np.unique(true_subject_ids))}")
print(f"Total epochs            : {len(true_subject_ids)}")


# Total PSG files: 153

# Checking subjects 13, 36, 52 (known missing nights):
#   Recording  24: SC4121E0-PSG.edf → subject 12, night 1
#   Recording  25: SC4122E0-PSG.edf → subject 12, night 2
#   Recording  26: SC4131E0-PSG.edf → subject 13, night 1
#   Recording  27: SC4141E0-PSG.edf → subject 14, night 1
#   Recording  28: SC4142E0-PSG.edf → subject 14, night 2
#   Recording  69: SC4351F0-PSG.edf → subject 35, night 1
#   Recording  70: SC4352F0-PSG.edf → subject 35, night 2
#   Recording  71: SC4362F0-PSG.edf → subject 36, night 2
#   Recording  72: SC4371F0-PSG.edf → subject 37, night 1
#   Recording  73: SC4372F0-PSG.edf → subject 37, night 2
#   Recording  98: SC4511E0-PSG.edf → subject 51, night 1
#   Recording  99: SC4512E0-PSG.edf → subject 51, night 2
#   Recording 100: SC4522E0-PSG.edf → subject 52, night 2
#   Recording 101: SC4531E0-PSG.edf → subject 53, night 1
#   Recording 102: SC4532E0-PSG.edf → subject 53, night 2

# ✅ Saved true_subject_ids.npy
# Unique recordings (old): 153
# Unique TRUE subjects    : 78
# Total epochs            : 195470

import numpy as np
SAVE = '/kaggle/working'

true_subj = np.load(f'{SAVE}/true_subject_ids.npy')
sids      = np.load(f'{SAVE}/subject_ids.npy')
y         = np.load(f'{SAVE}/y.npy')

# SleepEDF-20 = first 20 true subjects
mask20 = true_subj < 20
print(f"SleepEDF-20 (true subjects 0-19):")
print(f"  True subjects   : {len(np.unique(true_subj[mask20]))}")
print(f"  Recordings      : {len(np.unique(sids[mask20]))}")
print(f"  Total epochs    : {mask20.sum():,}")
print(f"  Class dist      : {np.bincount(y[mask20], minlength=5)}")

# Compare to old recording-based filter
mask20_old = sids < 40
print(f"\nOld filter (recordings < 40):")
print(f"  Epochs          : {mask20_old.sum():,}")
print(f"  Match new filter: {np.array_equal(mask20, mask20_old)}")

# SleepEDF-20 (true subjects 0-19):
#   True subjects   : 20
#   Recordings      : 39
#   Total epochs    : 42,308
#   Class dist      : [ 8285  2804 17799  5703  7717]
# 
# Old filter (recordings < 40):
#   Epochs          : 43,330
#   Match new filter: False