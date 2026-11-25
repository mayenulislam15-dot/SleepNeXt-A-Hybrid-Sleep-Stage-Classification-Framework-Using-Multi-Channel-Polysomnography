# ════════════════════════════════════════════════════════════════════════════
# SleepNeXt — Result analysis for Sleep-EDF-78, Sleep-EDF-20 and ISRUC-S3
# ════════════════════════════════════════════════════════════════════════════
#
# WHAT THIS SCRIPT DOES
#   Step 1: de-duplicates the raw per-fold test results of every dataset.
#     The test loader uses non-overlapping 20-epoch sequences plus one
#     overlapping tail sequence per recording, so a few epochs are predicted twice.
#     For each such epoch the FIRST prediction is kept, giving exactly one
#     prediction per 30-s epoch. (Skipped if the de-duplicated files already exist.)
#   Step 2 (all datasets): checks the de-duplicated results and reports
#     • per-fold metrics and fold mean ± std (Acc, MF1, WF1, Cohen's kappa,
#       per-class F1) — these are the values reported in the paper
#     • pooled metrics over all test epochs (for reference only)
#     • confusion matrices (raw counts and row-normalized %), per-class F1 plots,
#       CSV / LaTeX tables and a JSON summary
#
# REQUIRED FILES  (folder layout relative to ROOT_DIR)
#
#   ROOT_DIR/
#   ├── sleepedf/
#   │     y.npy                                       labels of all Sleep-EDF-78 epochs
#   │     subject_ids.npy                             recording (night) ID of each epoch
#   │     true_subject_ids.npy                        subject ID (0–77) of each epoch
#   │     fold_1_SleepNeXt_sleepedf78_results.pkl … fold_10_…   (10 raw files)
#   │     fold_1_SleepNeXt_sleepedf20_results.pkl … fold_20_…   (20 raw files)
#   │     (deduplicated_results/ is created by Step 1)
#   └── isruc_s3/
#         y_isruc_phase2_11ch.npy                     labels of all 8,589 epochs
#         subject_ids_isruc_phase2_11ch.npy           subject ID (0–9) of each epoch
#         fold_1_SleepNeXt_isruc_s3_phase2_11ch_results.pkl … fold_10_…   (10 raw files)
#         (deduplicated_results/ is created by Step 1)
#
#   The *_results.pkl / *_results_dedup.pkl files are written by the SleepNeXt
#   training script (one file per fold). Each must be a dict with at least:
#     'fold'         int, 1-based fold number
#     'labels'       list of true stages (0=W, 1=N1, 2=N2, 3=N3, 4=REM)
#     'predictions'  list of predicted stages, same order as 'labels'
#   De-duplicated files additionally contain 'deduplicated_evaluation' (True),
#   'confusion_matrix', 'n_test_epochs' and 'duplicate_predictions_removed'.
#
# CONSISTENCY CHECKS  (the script stops with an error if any fails)
#   • all folds present: 10 (Sleep-EDF-78), 20 (Sleep-EDF-20), 10 (ISRUC-S3)
#   • reconstructed test order reproduces the saved labels exactly, and every
#     held-out epoch is scored exactly once (Sleep-EDF: after removing BAD_EPOCHS)
#   • stored confusion matrices match the ones recomputed from labels/predictions
#   • unique test epochs: 195,395 / 42,308 / 8,589; ISRUC-S3 class counts
#     W 1,674 | N1 1,217 | N2 2,616 | N3 2,016 | REM 1,066
#
# HOW TO RUN
#   pip install numpy pandas matplotlib seaborn scikit-learn
#   python sleepnext_results_analysis.py
#   (set ROOT_DIR below if your files are elsewhere)
#
# OUTPUTS  (ROOT_DIR/analysis_outputs/)
#   per_fold_<dataset>.csv, summary_<dataset>.csv, results_<dataset>.tex,
#   confusion_counts_<dataset>.csv, confusion_row_normalized_<dataset>.csv,
#   confusion_both_<dataset>.png, confusion_raw_<dataset>.png,
#   confusion_row_normalized_<dataset>.png, per_class_f1_<dataset>.png,
#   summary_all_datasets.json
#   with <dataset> ∈ {sleepedf78, sleepedf20, isruc_s3}
# ════════════════════════════════════════════════════════════════════════════
import os
import glob
import json
import pickle
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold
from sklearn.metrics import (accuracy_score, f1_score, precision_score, recall_score,
                             cohen_kappa_score, confusion_matrix, classification_report)

# ═══════════════════════════ CONFIGURATION ═══════════════════════════
ROOT_DIR   = './results'
SHOW_PLOTS = False                 # True to display figures interactively

SLEEPEDF_DIR       = os.path.join(ROOT_DIR, 'sleepedf')
SLEEPEDF_DEDUP_DIR = os.path.join(SLEEPEDF_DIR, 'deduplicated_results')
ISRUC_DIR          = os.path.join(ROOT_DIR, 'isruc_s3')
ISRUC_DEDUP_DIR    = os.path.join(ISRUC_DIR, 'deduplicated_results')
OUTPUT_DIR         = os.path.join(ROOT_DIR, 'analysis_outputs')

MODEL_NAME   = 'SleepNeXt'
STAGE_NAMES  = ['Wake', 'N1', 'N2', 'N3', 'REM']
STAGE_COLORS = ['#e74c3c', '#f39c12', '#2ecc71', '#3498db', '#9b59b6']
LABELS       = [0, 1, 2, 3, 4]

SEQ_LEN       = 20                 # test sequence length = test stride (all datasets)
ISRUC_N_FOLDS = 10

# Sleep-EDF epochs excluded from training and testing (same list as the training script)
BAD_EPOCHS = {
    73632, 73639, 73649, 73652, 73656, 73657, 73659, 73680, 73683, 73685, 73741, 73753, 73768,
    73790, 73793, 73806, 73807, 73808, 73832, 73834, 73844, 73855, 73893, 73939, 73953, 73984,
    74012, 74053, 74055, 74061, 74070, 74079, 74092, 74114, 74131, 74157, 74164, 74198, 74321,
    74383, 74386, 74389, 74418, 74421, 74424, 74439, 74462, 74464, 74471, 74483, 74549, 74589,
    74599, 74618, 74637, 74653, 74719, 74721, 74797, 74824, 74845, 74877, 74880, 74933, 74957,
    75025, 75057, 75060, 75090, 75091, 75131, 75145, 75147, 75150, 75155}

DATASETS = {
    'sleepedf78': dict(label='Sleep-EDF-78', n_folds=10, n_epochs=195395, class_counts=None,
                       pattern=os.path.join(SLEEPEDF_DEDUP_DIR, f'fold_*_{MODEL_NAME}_sleepedf78_results_dedup.pkl')),
    'sleepedf20': dict(label='Sleep-EDF-20', n_folds=20, n_epochs=42308, class_counts=None,
                       pattern=os.path.join(SLEEPEDF_DEDUP_DIR, f'fold_*_{MODEL_NAME}_sleepedf20_results_dedup.pkl')),
    'isruc_s3':   dict(label='ISRUC-S3', n_folds=10, n_epochs=8589, class_counts=[1674, 1217, 2616, 2016, 1066],
                       pattern=os.path.join(ISRUC_DEDUP_DIR, f'fold_*_{MODEL_NAME}_isruc_s3_phase2_11ch_results_dedup.pkl')),
}
F1_COLS = ['W F1 (%)', 'N1 F1 (%)', 'N2 F1 (%)', 'N3 F1 (%)', 'REM F1 (%)']
METRIC_COLS = ['Acc (%)', 'MF1 (%)', 'WF1 (%)', 'Kappa'] + F1_COLS

if not SHOW_PLOTS:
    matplotlib.use('Agg')


# ═══════════════════════════ SHARED HELPERS ═══════════════════════════
def fold_metrics(labels, preds):
    labels = np.asarray(labels, dtype=np.int64)
    preds  = np.asarray(preds, dtype=np.int64)
    return {
        'accuracy'           : accuracy_score(labels, preds),
        'macro_f1'           : f1_score(labels, preds, average='macro', labels=LABELS, zero_division=0),
        'weighted_f1'        : f1_score(labels, preds, average='weighted', labels=LABELS, zero_division=0),
        'kappa'              : cohen_kappa_score(labels, preds),
        'per_class_f1'       : f1_score(labels, preds, average=None, labels=LABELS, zero_division=0),
        'per_class_precision': precision_score(labels, preds, average=None, labels=LABELS, zero_division=0),
        'per_class_recall'   : recall_score(labels, preds, average=None, labels=LABELS, zero_division=0),
        'cm'                 : confusion_matrix(labels, preds, labels=LABELS),
    }


def finish_figure(path):
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches='tight')
    if SHOW_PLOTS:
        plt.show()
    plt.close()
    print(f'  Saved: {path}')


# ═══════════════════ STEP 1 — DE-DUPLICATION (ALL DATASETS) ═══════════════════
def build_test_sequences(group_ids, test_idx, seq_len=SEQ_LEN):
    """Exact reconstruction of the test-sequence builder used in training
    (SleepSequenceDataset with stride = seq_len): non-overlapping sequences per
    recording plus one overlapping tail sequence when the recording length is not
    divisible by seq_len. group_ids = recording IDs (Sleep-EDF) or subject IDs (ISRUC-S3)."""
    sequences = []
    test_mask = np.isin(np.arange(len(group_ids)), test_idx)
    for subj in np.unique(group_ids[test_idx]):
        subj_idx = np.where((group_ids == subj) & test_mask)[0]
        n = len(subj_idx)
        if n < seq_len:
            continue
        start = 0
        while start + seq_len <= n:
            sequences.append(subj_idx[start:start + seq_len])
            start += seq_len
        if start > seq_len and start < n:
            sequences.append(subj_idx[n - seq_len:n])
    return sequences


def test_epoch_order(key, fold_number, data):
    """Rebuilds the subject-wise split used in training,
    KFold(n_splits=K, shuffle=True, random_state=42) over unique subjects, and the
    exact global epoch order sent to the test DataLoader for one fold."""
    y, true_subj, group = data['y'], data['true_subject_ids'], data['group_ids']
    if key == 'sleepedf20':
        valid = true_subj < 20                                   # Sleep-EDF-20 = first 20 subjects
    else:
        valid = np.ones(len(y), dtype=bool)
    unique_subj = np.unique(true_subj[valid])
    splits    = list(KFold(n_splits=DATASETS[key]['n_folds'], shuffle=True, random_state=42).split(unique_subj))
    test_subj = unique_subj[splits[fold_number - 1][1]]
    test_idx  = np.where(np.isin(true_subj, test_subj) & valid)[0]
    if key.startswith('sleepedf'):
        test_idx = np.asarray([i for i in test_idx if int(i) not in BAD_EPOCHS], dtype=np.int64)
    order = np.concatenate(build_test_sequences(group, test_idx))
    return order, test_idx, test_subj


def deduplicate_fold(key, path, data, out_dir):
    y = data['y']
    with open(path, 'rb') as f:
        result = pickle.load(f)
    fold   = int(result['fold'])
    preds  = np.asarray(result['predictions'], dtype=np.int64)
    labels = np.asarray(result['labels'], dtype=np.int64)
    order, test_idx, test_subj = test_epoch_order(key, fold, data)

    assert len(preds) == len(labels), f'Fold {fold}: prediction/label length mismatch.'
    assert len(preds) == len(order), (f'Fold {fold}: saved prediction count ({len(preds)}) does not '
                                      f'match the reconstructed loader count ({len(order)}).')
    assert np.array_equal(labels, y[order]), (
        f'Fold {fold}: reconstructed labels do not match the saved labels. Check that the files were '
        'generated with the same arrays, sequence length 20, test stride 20, KFold random_state=42 '
        'and the same BAD_EPOCHS list.')

    # keep the FIRST prediction of every original epoch
    _, first = np.unique(order, return_index=True)
    first = np.sort(first)
    d_preds, d_labels, d_epochs = preds[first], labels[first], order[first]
    assert len(np.unique(d_epochs)) == len(d_epochs)
    assert set(d_epochs) == set(test_idx), f'Fold {fold}: not every held-out epoch is scored exactly once.'

    m = fold_metrics(d_labels, d_preds)
    revised = dict(result)
    revised.update({
        'accuracy': float(m['accuracy']), 'macro_f1': float(m['macro_f1']),
        'weighted_f1': float(m['weighted_f1']), 'kappa': float(m['kappa']),
        'per_class_f1': m['per_class_f1'].tolist(),
        'per_class_precision': m['per_class_precision'].tolist(),
        'per_class_recall': m['per_class_recall'].tolist(),
        'confusion_matrix': m['cm'].tolist(),
        'predictions': d_preds.tolist(), 'labels': d_labels.tolist(),
        'epoch_indices': d_epochs.tolist(), 'n_test_epochs': int(len(d_labels)),
        'deduplicated_evaluation': True,
        'deduplication_method': 'First occurrence retained for overlapping test tail windows',
        'original_prediction_count': int(len(preds)),
        'duplicate_predictions_removed': int(len(preds) - len(d_preds)),
    })
    out_path = os.path.join(out_dir, os.path.basename(path).replace('_results.pkl', '_results_dedup.pkl'))
    with open(out_path, 'wb') as f:
        pickle.dump(revised, f)
    print(f'  Fold {fold:02d} | test subject(s) {test_subj.tolist()} | original {len(preds):6,} | '
          f'unique {len(d_preds):6,} | removed {len(preds) - len(d_preds):4,}')


def load_arrays(key):
    if key.startswith('sleepedf'):
        rec = np.load(os.path.join(SLEEPEDF_DIR, 'subject_ids.npy'))            # recording (night) IDs
        return dict(y=np.load(os.path.join(SLEEPEDF_DIR, 'y.npy')), group_ids=rec,
                    true_subject_ids=np.load(os.path.join(SLEEPEDF_DIR, 'true_subject_ids.npy')))
    sid = np.load(os.path.join(ISRUC_DIR, 'subject_ids_isruc_phase2_11ch.npy'))  # one recording per subject
    return dict(y=np.load(os.path.join(ISRUC_DIR, 'y_isruc_phase2_11ch.npy')), group_ids=sid, true_subject_ids=sid)


RAW_PATTERNS = {
    'sleepedf78': (SLEEPEDF_DIR, SLEEPEDF_DEDUP_DIR, f'fold_*_{MODEL_NAME}_sleepedf78_results.pkl'),
    'sleepedf20': (SLEEPEDF_DIR, SLEEPEDF_DEDUP_DIR, f'fold_*_{MODEL_NAME}_sleepedf20_results.pkl'),
    'isruc_s3':   (ISRUC_DIR,    ISRUC_DEDUP_DIR,    f'fold_*_{MODEL_NAME}_isruc_s3_phase2_11ch_results.pkl'),
}


def run_deduplication(key):
    cfg = DATASETS[key]
    if len(glob.glob(cfg['pattern'])) == cfg['n_folds']:
        print(f'{cfg["label"]}: de-duplicated files already present — skipped.')
        return
    in_dir, out_dir, pattern = RAW_PATTERNS[key]
    files = glob.glob(os.path.join(in_dir, pattern))
    if len(files) != cfg['n_folds']:
        raise RuntimeError(f'{cfg["label"]}: expected {cfg["n_folds"]} raw result files in {in_dir}, found {len(files)}.')
    os.makedirs(out_dir, exist_ok=True)
    data = load_arrays(key)
    print(f'{cfg["label"]}: de-duplicating {len(files)} folds')
    for path in sorted(files, key=lambda p: int(os.path.basename(p).split('_')[1])):
        deduplicate_fold(key, path, data, out_dir)


# ═══════════════════ STEP 2 — ANALYSIS (ALL DATASETS) ═══════════════════
def load_results(key):
    cfg   = DATASETS[key]
    files = glob.glob(cfg['pattern'])
    if not files:
        raise FileNotFoundError(f'No de-duplicated files found for {cfg["label"]}:\n{cfg["pattern"]}')
    results = []
    for path in files:
        with open(path, 'rb') as f:
            r = pickle.load(f)
        if not r.get('deduplicated_evaluation', False):
            raise ValueError(f'{os.path.basename(path)} is not marked as de-duplicated.')
        results.append(r)
    results.sort(key=lambda r: int(r['fold']))
    fold_ids = [int(r['fold']) for r in results]
    if fold_ids != list(range(1, cfg['n_folds'] + 1)):
        raise ValueError(f'{cfg["label"]}: expected folds 1–{cfg["n_folds"]}, got {fold_ids}')
    return results


def summarize_dataset(key):
    cfg = DATASETS[key]
    results = load_results(key)
    rows, all_labels, all_preds = [], [], []
    for r in results:
        fold   = int(r['fold'])
        labels = np.asarray(r['labels'], dtype=np.int64)
        preds  = np.asarray(r['predictions'], dtype=np.int64)
        m = fold_metrics(labels, preds)
        if not np.array_equal(m['cm'], np.asarray(r['confusion_matrix'], dtype=np.int64)):
            raise ValueError(f'{cfg["label"]}, fold {fold}: stored confusion matrix does not match the recomputed one.')
        if len(labels) != int(r['n_test_epochs']):
            raise ValueError(f'{cfg["label"]}, fold {fold}: n_test_epochs mismatch.')
        rows.append({'Fold': fold, 'Test epochs': len(labels),
                     'Duplicates removed': int(r.get('duplicate_predictions_removed', 0)),
                     'Acc (%)': m['accuracy'] * 100, 'MF1 (%)': m['macro_f1'] * 100,
                     'WF1 (%)': m['weighted_f1'] * 100, 'Kappa': m['kappa'],
                     **{c: v * 100 for c, v in zip(F1_COLS, m['per_class_f1'])}})
        all_labels.append(labels); all_preds.append(preds)

    all_labels, all_preds = np.concatenate(all_labels), np.concatenate(all_preds)
    if len(all_labels) != cfg['n_epochs']:
        raise ValueError(f'{cfg["label"]}: expected {cfg["n_epochs"]:,} unique epochs, got {len(all_labels):,}.')
    pooled = fold_metrics(all_labels, all_preds)
    if cfg['class_counts'] is not None and not np.array_equal(pooled['cm'].sum(axis=1), cfg['class_counts']):
        raise ValueError(f'{cfg["label"]}: class counts {pooled["cm"].sum(axis=1).tolist()} '
                         f'differ from the expected {cfg["class_counts"]}.')

    df = pd.DataFrame(rows).sort_values('Fold').reset_index(drop=True)
    mean_row = {'Fold': 'Mean', 'Test epochs': int(df['Test epochs'].mean()),
                'Duplicates removed': int(df['Duplicates removed'].sum())}
    std_row  = {'Fold': 'Std', 'Test epochs': np.nan, 'Duplicates removed': np.nan}
    for c in METRIC_COLS:
        mean_row[c] = df[c].mean()
        std_row[c]  = df[c].std(ddof=0)
    summary_df = pd.concat([df, pd.DataFrame([mean_row, std_row])], ignore_index=True)
    return dict(key=key, label=cfg['label'], fold_df=df, summary_df=summary_df,
                all_labels=all_labels, all_preds=all_preds, pooled=pooled,
                mean_row=mean_row, std_row=std_row)


def print_summary(a):
    m, s, p = a['mean_row'], a['std_row'], a['pooled']
    print('\n' + '=' * 96)
    print(f'{a["label"].upper()} — DE-DUPLICATED RESULTS (one prediction per 30-s epoch)')
    print('=' * 96)
    print(a['fold_df'].to_string(index=False, float_format=lambda x: f'{x:.2f}'))
    print('\nPER-FOLD MEAN ± STD (reported in the paper)')
    print(f'Acc   : {m["Acc (%)"]:.2f} ± {s["Acc (%)"]:.2f}%\n'
          f'MF1   : {m["MF1 (%)"]:.2f} ± {s["MF1 (%)"]:.2f}%\n'
          f'WF1   : {m["WF1 (%)"]:.2f} ± {s["WF1 (%)"]:.2f}%\n'
          f'Kappa : {m["Kappa"]:.3f} ± {s["Kappa"]:.3f}')
    print('\nPER-CLASS F1: MEAN ± STD')
    for stage, c in zip(STAGE_NAMES, F1_COLS):
        print(f'  {stage:5s}: {m[c]:.2f} ± {s[c]:.2f}%')
    print('\nPOOLED METRICS (all unique test epochs; reference only)')
    print(f'Unique epochs : {len(a["all_labels"]):,}')
    print(f'Accuracy      : {p["accuracy"] * 100:.2f}%')
    print(f'Macro F1      : {p["macro_f1"] * 100:.2f}%')
    print(f'Weighted F1   : {p["weighted_f1"] * 100:.2f}%')
    print(f"Cohen's kappa : {p['kappa']:.4f}")
    print('\nCLASSIFICATION REPORT (pooled)')
    print(classification_report(a['all_labels'], a['all_preds'], labels=LABELS,
                                target_names=STAGE_NAMES, digits=4, zero_division=0))


def save_tables(a):
    key, m, s, cm = a['key'], a['mean_row'], a['std_row'], a['pooled']['cm']
    paths = {
        'fold'   : os.path.join(OUTPUT_DIR, f'per_fold_{key}.csv'),
        'summary': os.path.join(OUTPUT_DIR, f'summary_{key}.csv'),
        'latex'  : os.path.join(OUTPUT_DIR, f'results_{key}.tex'),
        'cm'     : os.path.join(OUTPUT_DIR, f'confusion_counts_{key}.csv'),
        'cm_pct' : os.path.join(OUTPUT_DIR, f'confusion_row_normalized_{key}.csv'),
    }
    a['fold_df'].to_csv(paths['fold'], index=False)
    a['summary_df'].to_csv(paths['summary'], index=False)
    np.savetxt(paths['cm'], cm, delimiter=',', fmt='%d', header=','.join(STAGE_NAMES), comments='')
    np.savetxt(paths['cm_pct'], cm / cm.sum(axis=1, keepdims=True) * 100, delimiter=',', fmt='%.4f',
               header=','.join(STAGE_NAMES), comments='')
    latex = ('\\begin{table}[t]\n\\centering\n'
             f'\\caption{{SleepNeXt results on {a["label"]} (mean $\\pm$ std across folds).}}\n'
             f'\\label{{tab:results-{key}}}\n\\scriptsize\n'
             '\\begin{tabular}{lccc}\n\\toprule\n'
             ' & Acc (\\%) & MF1 (\\%) & $\\kappa$ \\\\\n\\midrule\n'
             f'SleepNeXt & {m["Acc (%)"]:.2f} $\\pm$ {s["Acc (%)"]:.2f} & '
             f'{m["MF1 (%)"]:.2f} $\\pm$ {s["MF1 (%)"]:.2f} & '
             f'{m["Kappa"]:.3f} $\\pm$ {s["Kappa"]:.3f} \\\\\n'
             '\\bottomrule\n\\end{tabular}\n\\end{table}\n')
    with open(paths['latex'], 'w', encoding='utf-8') as f:
        f.write(latex)
    for p in paths.values():
        print(f'  Saved: {p}')


def _heatmap(data, ax, pct, title):
    sns.heatmap(data, annot=True, fmt='.1f' if pct else 'd', cmap='Blues',
                xticklabels=STAGE_NAMES, yticklabels=STAGE_NAMES, linewidths=0.35, linecolor='white',
                vmin=0 if pct else None, vmax=100 if pct else None,
                cbar_kws={'label': 'True-class proportion (%)' if pct else 'Epoch count'}, ax=ax)
    ax.set_title(title, fontweight='bold')
    ax.set_xlabel('Predicted stage')
    ax.set_ylabel('True stage')


def plot_confusion_matrices(a):
    sns.set_theme(style='white', font_scale=1.05)
    cm = a['pooled']['cm']
    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100
    key, lab = a['key'], a['label']

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.8))
    _heatmap(cm, axes[0], False, f'{lab}: Raw Counts')
    _heatmap(cm_pct, axes[1], True, f'{lab}: Row-Normalized (%)')
    fig.suptitle(f'SleepNeXt Confusion Matrices — {lab}', fontsize=15, fontweight='bold', y=1.02)
    finish_figure(os.path.join(OUTPUT_DIR, f'confusion_both_{key}.png'))

    fig, ax = plt.subplots(figsize=(7.2, 6.0))
    _heatmap(cm, ax, False, f'Confusion Matrix — {lab}\nRaw Counts')
    finish_figure(os.path.join(OUTPUT_DIR, f'confusion_raw_{key}.png'))

    fig, ax = plt.subplots(figsize=(7.2, 6.0))
    _heatmap(cm_pct, ax, True, f'Confusion Matrix — {lab}\nRow-Normalized (%)')
    finish_figure(os.path.join(OUTPUT_DIR, f'confusion_row_normalized_{key}.png'))


def plot_per_class_f1(a):
    df = a['fold_df']
    means, stds = df[F1_COLS].mean().to_numpy(), df[F1_COLS].std(ddof=0).to_numpy()
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    bars = ax.bar(STAGE_NAMES, means, yerr=stds, capsize=7, color=STAGE_COLORS, edgecolor='white', linewidth=1.0)
    ax.set_ylim(0, 100)
    ax.set_ylabel('F1-score (%)')
    ax.set_title(f'Per-Class F1 Across Folds — {a["label"]} ({len(df)}-fold mean ± std)', fontweight='bold')
    ax.grid(axis='y', alpha=0.25)
    for bar, mean, std in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width() / 2, min(mean + std + 2.0, 98), f'{mean:.1f}±{std:.1f}',
                ha='center', va='bottom', fontsize=9, fontweight='bold')
    finish_figure(os.path.join(OUTPUT_DIR, f'per_class_f1_{a["key"]}.png'))


def export_json(analyses):
    out = {}
    for a in analyses:
        m, s, p = a['mean_row'], a['std_row'], a['pooled']
        out[a['key']] = {
            'dataset': a['label'],
            'n_folds': int(len(a['fold_df'])),
            'unique_test_epochs': int(len(a['all_labels'])),
            'per_fold_mean': {'accuracy_percent': float(m['Acc (%)']), 'macro_f1_percent': float(m['MF1 (%)']),
                              'weighted_f1_percent': float(m['WF1 (%)']), 'kappa': float(m['Kappa'])},
            'per_fold_std': {'accuracy_percent': float(s['Acc (%)']), 'macro_f1_percent': float(s['MF1 (%)']),
                             'weighted_f1_percent': float(s['WF1 (%)']), 'kappa': float(s['Kappa'])},
            'per_class_f1_mean_percent': {st: float(m[c]) for st, c in zip(STAGE_NAMES, F1_COLS)},
            'per_class_f1_std_percent': {st: float(s[c]) for st, c in zip(STAGE_NAMES, F1_COLS)},
            'pooled_metrics': {'accuracy_percent': float(p['accuracy'] * 100),
                               'macro_f1_percent': float(p['macro_f1'] * 100),
                               'weighted_f1_percent': float(p['weighted_f1'] * 100),
                               'kappa': float(p['kappa'])},
            'pooled_confusion_matrix_raw_counts': p['cm'].tolist(),
        }
    path = os.path.join(OUTPUT_DIR, 'summary_all_datasets.json')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2)
    print(f'  Saved: {path}')


# ═══════════════════════════════ MAIN ═══════════════════════════════
if __name__ == '__main__':
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print('=' * 96 + '\nSTEP 1 — De-duplication\n' + '=' * 96)
    for key in ['sleepedf78', 'sleepedf20', 'isruc_s3']:
        run_deduplication(key)

    print('\n' + '=' * 96 + '\nSTEP 2 — Analysis\n' + '=' * 96)
    analyses = []
    for key in ['sleepedf78', 'sleepedf20', 'isruc_s3']:
        a = summarize_dataset(key)
        print_summary(a)
        save_tables(a)
        plot_confusion_matrices(a)
        plot_per_class_f1(a)
        analyses.append(a)
    export_json(analyses)

    print('\n' + '=' * 96)
    print('SUMMARY (per-fold mean ± std)')
    print('=' * 96)
    for a in analyses:
        m, s = a['mean_row'], a['std_row']
        print(f'{a["label"]:13s} | Acc {m["Acc (%)"]:.2f} ± {s["Acc (%)"]:.2f} | '
              f'MF1 {m["MF1 (%)"]:.2f} ± {s["MF1 (%)"]:.2f} | κ {m["Kappa"]:.3f} ± {s["Kappa"]:.3f} | '
              + ' '.join(f'{st} {m[c]:.1f}' for st, c in zip(STAGE_NAMES, F1_COLS)))
    print(f'\nAll outputs saved to: {os.path.abspath(OUTPUT_DIR)}')