# SleepNeXt

**SleepNeXt: A Hybrid Sleep Stage Classification Framework with Improved Minority-Stage Detection Using Multi-Channel Polysomnography**

## Overview

SleepNeXt classifies 30-s polysomnography (PSG) epochs into five sleep stages (W, N1, N2, N3, REM). It combines three parts:

- **Epoch encoder:** a 1-D ConvNeXt V2 encoder with a dual-kernel front end (kernel sizes 5 and 51) and Global Response Normalization (GRN).
- **Attention:** coordinate attention on the encoder output.
- **Context model:** a two-layer BiLSTM over sequences of 20 epochs.

Training uses focal loss with label smoothing and class weights. The model has about 4.5 M parameters.

It is evaluated with subject-wise K-fold cross-validation (K − 1 folds for training, 1 fold for testing):

| Dataset | Channels | Folds |
|---|---|---|
| Sleep-EDF-78 | Fpz-Cz, Pz-Oz, EOG | 10 |
| Sleep-EDF-20 | Fpz-Cz, Pz-Oz, EOG | 20 |
| ISRUC-S3 | 6 EEG, 2 EOG, chin EMG, ECG, leg EMG (11) | 10 |

## Files

| File | Purpose |
|---|---|
| `Preprocessing_SleepEDF.py` | Reads the Sleep-EDF (sleep-cassette) EDF files and keeps the 3 channels. Trims wake to 30 min before and after sleep, merges S3 and S4 into N3, and applies 5th–95th percentile normalization to [−1, 1]. Saves `X_raw.npy`, `y.npy`, `subject_ids.npy` (recording IDs) and `true_subject_ids.npy` (subject IDs 0–77). Also finds the 75 unnormalized epochs that are excluded during training. |
| `Preprocessing_ISRUC-S3.py` | Reads the ISRUC-S3 `.mat` files and expert-1 labels. Decimates the 11 channels from 200 Hz to 100 Hz and applies percentile normalization. Saves `X_raw_isruc_phase2_11ch.npy`, `y_isruc_phase2_11ch.npy` and `subject_ids_isruc_phase2_11ch.npy` (8,589 epochs). |
| `SleepNeXt_SleepEDF.py` | Model definition and training for one fold of Sleep-EDF-78 or Sleep-EDF-20 (35 epochs, resumable). Writes `fold_k_SleepNeXt_<dataset>_results.pkl`. |
| `SleepNeXt_ISRUC-S3.py` | The same model and training procedure for one fold of ISRUC-S3 with 11 input channels. Writes `fold_k_SleepNeXt_isruc_s3_phase2_11ch_results.pkl`. |
| `SleepNeXt_Result_Analysis.py` | Removes duplicate predictions from the overlapping test sequences so that each epoch is scored once. Then reports per-fold and mean ± std metrics (Acc, MF1, κ, per-class F1), confusion matrices, plots and tables for all three datasets. |

## How to run

1. **Install the dependencies:**
   ```
   pip install torch timm numpy scipy scikit-learn mne pandas matplotlib seaborn tqdm
   ```
2. **Preprocess.** Run `Preprocessing_SleepEDF.py` and `Preprocessing_ISRUC-S3.py`. First set the dataset paths (`BASE`, `MAT_BASE`, `LBL_BASE`) and the output folder (`SAVE`).
3. **Train.** Run `SleepNeXt_SleepEDF.py` once per fold:
   - set `fold_idx = 0 … K−1`;
   - set `dataset_name = 'sleepedf78'` or `'sleepedf20'`.

   Then run `SleepNeXt_ISRUC-S3.py` for `fold_idx = 0 … 9`.
4. **Analyse.** Copy the outputs into this layout and run `python SleepNeXt_Result_Analysis.py`:
   ```
   results/
   ├── sleepedf/   y.npy, subject_ids.npy, true_subject_ids.npy,
   │               fold_*_SleepNeXt_sleepedf78_results.pkl (10),
   │               fold_*_SleepNeXt_sleepedf20_results.pkl (20)
   └── isruc_s3/   y_isruc_phase2_11ch.npy, subject_ids_isruc_phase2_11ch.npy,
                   fold_*_SleepNeXt_isruc_s3_phase2_11ch_results.pkl (10)
   ```
   The analysis writes its outputs to `results/analysis_outputs/`.

**Use only the analysis script for reported numbers.** The raw `*_results.pkl` files contain a few duplicate predictions and should not be used to report results directly.
