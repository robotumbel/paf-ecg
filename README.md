# Patient-Aggregated Fusion (PAF) for patient-independent MI detection

Code, configuration, random seeds, split identifiers and result tables for the
manuscript *Patient-Aggregated Fusion (PAF): Robust Patient-Independent
Myocardial Infarction Detection from 12-Lead ECG* (Computers in Biology and
Medicine, manuscript CIBM-D-26-06413, revision 1).

## Data

Both databases are public on PhysioNet and are not redistributed here.

- PTB-XL 1.0.3: <https://physionet.org/content/ptb-xl/1.0.3/>. Place it so that
  `ptbxl/ptbxl/ptbxl_database.csv` exists relative to `src/`, or set
  `PTBXL_DIR`. Only the 100 Hz files (`records100`) are used.
- PTB Diagnostic ECG Database 1.0.0:
  <https://physionet.org/content/ptbdb/1.0.0/>, placed in `src/ptbdb/`.

## Environment

- Preprocessing: Python 3.12, `wfdb` 4.3, `PyWavelets` 1.9, NumPy, pandas,
  scikit-learn (`requirements-preprocess.txt`).
- Training: Python 3.11, TensorFlow 2.21, Keras 3.15 (`requirements-train.txt`).
  The experiments ran on one NVIDIA RTX 4050 (6 GB). `run_gpu.sh` puts the CUDA
  libraries on the path under WSL2.

## Pipeline (run from `src/`)

| Step | Script | Output |
|---|---|---|
| Preprocess PTB-XL (corrected R-peak detector, `fs=100`) | `preprocess_fusion_fix.py` | `data_fusion_fix/all_{X,y,groups}.npy` |
| Preprocess PTB Diagnostic | `preprocess_ptbdb.py` | `data_ptbdb/` |
| Six backbones x three pooling rules, five random patient-disjoint splits | `PAF_DATA=data_fusion_fix PAF_OUT=results/paf_bench_fix ./run_gpu.sh train_paf_bench.py` | `results/paf_bench_fix/` |
| Attention-MIL pooling, same splits | `./run_gpu.sh train_attention_pool.py` | `results/attention_pool/` |
| Four-territory task, mean vs attention | `./run_gpu.sh train_territory.py` | `results/territory/` |
| Official PTB-XL folds (train 1-8, val 9, test 10) | `run_revision.sh` (`PAF_SPLIT=official`) | `results/paf_official/`, `results/attention_official/` |
| External validation | `PAF_DATA=data_fusion_fix EXT_OUT=results/external_fix ./run_gpu.sh eval_external.py` | `results/external_fix/` |
| Beats-per-patient saturation | `beats_scaling.py` | `results/beats_scaling/` |
| Statistics (CIs, McNemar, Wilcoxon) | `stats_paf_bench.py` | tables in `results/paf_bench_fix/` |
| Hyperparameter search | `tune_hparams.py` | `results/hparam_search/` |
| Grad-CAM quantification, faithfulness, sanity, stability | `run_xai.sh` | `results/gradcam_quant_fix/`, `results/xai_faithfulness/` |
| Split support and patient IDs | `split_support.py` | `splits/` |
| Per-class tables and confusion matrices | `make_revision_tables.py` | `results/revision/` |

## Seeds and splits

- Random patient-disjoint splits: seeds 45, 46, 47, 48, 49
  (`StratifiedGroupKFold`, 7 folds, grouped on PTB-XL `patient_id`, applied
  twice for test and validation).
- Official folds: PTB-XL `strat_fold`; seeds 45-47 vary only the initialisation.
- `splits/patients_{task}_{split}.csv` lists every `patient_id` with its
  partition, and `splits/split_support.csv` gives per-class counts.

## Note on replicated records

The records of six rare MI subtypes (LMI, AMI, ALMI, IPLMI, IPMI, PMI) are
replicated ten times before beat extraction (`OVERSAMPLE` in
`preprocess_ptbxl_all_leads.py`). Copies stay within a patient and therefore
within one partition; patient-level mean-posterior and majority-vote decisions
are unchanged by exact copies. The dataset has 125,520 beat windows, 96,189 of
them unique.

## Citation

If you use this code, please cite the article above (citation details will be
added on publication).

## Licence

MIT, see `LICENSE`.
