# Mem-TTSAD: Memory-Augmented Transformer Models for Time-Series Anomaly Detection

Code for the M.Sc. research *Memory-Augmented Transformer Models for
Time-Series Anomaly Detection*. Two reconstruction-based Transformer backbones
from TranAD (Tuli et al., VLDB 2022) are each extended with three parametric
memory mechanisms adapted from the Memory Transformer (Burtsev et al., 2020):
**Direct Memory**, **MemCtrl** and **MemBottleneck**. The repository contains
the exact implementation used for the reported results, together with the
data preprocessing, training, inference, anomaly scoring and evaluation code.

## Supported architectures

| Backbone | Baseline | Direct Memory | MemCtrl | MemBottleneck |
|---|---|---|---|---|
| Basic Transformer (single phase) | `TranAD_Basic` | `Mem_TTSAD_Basic` | `MemCtrl_TTSAD_Basic` | `MemBottleneck_TTSAD_Basic` |
| TranAD (two-phase, self-conditioning) | `TranAD` | `Mem_TTSAD` | `MemCtrl_TTSAD` | `MemBottleneck_TTSAD` |

- **Direct Memory** (`Mem_*`): *m* learnable memory tokens are prepended to the
  encoder input; the decoder reads the full encoder output `[Hmem; Hseq]`.
- **MemCtrl** (Burtsev et al., Sec. 2.3): memory and sequence streams are
  updated *in parallel* from `[Xmem; Xseq]` by two separate sub-networks
  (`MemoryLayer`); the decoder reads `[Hmem; Hseq]`.
- **MemBottleneck** (Burtsev et al., Sec. 2.4): the memory attends to
  `[Xmem; Xseq]`, then the sequence attends *only* to the updated memory
  (no sequence-to-sequence path); the decoder reads `[Hmem; Hseq]`.

All models reconstruct the last observation of a sliding window of length 10
and use the per-dimension squared reconstruction error as anomaly score.
The TranAD-based models use the two-phase objective and score with the
phase-2 reconstruction. The memory models add learnable slots and (for
MemCtrl / MemBottleneck) two post-LN memory layers, but leave decoders,
output head, objective, scoring and thresholding of the backbone unchanged.

## Repository layout

```
main.py                        entry point: train and/or evaluate one model on one dataset
preprocess.py                  raw data -> processed/<DATASET>/{[entity_]train,test,labels}.npy
preprocess_full.py             optional entity-complete SMD_FULL / SMAP_FULL / MSL_FULL datasets
requirements.txt
memttsad/
├── cli.py                     argparse flags (bridged into MEMTRANAD_* environment variables)
├── config.py                  per-dataset learning rates and POT settings (from TranAD)
├── data.py                    dataset loading, entity selection, sliding windows
├── training.py                TranAD-protocol training/evaluation loop, checkpoints
├── regularization.py          memory-slot diversity regularisation (default off)
├── plotting.py                training-curve and per-dimension output plots
├── models/
│   ├── __init__.py            seeding from TRANAD_SEED, model registry, build_model()
│   ├── layers.py              PositionalEncoding, Transformer encoder/decoder layers, MemoryLayer
│   ├── backbones.py           TranAD_Basic, TranAD
│   ├── memory_basic.py        Mem_TTSAD_Basic, MemCtrl_TTSAD_Basic, MemBottleneck_TTSAD_Basic
│   ├── memory_tranad.py       Mem_TTSAD, MemCtrl_TTSAD, MemBottleneck_TTSAD
│   └── variants.py            default-off experiment switches (environment driven)
└── evaluation/
    ├── pot.py                 POT thresholding, point adjustment, F1 / AUC metrics
    ├── spot.py                SPOT streaming peaks-over-threshold (Siffer et al., 2017)
    └── metrics.py             Hit@P% and NDCG@P% diagnostics
tests/                         unit tests of the memory mechanisms and switches (CPU, seconds)
```

## Installation

Python 3.10+ (3.11 and 3.12 were used). The pipeline runs on CPU in float64.

```bash
git clone <repository-url> MEM-TTSAD
cd MEM-TTSAD
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Versions used for the reported results: torch 2.2.1 (CPU build), numpy 1.26.4,
pandas 3.0.3, scikit-learn 1.9.0, scipy 1.11.4, matplotlib 3.11.0,
tqdm 4.68.3, openpyxl 3.1.5. NumPy 1.x is recommended: the WADI preprocessing
uses `ndarray.ptp`, which NumPy 2 removed.

Quick check of the installation (no data required):

```bash
python -m unittest discover tests -v
```

## Dataset preparation

The datasets are not distributed with this repository. Obtain them from
their owners, place the raw files under `data/<DATASET>/` as listed below,
and run `preprocess.py`. Preprocessing writes `processed/<DATASET>/`, which
`main.py` reads from the current working directory.

| Dataset | Source | Raw files expected |
|---|---|---|
| **SMD** | Server Machine Dataset, released with OmniAnomaly (Su et al., KDD 2019): <https://github.com/NetManAIOps/OmniAnomaly> | `data/SMD/train/machine-*.txt`, `data/SMD/test/machine-*.txt`, `data/SMD/interpretation_label/machine-*.txt` |
| **SMAP** | NASA SMAP telemetry released with Telemanom (Hundman et al., KDD 2018): <https://github.com/khundman/telemanom> | `data/SMAP_MSL/labeled_anomalies.csv`, `data/SMAP_MSL/train/<channel>.npy`, `data/SMAP_MSL/test/<channel>.npy` |
| **SWaT** | Secure Water Treatment testbed, iTrust (SUTD); request access at <https://itrust.sutd.edu.sg/itrust-labs_datasets/> | `data/SWaT/SWaT_Dataset_Normal_v1.xlsx`, `data/SWaT/SWaT_Dataset_Attack_v0.xlsx` (Dec 2015 release) |
| **WADI** | Water Distribution testbed, iTrust (SUTD); same request form | `data/WADI/WADI_14days.csv`, `data/WADI/WADI_attackdata.csv`, `data/WADI/WADI_attacklabels.csv` |

```bash
python preprocess.py SMD SMAP SWaT WADI      # any subset; one argument per dataset
```

What the preprocessing does (inherited from the TranAD pipeline; see the
comments in `preprocess.py`):

- **SMD**: one entity per machine file, values kept as shipped (already in
  [0, 1]); per-dimension labels built from `interpretation_label`. Default
  entity `machine-1-1`.
- **SMAP**: one entity per channel listed in `labeled_anomalies.csv`;
  per-channel min-max scaling fitted on the training series and applied to
  the test series; anomaly sequences expanded to per-timestamp labels.
  Default entity `P-1`.
- **SWaT**: the official 51-channel Normal_v1 (train) and Attack_v0 (test)
  files; the first 21,600 s of normal operation are discarded, every 10th row
  is kept, scaling uses training statistics only and the test series is
  clipped to [0, 1]; the `Normal/Attack` column gives the labels.
- **WADI**: 200,000 rows of `WADI_14days.csv` after the first 1,000 (train)
  and `WADI_attackdata.csv` (test); attack intervals from
  `WADI_attacklabels.csv` are mapped onto the affected channels; every 10th
  row is kept and each channel is min-max scaled.

`preprocess.py` also supports the other datasets of the TranAD code base
(`MSL`, `MSDS`, `UCR`, `NAB`, `MBA`, `PSM`, `synthetic`, ...); they were not
used in this research. `preprocess_full.py` builds optional entity-complete
`SMD_FULL` / `SMAP_FULL` / `MSL_FULL` variants with a manifest.

## Training

`main.py` trains one model on one dataset and evaluates it at the end.
Model and dataset are command-line flags; the reproducibility settings are
environment variables (the convention of the TranAD code base, convenient for
shell runners).

```bash
# Baselines
TRANAD_SEED=1 TRANAD_EPOCHS=5 python main.py --model TranAD       --dataset SMD  --retrain
TRANAD_SEED=1 TRANAD_EPOCHS=5 python main.py --model TranAD_Basic --dataset SMD  --retrain

# Memory-augmented models; MEMTRANAD_M is the memory size m (1, 2, 3, 4, 5 in the research)
MEMTRANAD_M=3 TRANAD_SEED=1 TRANAD_EPOCHS=5 python main.py --model Mem_TTSAD                 --dataset WADI --retrain
MEMTRANAD_M=3 TRANAD_SEED=1 TRANAD_EPOCHS=5 python main.py --model MemCtrl_TTSAD             --dataset SMD  --retrain
MEMTRANAD_M=3 TRANAD_SEED=1 TRANAD_EPOCHS=5 python main.py --model MemBottleneck_TTSAD       --dataset SWaT --retrain
MEMTRANAD_M=3 TRANAD_SEED=1 TRANAD_EPOCHS=5 python main.py --model Mem_TTSAD_Basic           --dataset SMAP --retrain
MEMTRANAD_M=3 TRANAD_SEED=1 TRANAD_EPOCHS=5 python main.py --model MemCtrl_TTSAD_Basic       --dataset SMD  --retrain
MEMTRANAD_M=3 TRANAD_SEED=1 TRANAD_EPOCHS=5 python main.py --model MemBottleneck_TTSAD_Basic --dataset SMD  --retrain

# Another entity of a multi-entity dataset
TRANAD_ENTITY=machine-2-1 MEMTRANAD_M=2 python main.py --model MemCtrl_TTSAD --dataset SMD --retrain
```

The full research grid is the 8 models above on the 4 datasets with
`MEMTRANAD_M` in `{1,2,3,4,5}` for the six memory models, `TRANAD_SEED=1`,
`TRANAD_EPOCHS=5`:

```bash
for ds in SMD SMAP SWaT WADI; do
  for model in TranAD TranAD_Basic; do
    TRANAD_SEED=1 TRANAD_EPOCHS=5 python main.py --model $model --dataset $ds --retrain
  done
  for model in Mem_TTSAD MemCtrl_TTSAD MemBottleneck_TTSAD Mem_TTSAD_Basic MemCtrl_TTSAD_Basic MemBottleneck_TTSAD_Basic; do
    for m in 1 2 3 4 5; do
      MEMTRANAD_M=$m TRANAD_SEED=1 TRANAD_EPOCHS=5 python main.py --model $model --dataset $ds --retrain
    done
  done
done
```

Training writes `checkpoints/<run>/model.ckpt` (weights, optimizer and
scheduler state, per-epoch training loss), `plots/<model>_<dataset>/training-graph.pdf`
and `plots/<model>_<dataset>/output.pdf` (per-dimension series, reconstruction,
labels and anomaly score), then runs the evaluation below. The checkpoint
directory encodes every setting of the run so that runs never overwrite
each other:

```
checkpoints/<model>_<dataset>[-<entity>]                                       backbones (TranAD, TranAD_Basic)
checkpoints/<model>_<dataset>_<entity>_m<m>_seed<seed>_e<epochs>_b128[...]     memory models
```

Set `MPLBACKEND=Agg` on headless machines.

## Evaluation and inference

`--test` loads the checkpoint that matches the current model, dataset and
settings, skips training and runs inference and evaluation only. The same
environment variables select the checkpoint:

```bash
MEMTRANAD_M=3 TRANAD_SEED=1 TRANAD_EPOCHS=5 python main.py --model MemCtrl_TTSAD --dataset SMD --test
python main.py --model TranAD --dataset SMD --test
```

Evaluation protocol (`main.py`, `memttsad/evaluation/`):

1. **Anomaly scores.** Inference over the whole training and test series;
   the score of a timestamp is the squared reconstruction error of the
   window's last observation, per dimension (TranAD models: phase-2 output).
   Windows are built from the 10 preceding rows, left-padded at the start of
   the series; as in the TranAD code base, the score and label sequences are
   paired at an offset of one time step, identically for every model.
2. **Thresholding.** Peaks-over-threshold (POT, `SPOT` with risk
   `q = 1e-5`): the threshold is initialised on the training scores with a
   dataset-specific level and multiplied by a dataset-specific factor
   (`memttsad/config.py`, values from TranAD: SMD 0.99995 / 1.06,
   SMAP 0.98 / 1, SWaT 0.993 / 1, WADI 0.999 / 1). Test labels are never
   used for threshold selection.
3. **Metrics.** Per dimension and for the mean score over dimensions:
   precision, recall and F1 **with point adjustment** (`f1`, `precision`,
   `recall`; OmniAnomaly protocol: once any point of an anomalous segment is
   detected, the whole segment counts as detected), the unadjusted
   point-wise `f1_unadjusted` / `precision_unadjusted` / `recall_unadjusted`,
   the threshold-free `roc_auc_raw` and `pr_auc_raw`, and the Hit@P% / NDCG@P%
   ranking diagnostics. The per-dimension table and the aggregate dictionary
   are printed at the end of the run.

## Configuration reference

| Setting | How to set | Default | Meaning |
|---|---|---|---|
| Model | `--model` | `TranAD` | one of the 8 architectures above |
| Dataset | `--dataset` | `SMD` | `SMD`, `SMAP`, `SWaT`, `WADI` (plus the other TranAD datasets) |
| Mode | `--retrain` / `--test` | load-or-train | `--retrain` trains from scratch; `--test` evaluates a saved checkpoint |
| Memory size *m* | `MEMTRANAD_M` | 4 for `Mem_TTSAD`, 2 for the other memory models | number of learnable memory slots |
| Seed | `TRANAD_SEED` | 1 | torch seed, applied when `memttsad.models` is imported, before model construction |
| Epochs | `TRANAD_EPOCHS` | 5 | training epochs |
| Entity | `TRANAD_ENTITY` | SMD `machine-1-1`, SMAP `P-1` | entity of a multi-entity dataset |
| Batch size | fixed | 128 | `model.batch` |
| Window size | fixed | 10 | `model.n_window` |
| Optimiser | fixed | AdamW, weight decay 1e-5, StepLR(5, 0.9) | learning rate per dataset in `memttsad/config.py` |
| Reduced data | `--less` | off | train on the central 20% of the training series (TranAD option) |

Default-off research switches (each is a no-op unless set; see
`python main.py --help`): memory-slot diversity regularisation
(`--memory_diversity {raw,updated,both}` with `--diversity_lambda`,
`--diversity_lambda_raw`, `--diversity_lambda_updated`; environment
`MEMTRANAD_DIVERSITY`, `MEMTRANAD_DIV_LAMBDA*`), memory initialisation scale
(`--memory_init_norm`), and the `MemCtrl_TTSAD_Basic` decoder/retrieval
variants (`--decoder_query`, `--decoder_keys`, `--history_only`,
`--no_memory`, `--memory_retrieval`, `--retrieval_dim`, `--retrieval_context`).
When a switch is on, its value is appended to the checkpoint directory name.

Example with the diversity regulariser used in the research
(penalty on the updated memory read by the decoder):

```bash
MEMTRANAD_M=3 TRANAD_SEED=1 TRANAD_EPOCHS=5 python main.py --model MemCtrl_TTSAD --dataset SMD --retrain \
    --memory_diversity updated --diversity_lambda_updated 0.01
```

## Reproducibility

- Computation is CPU, float64. Models are constructed in a fixed order from
  the torch RNG seeded by `TRANAD_SEED`; training data are not shuffled and
  the test labels are never consulted during training (no early stopping, no
  checkpoint selection). A run is therefore determined by its seed, memory
  size, epochs, dataset and entity. Runs with identical settings reproduce
  bit-identically on the same software versions.
- Set `OMP_NUM_THREADS` / `MKL_NUM_THREADS` to control CPU threads; the
  research runs used 2 threads per job.
- `tests/` exercises the memory mechanisms, the regulariser and the
  default-off switches on small random inputs and runs in seconds on CPU:

```bash
python -m unittest discover tests -v
```

## Citation

If you use this code, please cite the thesis and the works it builds on.

```bibtex
@mastersthesis{hussien2026memttsad,
  author = {Talal Hussien},
  title  = {Memory-Augmented Transformer Models for Time-Series Anomaly Detection},
  school = {Ben-Gurion University of the Negev},
  year   = {2026}
}
@article{tuli2022tranad,
  author  = {Tuli, Shreshth and Casale, Giuliano and Jennings, Nicholas R.},
  title   = {{TranAD}: Deep Transformer Networks for Anomaly Detection in Multivariate Time Series Data},
  journal = {Proceedings of the VLDB Endowment},
  volume  = {15}, number = {6}, pages = {1201--1214}, year = {2022}
}
@article{burtsev2020memory,
  author  = {Burtsev, Mikhail S. and Kuratov, Yuri and Peganov, Anton and Sapunov, Grigory V.},
  title   = {Memory Transformer},
  journal = {arXiv preprint arXiv:2006.11527},
  year    = {2020}
}
```

## License and acknowledgements

Released under the BSD 3-Clause License (see `LICENSE`). The backbones,
training protocol, preprocessing and evaluation derive from the TranAD
reference implementation (BSD 3-Clause); `memttsad/evaluation/spot.py` is the
SPOT implementation by Alban Siffer (GPLv3) and `adjust_predicts` comes from
OmniAnomaly (MIT). See the notices in `LICENSE`.
