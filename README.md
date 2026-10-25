# Mem-TTSAD: Memory-Augmented Transformer Models for Time-Series Anomaly Detection

Code for the M.Sc. research *Memory-Augmented Transformer Models for
Time-Series Anomaly Detection*. Two reconstruction-based Transformer backbones
from TranAD (Tuli et al., VLDB 2022) are each extended with three parametric
memory mechanisms adapted from the Memory Transformer (Burtsev et al., 2020).

## Models

| Backbone | Baseline | Direct Memory | MemCtrl | MemBottleneck |
|---|---|---|---|---|
| Basic Transformer (single phase) | `TranAD_Basic` | `Mem_TTSAD_Basic` | `MemCtrl_TTSAD_Basic` | `MemBottleneck_TTSAD_Basic` |
| TranAD (two-phase, self-conditioning) | `TranAD` | `Mem_TTSAD` | `MemCtrl_TTSAD` | `MemBottleneck_TTSAD` |

- **Direct Memory** (`Mem_*`): *m* learnable memory tokens prepended to the encoder input; the decoder reads `[Hmem; Hseq]`.
- **MemCtrl**: memory and sequence streams updated in parallel from `[Xmem; Xseq]` by two separate memory layers.
- **MemBottleneck**: the memory attends to `[Xmem; Xseq]`; the sequence then attends only to the updated memory.

All models reconstruct the last observation of a window of 10 time steps and
score anomalies by the squared reconstruction error (TranAD models: phase-2 output).

## Installation

```bash
git clone <repository-url> MEM-TTSAD
cd MEM-TTSAD
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m unittest discover tests        # quick check, no data needed
```

Python 3.10+, CPU, float64. Results were produced with torch 2.2.1, numpy 1.26.4,
pandas 3.0.3, scikit-learn 1.9.0, scipy 1.11.4 (use NumPy 1.x).

## Data

Place the raw datasets under `data/<DATASET>/` (SMD, SMAP, SWaT, WADI) and run

```bash
python preprocess.py SMD SMAP SWaT WADI
```

This writes `processed/<DATASET>/`, which `main.py` reads from the working directory.

## Training and evaluation

`main.py` trains one model on one dataset, saves the checkpoint, then evaluates
it and prints the per-dimension and aggregate metrics (POT threshold,
point-adjusted and raw precision / recall / F1, ROC-AUC, PR-AUC).

```bash
# Baselines
TRANAD_SEED=1 TRANAD_EPOCHS=5 python main.py --model TranAD       --dataset SMD --retrain
TRANAD_SEED=1 TRANAD_EPOCHS=5 python main.py --model TranAD_Basic --dataset SMD --retrain

# Memory-augmented models: MEMTRANAD_M is the memory size m (1-5 in the research)
MEMTRANAD_M=3 TRANAD_SEED=1 TRANAD_EPOCHS=5 python main.py --model Mem_TTSAD                 --dataset WADI --retrain
MEMTRANAD_M=3 TRANAD_SEED=1 TRANAD_EPOCHS=5 python main.py --model MemCtrl_TTSAD             --dataset SMD  --retrain
MEMTRANAD_M=3 TRANAD_SEED=1 TRANAD_EPOCHS=5 python main.py --model MemBottleneck_TTSAD       --dataset SWaT --retrain
MEMTRANAD_M=3 TRANAD_SEED=1 TRANAD_EPOCHS=5 python main.py --model Mem_TTSAD_Basic           --dataset SMAP --retrain
MEMTRANAD_M=3 TRANAD_SEED=1 TRANAD_EPOCHS=5 python main.py --model MemCtrl_TTSAD_Basic       --dataset SMD  --retrain
MEMTRANAD_M=3 TRANAD_SEED=1 TRANAD_EPOCHS=5 python main.py --model MemBottleneck_TTSAD_Basic --dataset SMD  --retrain

# Evaluate a saved checkpoint only (same settings select the checkpoint)
MEMTRANAD_M=3 TRANAD_SEED=1 TRANAD_EPOCHS=5 python main.py --model MemCtrl_TTSAD --dataset SMD --test

# Another entity of a multi-entity dataset
TRANAD_ENTITY=machine-2-1 MEMTRANAD_M=2 python main.py --model MemCtrl_TTSAD --dataset SMD --retrain
```

Full research grid (8 models, 4 datasets, m = 1..5, seed 1, 5 epochs):

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

Outputs: `checkpoints/<run>/model.ckpt` (memory models:
`<model>_<dataset>_<entity>_m<m>_seed<S>_e<E>_b128`; baselines:
`<model>_<dataset>`) and `plots/<model>_<dataset>/`. Set `MPLBACKEND=Agg`
on headless machines.

## Settings

| Setting | How to set | Default |
|---|---|---|
| Model / dataset | `--model`, `--dataset` | `TranAD`, `SMD` |
| Mode | `--retrain` (train from scratch) / `--test` (evaluate checkpoint) | load if present, else train |
| Memory size *m* | `MEMTRANAD_M` | 4 for `Mem_TTSAD`, 2 for other memory models |
| Seed | `TRANAD_SEED` | 1 |
| Epochs | `TRANAD_EPOCHS` | 5 |
| Entity | `TRANAD_ENTITY` | SMD `machine-1-1`, SMAP `P-1` |
| Batch / window | fixed | 128 / 10 |
| Optimiser | fixed | AdamW (lr per dataset in `memttsad/config.py`), StepLR(5, 0.9) |

Default-off research switches (see `python main.py --help`): memory-slot
diversity regularisation (`--memory_diversity updated --diversity_lambda_updated 0.01`),
`--memory_init_norm`, and the `MemCtrl_TTSAD_Basic` decoder / retrieval
variants. Runs are deterministic for a given seed and settings (fixed
construction order, no data shuffling, no test-label use during training).

## Citation

```bibtex
@mastersthesis{hussien2026memttsad,
  author = {Talal Hussien},
  title  = {Memory-Augmented Transformer Models for Time-Series Anomaly Detection},
  school = {Ben-Gurion University of the Negev},
  year   = {2026}
}
```

## License

BSD 3-Clause (see `LICENSE`, which also lists the third-party code: TranAD,
SPOT and OmniAnomaly's point adjustment).
