# AF-LeWM-lite PushT Experiments

This repository is a compact, reliability-checked PushT study for AF-LeWM-lite, a LeWM-style JEPA world model with an appearance-shaping branch.

The repo now focuses on one experiment family:

- `Baseline LeWM`
- `AF-LeWM-lite v1`: shared encoder, dynamics projection, appearance projection, invariance loss, independence penalty
- `AF-LeWM-lite v2`: v1 plus sequence-consistent nuisance shaping and a dynamics-side gradient-reversal nuisance head

Planning uses only the dynamics latent. The appearance branch is a training-time shaping signal.

## Model Pipeline

![AF-LeWM-lite model pipeline](report/aflewm_model_pipeline.png)

The diagram source is tracked at `report/diagrams/aflewm_model_pipeline.mmd`.

## Reliable Experiment Flow

![Reliable PushT experiment flow](report/pusht_experiment_flow.png)

The diagram source is tracked at `report/diagrams/pusht_experiment_flow.mmd`.

## What Was Fixed

![Implementation fix map](report/implementation_fix_map.png)

The reliable rerun is designed to avoid these experiment-validity failures:

- clip-level train/validation leakage from overlapping HDF5 windows
- full-dataset normalization leakage
- accidental continuation from stale checkpoints
- CEM planning over normalized actions outside valid environment bounds
- weak provenance for sampled evaluation rows and report inputs

## Repository Layout

```text
.
|- train.py                         # PushT training entrypoint
|- eval.py                          # PushT CEM planning evaluation
|- jepa.py                          # JEPA model with AF-LeWM-lite heads
|- module.py                        # Predictor, MLP, SIGReg helpers
|- run_all.py                       # PushT status/train/eval helper
|- validate_setup.py                # Environment, dataset, checkpoint validation
|- config/train/                    # Three matched PushT training configs
|- config/eval/                     # PushT evaluation config and CEM solver config
|- tools/                           # Dataset download and report generation
`- report/                          # PDF, TeX, diagrams, plots, CSV/JSON summary
```

## Installation

Python 3.10 is the expected runtime.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Data

The study uses the official PushT dataset:

```text
$STABLEWM_HOME/pusht_expert_train.h5
```

`STABLEWM_HOME` defaults to `~/.stable-wm`.

Download and extract:

```powershell
.\tools\download_official_datasets.ps1
python extract_datasets.py
```

Validate the local setup:

```bash
python validate_setup.py
python run_all.py --mode status --env pusht
```

## Run

Train all three matched PushT models:

```bash
python run_all.py --mode train --env pusht
```

Evaluate each epoch-10 checkpoint on two 50-start seeds:

```bash
python run_all.py --mode eval --env pusht
```

The reliable run names are:

```text
lewm_pusht_reliable
aflewm_pusht_v1_reliable
aflewm_pusht_v2_reliable
```

## Current Reliable Result

The April 22, 2026 rerun uses two 50-start evaluation seeds per model.

| Model | Seed 42 | Seed 43 | Aggregate | Wilson 95% CI |
| --- | ---: | ---: | ---: | ---: |
| Baseline LeWM | 2/50 | 4/50 | 6/100 = 6.0% | [2.78%, 12.48%] |
| AF-LeWM-lite v1 | 5/50 | 5/50 | 10/100 = 10.0% | [5.52%, 17.44%] |
| AF-LeWM-lite v2 | 1/50 | 4/50 | 5/100 = 5.0% | [2.15%, 11.18%] |

The evidence supports a reliability-checked exploratory ranking. The success counts are small and the intervals overlap.

## Report

Regenerate JSON, CSV, plots, diagrams, and the LaTeX source:

```bash
python tools/generate_pusht_official_report_assets.py
```

Build the PDF:

```bash
xelatex -interaction=nonstopmode -halt-on-error -output-directory=report report/pusht_aflewm_official_summary.tex
xelatex -interaction=nonstopmode -halt-on-error -output-directory=report report/pusht_aflewm_official_summary.tex
```

Primary report artifacts:

- `report/pusht_aflewm_official_summary.pdf`
- `report/pusht_aflewm_official_summary.tex`
- `report/pusht_official_budget_results.json`
- `report/pusht_official_budget_summary.csv`
- `report/pusht_success_rate.png`
- `report/pusht_val_core_loss.png`

## License

MIT. See `LICENSE`.
