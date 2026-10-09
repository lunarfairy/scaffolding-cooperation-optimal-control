# Scaffolding Cooperation: optimal-control theory

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23257433.svg)](https://doi.org/10.5281/zenodo.23257433)

Core research code for **Network memory separates recruiting from retaining ties to defectors in AI-mediated cooperation**.

The code implements a calibrated network cooperation game, agent-based evaluation of tie-recommendation policies, a differentiable individual-based pair approximation with action-history memory, and bounded optimal control with L-BFGS-B. It also includes the analysis of tie-state sorting in the public human experiment of McKee et al. (2023), an independent behavioural-model refit from all seven experimental conditions, and policy evaluation under that refit.

## Scope and contents

This repository contains computational methods and their required inputs. The manuscript, submission documents, plotting scripts, precomputed experiment outputs, and third-party human datasets are supplied separately. Simulation drivers generate their outputs when run. Models, parameters, seeds and optimisation logic match the approved manuscript code. The release copy of the fitting entrypoint additionally exports the same fitted coefficients to the simulator-format parameter file required by a downstream driver; this adds file output only. The two small fitted-model records under `results/` are necessary runtime inputs, rather than policy-performance outputs.

| File | Purpose |
| --- | --- |
| `src/abm_fast.py` | Vectorised agent-based model, policy families, interventions, and tie-state statistics |
| `src/sim_pilot.py` | Loop-based reference simulator |
| `src/ipa.py` | History-augmented pair approximation and reverse-mode differentiation |
| `src/run_abm_a.py` | Reference checks, trajectories, and two-phase screening sweeps |
| `src/run_abm_c.py` | Window screening and shared analysis helpers required by the later drivers |
| `src/run_ipa_b.py` | Reduced-model sweeps, accuracy calculations, and prerequisite coupled-threshold calculations |
| `src/run_rev_abm.py` | Revised experiments and fresh-game evaluation |
| `src/run_rev_ipa.py` | L-BFGS-B optimisation, switching functions, and diagnostics |
| `src/run_rev_abm2.py` | Agent-based evaluation of reduced-model controls |
| `fit_behaviour_model.py` | Pooled behavioural-model fitting and within-condition group bootstrap |
| `src/run_refit_abm.py` | Screening, fresh-game evaluation and one-step diagnostics under the independently fitted model |
| `human_data_tie_state.py` | Human tie-state analysis; requires the upstream public CSV files |

## Installation

The original analysis used Python 3.11 and PyTorch 2.0.1. Local release checks used Python 3.12.4, NumPy 1.26.4, SciPy 1.13.1, PyTorch 2.9.1, and pandas 3.0.2. CPU execution is supported; full reduced-model computations are substantially faster on a suitable GPU.

The dependency files pin the versions used for the local release checks. Use Python 3.12 for that environment and install the dependencies:

```sh
python -m pip install -r requirements.txt
```

For human-data analysis and behavioural-model fitting, also install pandas:

```sh
python -m pip install -r requirements-human.txt
```

PyTorch builds depend on the platform and CUDA installation; consult the [official installation instructions](https://pytorch.org/get-started/locally/) when choosing a GPU build.

## Required inputs and parameters

Five computational inputs are included:

- `results/behaviour_model_refit.json`: fitted coefficients, recommendation acceptance rates and bootstrap intervals consumed by `run_refit_abm.py`. It is the independent pooled fit reported by this manuscript: 19,502 decisions, 1,384 participants and 87 groups; estimated responsiveness 0.892366, with bootstrap interval 0.843437–0.937388.
- `results/refit_params.json`: the same coefficients in the simulator format consumed by the shared `run_rev_abm.py` driver. The fitting entrypoint exports this file after estimating the model.
- `src/oc_starts.json`: optimisation initialisation records required by the `window`, `tp` and `adam_old` starts.
- `src/confirm.json`: selected policy configurations used by final agent-based evaluation; the revision driver can regenerate it.
- `src/theta_draws.npy`: fixed disposition draws for paired accuracy calculations, regenerated with `ipa.draw_theta(64, 16, seed=12345)`.

The full fit records are included because downstream calculations read them at import time. Simulation outputs, screening grids and bootstrap replicate files are generated locally and excluded from version control. The previous release's companion-model inputs and `run_emp.py` have been superseded by the independent refit workflow in this version.

The baseline has 16 players, 15 rounds, initial edge probability 0.3, benefit 0.10, cost 0.05, and initial capital 1.0. Dispositions have mean -0.304 and standard deviation 2.410. The main reduced model uses action-history depth 3. Recommendation controls distinguish additions from deletions and cooperator-cooperator, cooperator-defector, and defector-defector pairs. See each module's docstring for its arguments and environment variables.

## Quick CPU run

Run simulation drivers **from `src/`** because their relative input and output paths are defined there:

```sh
cd src
python -c "import abm_fast as a; r=a.run_batched(a.typed(a.switch_u(15,14)),500,seed=2024,T=15,kappa=1.0); print('Final cooperation:',r['final'].mean())"
```

For a short optimisation run, use the existing controls on evaluation size. This checks execution, not convergence or paper-level precision.

Linux/macOS:

```sh
IPA_DEVICE=cpu IPA_G=1 IPA_THREADS=2 MAXFUN=1 python run_rev_ipa.py oc final 15 1.0 window 3
```

Windows PowerShell:

```powershell
$env:IPA_DEVICE = "cpu"
$env:IPA_G = "1"
$env:IPA_THREADS = "2"
$env:MAXFUN = "1"
python run_rev_ipa.py oc final 15 1.0 window 3
```

Before a full run, remove the short-run overrides. Full optimisation uses 32 disposition groups and a limit of 300 evaluations. `SMALL=1` and `IPA_SMALL=1` are existing reduced-size diagnostic modes; outputs produced in these modes should be kept separate from full-run outputs.

## Experiment order and generated dependencies

The complete experiments are computationally expensive. Set `NPROC` to the number of CPU workers available and `IPA_DEVICE` to `cpu` or a supported CUDA device. Defaults in the original drivers were chosen for a shared research server. Do not run the full sequence merely to check installation.

| Order | Command, run from `src/` | Required/generated files |
| --- | --- | --- |
| 1 | `python run_abm_a.py` | Generates `abm_a_results.json`, including two-phase screening sweeps |
| 2 | `python run_ipa_b.py acc sweep fb` | Generates accuracy results, `ipa_sweep.json`, and `ipa_fb.json`; the latter supplies coupled-threshold starts |
| 3 | `python run_abm_c.py` | Reads preceding outputs and generates `abm_c_results.json`, including the main window grids |
| 4 | `python run_rev_abm.py` | Requires **both** `abm_a_results.json` and `abm_c_results.json`; generates `rev_abm_results.json` and refreshed `confirm.json` |
| 5 | `python run_rev_ipa.py oc final 15 1.0 window 3` | Example of a converged-control run using the full defaults; generates an `oc_*_d3.json` record |
| 6 | `python run_rev_ipa.py costate` | Requires `ipa_sweep.json`; generates the tie-value calculations |
| 7 | `python run_rev_ipa.py fb2` | Uses `oc_starts.json`; include `ipa_fb.json` to retain the coupled-threshold initialisation |
| 8 | `python run_rev_ipa.py gradbases` | Requires `theta_draws.npy`; generates the gradients at validation bases |
| 9 | `python run_rev_ipa.py post` | Evaluates existing `oc_*_d*.json` records, including out-of-sample and Hessian checks |
| 10 | `python run_rev_abm2.py` | Requires generated `oc_*_d*.json`, `confirm.json`, and generated `ipa_fb2.json` for threshold-family comparisons |

Order 5 is one setting, not the full paper. To recreate the other reported settings, run the `oc` command with the corresponding objective (`final`, `capital`, or `mcoop`), horizon, responsiveness, initialisation (`half`, `enc`, `window`, `tp`, `adam_old`, `r1`, or `r2`), and depth. The configuration keys in `oc_starts.json` identify the supplied settings. The later diagnostic and agent-based drivers consume the optimisation records that are present in `src/`.

Some older computations in the shared prerequisite modules are retained because the current analysis imports their functions or consumes their generated files. Their earlier projected-Adam estimates are initialisation/background calculations; the current bounded optimisation is in `run_rev_ipa.py`.

## Public human-data analysis

The upstream data are available from [McKee et al.'s OSF project](https://osf.io/8ahkg/) for [Scaffolding cooperation in human groups with deep reinforcement learning](https://doi.org/10.1038/s41562-023-01686-7). Download and extract the project data, then place these six input files in `data/osf/` at the repository root:

```text
baseline_cooperation_data.csv
baseline_graph_struct_data.csv
evaluation_cooperation_data.csv
evaluation_graph_struct_data.csv
validation_cooperation_data.csv
validation_graph_struct_data.csv
```

Preserve these filenames. Run from the repository root:

```sh
python human_data_tie_state.py
```

The script writes `results/human_tie_state_gaps.csv`, `results/human_tie_state_gaps_round_stratified.csv`, and `results/human_tie_state_by_round.csv`. It uses participant decisions for outcomes and reconstructed bot decisions only for classifying pairs. Human input files are not redistributed in this repository; follow the upstream data's access and reuse terms. The software MIT licence does not relicense those data.

## Independent behavioural-model refit

The six upstream CSV files listed above also provide the inputs for the pooled fit. After installing both dependency files and obtaining those public CSVs, run from the repository root:

```sh
python fit_behaviour_model.py
python fit_behaviour_model.py boot 0 200
python fit_behaviour_model.py combine
```

The first command writes the point estimate to `results/behaviour_model_refit.json` and simulator-format coefficients to `results/refit_params.json`. Bootstrap commands write group-resampled fits under `results/boot/`; `combine` adds their intervals to the fit record. The 200-replicate bootstrap is a full scientific computation, not an installation check. It can be split into non-overlapping `boot START COUNT` blocks. The model uses 40-point Gauss–Hermite integration and L-BFGS-B, with the participant-disposition mean fixed at -0.304. Running the full fit is optional when using the supplied fitted-model inputs.

For the current paper's re-estimated-model simulations, run the following **from `src/`**:

```sh
python run_refit_abm.py work screen 0 1
python run_refit_abm.py combine screen
python run_refit_abm.py work fresh 0 1
python run_refit_abm.py combine fresh
```

These commands run the complete screen and independent fresh-game evaluation, including both endpoints of the responsiveness interval, and write JSON outputs to `results/refit_abm/`. They are expensive. To use multiple workers, replace `0 1` with worker index `I` and worker count `N`, run every `I = 0..N-1` for a stage, and combine only after all workers finish. Screening uses 30,000 games per policy; fresh evaluation uses 200,000. Screening must be combined before fresh evaluation because it generates `finalists.json`. These dedicated outputs provide the current re-estimated-model analyses; the shared revision drivers also retain prerequisite/background calculations.

A small execution check using the supplied fit, rather than the full screen, is:

```sh
python -c "import run_refit_abm as r; f,m,c=r.run('W1,12',r.K_HAT,500,2024); print('Refit cooperation:',f.mean())"
```

## Reproducibility and citation

Seeds and the separation of screening from fresh evaluation remain as specified in the source. The ABM maintains fixed-shape random draws for common-random-number comparisons. Reduced-model calculations use double precision and per-round checkpointing. Full main comparisons use 200,000 fresh games per policy and robustness comparisons use 100,000; short checks use fewer games.

Release validation checks syntax and imports, the unchanged published-model reference policies, small re-estimated-model simulations, finite pooled-model likelihood on a small dataset, and the coefficient-export mapping against the supplied fit. These are execution and file-consistency checks. The full 200-replicate fit, all simulation screens and optimisation starts are not rerun for this release.

Please cite the archived code release:

Lu, J., & Tu, C. (2026). *Scaffolding cooperation: optimal-control theory - core models and analysis* (v1.1.0). Zenodo. https://doi.org/10.5281/zenodo.23257433

The version DOI identifies the `v1.1.0` source archive. The earlier [v1.0.0 archive](https://doi.org/10.5281/zenodo.23194573) remains available; its files have not been replaced. Citation metadata are also provided in `CITATION.cff`. Please cite the associated manuscript separately by its title until its publication details are available. For questions, use the [repository issue tracker](https://github.com/lunarfairy/scaffolding-cooperation-optimal-control/issues).

## Licence

The software is released under the MIT licence; see `LICENSE`.
