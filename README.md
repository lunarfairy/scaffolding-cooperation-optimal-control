# Network memory and cooperation

Code for **Network memory separates recruiting from retaining ties to defectors in AI-mediated cooperation**: an agent-based network game, history-augmented pair approximation, optimal control, and human-data analysis.

## Generate Figs. 1–6

From the repository root, using Python 3.12:

```sh
python -m pip install -r requirements-figures.txt
python figures/make_all_figures.py
```

This uses the manuscript's saved aggregate inputs in `figure_data/` and writes six PNG/PDF pairs to `figures/`. Individual figures can be generated with `python figures/make_figN.py` (N = 1…6).

## Recompute figure inputs

Install the computational dependencies:

```sh
python -m pip install -r requirements.txt -r requirements-human.txt -r requirements-figures.txt
```

Set `NPROC` to the available CPU worker count and `IPA_DEVICE` to `cpu` or `cuda`. Use the full defaults for sample sizes and optimisation; clear any previous `SMALL`, `IPA_SMALL`, `IPA_G` or `MAXFUN` overrides. Full experiments require substantial computing time.

Run these commands in order:

```sh
cd src
python run_abm_a.py
python run_ipa_b.py acc sweep fb
python run_abm_c.py
python run_rev_abm.py
cd ..
python run_optimisations.py
cd src
python run_rev_ipa.py costate
python run_rev_ipa.py fb2
python run_rev_abm2.py
python run_refit_abm.py work screen 0 1
python run_refit_abm.py combine screen
python run_refit_abm.py work fresh 0 1
python run_refit_abm.py combine fresh
cd ..
python human_data_tie_state.py
python prepare_figures.py
python figures/make_all_figures.py
```

`run_optimisations.py` runs the settings in `optimisation_tasks.json`. `run_rev_abm.py` refreshes `src/confirm.json`; `prepare_figures.py` assembles simulation outputs and human-data summaries into `figure_data/`. Use `--output PATH` to retain the supplied figure inputs. Optimisation records are selected by their reduced-model objective, before their ABM performance is compared.

Optional gradient and optimality diagnostics: run `python run_rev_ipa.py gradbases` and `python run_rev_ipa.py post` from `src/` after optimisation.

Reduced-size runs of `run_rev_ipa.py` (`IPA_SMALL=1`, `IPA_G<32` or `MAXFUN<300`) write to `src/quick_results/`. Other drivers retain their documented output paths. Seeds are specified in the source; numerical results can vary with the computational environment.

## Human data and behavioural fit

Download the data from [McKee et al.'s OSF project](https://osf.io/8ahkg/), accompanying [Scaffolding cooperation in human groups with deep reinforcement learning](https://doi.org/10.1038/s41562-023-01686-7). Put these files in `data/osf/`:

```text
baseline_cooperation_data.csv
baseline_graph_struct_data.csv
evaluation_cooperation_data.csv
evaluation_graph_struct_data.csv
validation_cooperation_data.csv
validation_graph_struct_data.csv
```

The simulation sequence uses the supplied pooled-fit parameters in `results/behaviour_model_refit.json` and `results/refit_params.json`. To re-estimate them, run the following from the repository root **before** the simulation sequence:

```sh
python fit_behaviour_model.py
python fit_behaviour_model.py boot 0 200
python fit_behaviour_model.py combine
```

The bootstrap resamples groups within conditions. Refit simulations use the point estimate and bootstrap interval endpoints from this record. Human input data remain subject to their upstream licence; the software licence does not relicense those data.

## Contents

| Path | Contents |
| --- | --- |
| `src/abm_fast.py`, `src/sim_pilot.py` | Vectorised and reference simulators |
| `src/ipa.py` | Differentiable history-augmented pair approximation |
| `src/run_*.py` | Simulation, control and diagnostic drivers |
| `src/oc_starts.json`, `src/theta_draws.npy` | Optimisation starts and paired disposition draws |
| `fit_behaviour_model.py`, `human_data_tie_state.py` | Behavioural fit and human tie-state analysis |
| `prepare_figures.py`, `figures/` | Figure-input assembly and plotting |
| `figure_data/` | Saved aggregate inputs for Figs. 1–6 |

## Citation and licence

Software: [MIT](LICENSE). Citation metadata for this repository are in `CITATION.cff`.
