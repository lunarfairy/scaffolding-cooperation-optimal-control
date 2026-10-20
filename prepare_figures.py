"""Assemble figure inputs from completed simulations and human-data analyses."""
import argparse
import gzip
import json
import shutil
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path, obj):
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def paired_difference(row, x, y):
    key = f"final|{x}|{y}"
    if key in row["diffs"]:
        return row["diffs"][key]
    value, se = row["diffs"][f"final|{y}|{x}"]
    return -value, se


def published_comparisons(rows):
    output = []
    for row in rows:
        st = row["setting"]
        if st["model"] != "pub":
            continue
        pol = row["policies"]
        windows = [n for n in pol if n.startswith("W") and "_enc" not in n]
        best_w = max(windows, key=lambda n: pol[n]["final"])
        best_p = max((n for n in pol if n.startswith("P")), key=lambda n: pol[n]["final"])
        nonoverlap = [n for n in windows if int(n[1:].split(",")[0]) < int(n[1:].split(",")[1])]
        gain, se = paired_difference(row, best_w, best_p)
        gain_enc, se_enc = paired_difference(row, best_w, "enc") if "enc" in pol else (None, None)
        output.append(dict(
            tag=row["tag"], model="pub", cond=st["cond"], kappa=st["kappa"], T=st["T"],
            best_window=best_w, win=pol[best_w]["final"], win_se=pol[best_w]["final_se"],
            best_tp=best_p, tp=pol[best_p]["final"], gain=gain, gain_se=se,
            best_window_nonoverlap=max(nonoverlap, key=lambda n: pol[n]["final"]),
            excl=pol["P0"]["final"], enc=pol["enc"]["final"] if "enc" in pol else None,
            gain_enc=gain_enc, gain_enc_se=se_enc))
    return output


def refit_comparisons(summary):
    output = []
    for key in ("hat", "lo", "hi"):
        row = summary[key]
        pol = row["policies"]
        best_w, best_p = row["best_window"], row["best_twophase"]
        output.append(dict(
            tag="refit_main" if key == "hat" else "refit", model="refit", cond="base",
            kappa=row["kappa"], T=15, best_window=best_w, win=pol[best_w]["final"],
            win_se=pol[best_w]["final_se"], best_tp="P" + best_p[1:].split(",")[0],
            tp=pol[best_p]["final"], gain=row["gain"][0], gain_se=row["gain"][1],
            excl=pol["W0,1"]["final"], enc=pol["enc"]["final"],
            gain_enc=row["gain_enc"][0], gain_enc_se=row["gain_enc"][1]))
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "figure_data")
    out = parser.parse_args().output.resolve()
    src, results = ROOT / "src", ROOT / "results"
    # Read all prerequisites before replacing any figure input.
    abm_a = read(src / "abm_a_results.json")
    abm_c = read(src / "abm_c_results.json")
    accuracy = read(src / "ipa_acc.json")
    costate = read(src / "ipa_costate.json")
    sweep = read(src / "ipa_sweep.json") + costate["sweep40"]
    revised = read(src / "rev_abm_results.json")
    evaluated = read(src / "rev_abm2_results.json")
    refit = read(results / "refit_abm/fresh_summary.json")
    records = [read(p) for p in sorted(src.glob("oc_*_d3.json"))]
    if not records:
        raise FileNotFoundError("No full-depth optimisation records in src/.")
    best = {}
    for record in records:
        key = (record["obj"], record["T"], record["kappa"])
        if key not in best or record["J"] > best[key]["J"]:
            best[key] = record
    selected = []
    for row in evaluated:
        st = row["setting"]
        record = best[(row["obj"], st["T"], st["kappa"])]
        policy = row["policies"][f"ipa_{record['start']}"]
        selected.append(dict(obj=row["obj"], T=st["T"], kappa=st["kappa"],
                             start=record["start"], J_red=record["J"],
                             abm_final=policy["final"], abm_se=policy["final_se"]))
    human_files = ("human_tie_state_gaps.csv", "human_tie_state_by_round.csv")
    for name in human_files:
        if not (results / name).is_file():
            raise FileNotFoundError(f"Run human_data_tie_state.py to generate results/{name}.")
    table = pd.DataFrame(sweep)
    chosen = table.loc[table.groupby(["kappa", "T"])["final"].idxmax()].copy()
    chosen["L"] = chosen["T"] - 1 - chosen["s"]
    comparisons = published_comparisons(revised["reeval"]) + refit_comparisons(refit)
    controls = {str(k): best[("final", 15, k)]["u"] for k in (0.5, 0.75, 1.0)}
    out.mkdir(parents=True, exist_ok=True)
    write(out / "abm_trajectories.json", [r for r in abm_a if r["kind"] == "traj"])
    write(out / "ipa_acc.json", accuracy)
    pd.DataFrame([r for r in abm_a if r["kind"] == "sweep"]).to_csv(out / "abm_two_phase_sweep.csv", index=False)
    chosen.pivot(index="kappa", columns="T", values="L").to_csv(out / "ipa_exclusion_window_length_L.csv")
    write(out / "ipa_costate.json", costate)
    plot_revised = {k: revised[k] for k in ("capacc", "tieiv", "tiestats")}
    plot_revised["reeval"] = [r for r in revised["reeval"] if r["setting"]["model"] == "pub"]
    (out / "rev_abm_results.json.gz").write_bytes(gzip.compress(json.dumps(plot_revised).encode(), mtime=0))
    pd.DataFrame([r for r in abm_c if r["task"] == "window"]).to_csv(out / "abm_window_screening_grid.csv", index=False)
    pd.DataFrame(comparisons).to_csv(out / "fresh_window_vs_twophase.csv", index=False)
    pd.DataFrame(selected).to_csv(out / "reduced_model_optimum_preselected_abm.csv", index=False)
    write(out / "reduced_model_best_controls_T15.json", controls)
    for name in human_files:
        shutil.copy2(results / name, out / name)
    print(f"Figure inputs written to {out}")


if __name__ == "__main__":
    main()
