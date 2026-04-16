# Report Reproduction Guide

This folder is the code-and-artifact bundle to publish for reproducing the report under `codex_work/codonrl_vs_ga_report.tex`.

It intentionally does **not** include the large checkpoint folders or GA result folders.

## Required external inputs

Place the following folders at the repository root next to this `to_github/` folder:

```text
checkpoints/
results_ga_deap_v2_55_a_cai_1.0/
results_ga_deap_v2_55_a_cai_2.5/
```

Required contents:

- `checkpoints/{i}_linearfold_linearfold/training_summary.json`
- `checkpoints/{i}_linearfold_linearfold/ckpt_best_objective.pth`
- `checkpoints/{i}_linearfold_linearfold/scaling_decoding_summary.json`
- `results_ga_deap_v2_55_a_cai_1.0/seq_{i}/best_ga.fasta`
- `results_ga_deap_v2_55_a_cai_2.5/results_ga_deap_v2_55_a_cai_2.5/seq_{i}/best_ga.fasta`

for `i = 1..55`.

Also required:

```text
config/csc.json
CodonRL_main.py
```

Those two are already included in this bundle.

## Environment

Recommended environment:

```bash
conda env create -f environment.yml
conda activate codonrl
unset PYTHONPATH
```

## What each script does

- `codex_work/compare_codex_metrics.py`
  - Regenerates the first four tables.
  - Uses GA from `results_ga_deap_v2_55_a_cai_1.0/`.
  - Uses CodonRL sequences from `checkpoints/*/scaling_decoding_summary.json`.

- `codex_work/scripts/agent4_generate_decoding_outputs.py`
  - Generates fresh deterministic AGENT-4 CodonRL outputs for `K=1` and `K=50`.
  - Uses decoding objective:
    - `K * Q + 2.5 * CAI + 1.0 * CSC + 1.0 * GC + 1.0 * U`
  - Keeps `alpha_mfe = 0.0`.

- `codex_work/scripts/agent4_compare_codex_vs_ga.py`
  - Regenerates the second four tables.
  - Uses GA from `results_ga_deap_v2_55_a_cai_2.5/results_ga_deap_v2_55_a_cai_2.5/`.
  - Uses AGENT-4 CodonRL outputs from `codex_work/agent4_outputs/`.

## Reproduction steps

From the repository root:

```bash
conda activate codonrl
unset PYTHONPATH
python codex_work/compare_codex_metrics.py
python codex_work/scripts/agent4_generate_decoding_outputs.py --device cpu
python codex_work/scripts/agent4_compare_codex_vs_ga.py
```

Main outputs:

- `codex_work/codonrl_vs_ga_report.tex`
- `codex_work/collected_metrics.csv`
- `codex_work/relative_improvement.csv`
- `codex_work/tables/agent4_*.csv`
- `codex_work/report_updates/agent4_section.tex`
- `codex_work/report_updates/agent4_summary.txt`

## Notes

- The first four tables use GA `a_cai_1.0`.
- The second four tables use GA `a_cai_2.5`.
- The AGENT-4 block does **not** use an explicit MFE decoding term.
- `codex_work/agent4_outputs/` in this bundle contains the generated AGENT-4 CodonRL outputs that were used in the report.
