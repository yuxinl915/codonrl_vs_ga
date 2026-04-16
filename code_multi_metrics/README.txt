Files created in codex_work/

- analyze_mrna_designs.py: standalone analysis script that scans checkpoints and baseline folders, recomputes metrics, and writes all outputs.
- metrics_comparison.csv: one row per (protein_index, method) pair for the available sequences.
- relative_improvements_vs_random.csv: CodonRL-vs-random and GA-vs-random relative improvements for MFE, CAI, CSC, GC penalty, and U percent.
- report.txt: plain-English summary of processed indices, skipped indices, win counts, and caveats.
- tmp/: temporary directory used for matplotlib cache and any scratch files.

How to rerun

1. source ~/miniconda3/etc/profile.d/conda.sh
2. conda activate codonrl
3. unset PYTHONPATH
4. python codex_work/analyze_mrna_designs.py

Environment used

- Conda environment: codonrl
- Python interpreter: python from the activated `codonrl` environment
- Packages installed during this task: none

Metric details

- MFE: repository MFE calculator, preferring ViennaRNA and falling back to LinearFold if needed.
- CAI: `CodonRL_main.calculate_cai` with repository codon-usage tables.
- CSC: arithmetic mean of per-codon values from `config/csc.json` after converting RNA codons to DNA keys.
- GC penalty: `abs(gc_fraction - 0.50)`.
- U percent: `100 * count('U') / sequence_length`.

Relative improvement formulas

- Lower-is-better metrics: `(baseline - candidate) / (abs(baseline) + 1e-6)`
- Higher-is-better metrics: `(candidate - baseline) / (abs(baseline) + 1e-6)`
- GC penalty: `(baseline_penalty - candidate_penalty) / (abs(baseline_penalty) + 1e-6)` where `gc_penalty = abs(gc_fraction - 0.50)`.
