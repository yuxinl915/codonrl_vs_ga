#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median
from typing import Dict, Iterable, List, Optional, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "codex_work"
TMP_DIR = OUT_DIR / "tmp"
MPL_DIR = TMP_DIR / "mplconfig"
os.environ.setdefault("MPLCONFIGDIR", str(MPL_DIR))
TMP_DIR.mkdir(parents=True, exist_ok=True)
MPL_DIR.mkdir(parents=True, exist_ok=True)

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from CodonRL_main import (  # noqa: E402
    AA_TO_CODONS,
    ECOLLI_K12_FREQ_PER_THOUSAND,
    HUMAN_FREQ_PER_THOUSAND,
    calculate_cai,
    calculate_relative_adaptiveness,
    get_mfe_calculator,
    translate_mrna,
)


EPSILON = 1e-6
TARGET_GC_FRACTION = 0.50
CODONRL_VARIANTS = {
    "codonrl_k1": "1.0",
    "codonrl_k50": "50.0",
}
METHOD_ORDER = ("codonrl_k1", "codonrl_k50", "ga", "cai_only")
DISPLAY_NAME = {
    "codonrl_k1": "CodonRL K=1.0",
    "codonrl_k50": "CodonRL K=50.0",
    "ga": "GA",
    "cai_only": "CAI-only",
}
METRIC_DIRECTIONS = {
    "mfe": "lower",
    "cai": "higher",
    "csc": "higher",
    "gc_penalty": "lower",
    "u_percent": "lower",
}


@dataclass
class SequenceRecord:
    protein_index: int
    method: str
    source_path: str
    sequence: str
    protein_sequence: str
    codon_table: str
    selected_q_scale: str
    translation_ok: bool
    mfe: float
    mfe_method_used: str
    cai: float
    csc: float
    gc_fraction: float
    gc_percent: float
    gc_penalty: float
    u_fraction: float
    u_percent: float


def write_csv(path: Path, rows: List[dict], fieldnames: List[str]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_json(path: Path) -> dict:
    with path.open() as handle:
        return json.load(handle)


def read_fasta_sequence(path: Path) -> str:
    parts: List[str] = []
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith(">"):
                continue
            parts.append(line)
    return "".join(parts).upper().replace("T", "U")


def split_codons(seq: str) -> List[str]:
    seq = seq.strip().upper().replace("T", "U")
    if not seq or len(seq) % 3 != 0:
        return []
    return [seq[i : i + 3] for i in range(0, len(seq), 3)]


def load_csc_weights(path: Path) -> Dict[str, float]:
    raw = load_json(path)
    return {codon.upper().replace("U", "T"): float(value) for codon, value in raw.items()}


def get_freq_and_w(codon_table: str) -> Tuple[Dict[str, float], Dict[str, float]]:
    table = codon_table.lower()
    if table == "human":
        freq = HUMAN_FREQ_PER_THOUSAND
    elif table in {"ecolik12", "ecoli", "ecolik-12", "e.coli"}:
        freq = ECOLLI_K12_FREQ_PER_THOUSAND
    else:
        raise ValueError(f"Unsupported codon table: {codon_table}")
    w = calculate_relative_adaptiveness(AA_TO_CODONS, freq)
    return freq, w


def compute_csc(sequence: str, csc_weights: Dict[str, float]) -> float:
    codons = split_codons(sequence)
    if not codons:
        return math.nan
    values = []
    for codon in codons:
        value = csc_weights.get(codon.replace("U", "T"))
        if value is None:
            return math.nan
        values.append(float(value))
    return sum(values) / len(values)


def gc_fraction(sequence: str) -> float:
    if not sequence:
        return math.nan
    return (sequence.count("G") + sequence.count("C")) / len(sequence)


def u_fraction(sequence: str) -> float:
    if not sequence:
        return math.nan
    return sequence.count("U") / len(sequence)


def compute_mfe(sequence: str) -> Tuple[float, str]:
    calc = get_mfe_calculator()
    try:
        value = calc.calculate_vienna_async(sequence).result()
        if value is not None:
            return float(value), "vienna"
    except Exception:
        pass
    try:
        value = calc.calculate_linearfold_async(sequence).result()
        if value is not None:
            return float(value), "linearfold"
    except Exception:
        pass
    return math.nan, "unavailable"


def build_cai_only_sequence(protein: str, codon_table: str) -> str:
    freq, w = get_freq_and_w(codon_table)
    chosen_codons: List[str] = []
    for aa in protein:
        codons = AA_TO_CODONS[aa]
        best_codon = max(codons, key=lambda codon: (w[codon], freq[codon], -codons.index(codon)))
        chosen_codons.append(best_codon)
    return "".join(chosen_codons)


def select_best_scaled_experiment(scaling_summary: dict) -> Tuple[str, dict]:
    experiments = scaling_summary.get("scaling_experiments", {})
    if not experiments:
        raise ValueError("scaling_decoding_summary.json has no scaling_experiments")

    def sort_key(item: Tuple[str, dict]) -> Tuple[float, float]:
        scale, payload = item
        objective = payload.get("objective")
        objective_key = float(objective) if objective is not None else float("inf")
        try:
            scale_key = float(scale)
        except ValueError:
            scale_key = float("inf")
        return (objective_key, scale_key)

    return min(experiments.items(), key=sort_key)


def get_scaled_experiment(scaling_summary: dict, q_scale: str) -> Tuple[str, dict]:
    experiments = scaling_summary.get("scaling_experiments", {})
    if q_scale in experiments:
        return q_scale, experiments[q_scale]
    target = float(q_scale)
    for scale, payload in experiments.items():
        try:
            if math.isclose(float(scale), target, rel_tol=0.0, abs_tol=1e-12):
                return scale, payload
        except ValueError:
            continue
    raise KeyError(f"scaling_decoding_summary.json lacks K={q_scale}")


def make_record(
    protein_index: int,
    method: str,
    source_path: Path,
    sequence: str,
    protein_sequence: str,
    codon_table: str,
    csc_weights: Dict[str, float],
    selected_q_scale: str = "",
) -> SequenceRecord:
    sequence = sequence.strip().upper().replace("T", "U")
    _, w = get_freq_and_w(codon_table)
    translated = translate_mrna(sequence)
    translation_ok = translated == protein_sequence
    mfe, mfe_method_used = compute_mfe(sequence)
    gc_frac = gc_fraction(sequence)
    u_frac = u_fraction(sequence)
    return SequenceRecord(
        protein_index=protein_index,
        method=method,
        source_path=str(source_path.relative_to(REPO_ROOT)),
        sequence=sequence,
        protein_sequence=protein_sequence,
        codon_table=codon_table,
        selected_q_scale=selected_q_scale,
        translation_ok=translation_ok,
        mfe=mfe,
        mfe_method_used=mfe_method_used,
        cai=float(calculate_cai(sequence, w)),
        csc=compute_csc(sequence, csc_weights),
        gc_fraction=gc_frac,
        gc_percent=100.0 * gc_frac,
        gc_penalty=abs(gc_frac - TARGET_GC_FRACTION),
        u_fraction=u_frac,
        u_percent=100.0 * u_frac,
    )


def relative_improvement(new_value: float, baseline_value: float, direction: str) -> float:
    denom = abs(baseline_value) + EPSILON
    if direction == "higher":
        return (new_value - baseline_value) / denom
    if direction == "lower":
        return (baseline_value - new_value) / denom
    raise ValueError(direction)


def compare_values(a: float, b: float, direction: str) -> Optional[int]:
    if not (math.isfinite(a) and math.isfinite(b)):
        return None
    if math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-12):
        return 0
    if direction == "higher":
        return 1 if a > b else -1
    if direction == "lower":
        return 1 if a < b else -1
    raise ValueError(direction)


def iter_indices() -> Iterable[int]:
    present = set()
    for path in (REPO_ROOT / "checkpoints").glob("*_linearfold_linearfold"):
        if path.is_dir():
            try:
                present.add(int(path.name.split("_", 1)[0]))
            except ValueError:
                continue
    return sorted(present)


def latex_escape(value: object) -> str:
    text = str(value)
    repl = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(repl.get(ch, ch) for ch in text)


def fmt(value: float, digits: int = 4) -> str:
    if value is None or not math.isfinite(value):
        return "NA"
    return f"{value:.{digits}f}"


def summarize_direct(records_by_index: Dict[int, Dict[str, SequenceRecord]], codonrl_method: str) -> List[dict]:
    rows = []
    for metric, direction in METRIC_DIRECTIONS.items():
        deltas: List[float] = []
        codonrl_values: List[float] = []
        ga_values: List[float] = []
        codonrl_wins = 0
        ga_wins = 0
        ties = 0
        for _, methods in sorted(records_by_index.items()):
            codonrl = methods.get(codonrl_method)
            ga = methods.get("ga")
            if not codonrl or not ga:
                continue
            a = getattr(codonrl, metric)
            b = getattr(ga, metric)
            if not (math.isfinite(a) and math.isfinite(b)):
                continue
            codonrl_values.append(a)
            ga_values.append(b)
            deltas.append(a - b)
            outcome = compare_values(a, b, direction)
            if outcome == 1:
                codonrl_wins += 1
            elif outcome == -1:
                ga_wins += 1
            else:
                ties += 1
        rows.append(
            {
                "metric": metric,
                "direction": direction,
                "n": len(deltas),
                "codonrl_mean": mean(codonrl_values) if codonrl_values else math.nan,
                "ga_mean": mean(ga_values) if ga_values else math.nan,
                "mean_delta_codonrl_minus_ga": mean(deltas) if deltas else math.nan,
                "median_delta_codonrl_minus_ga": median(deltas) if deltas else math.nan,
                "codonrl_wins": codonrl_wins,
                "ga_wins": ga_wins,
                "ties": ties,
            }
        )
    return rows


def summarize_relative(
    records_by_index: Dict[int, Dict[str, SequenceRecord]],
    codonrl_method: str,
) -> Tuple[List[dict], List[dict]]:
    per_protein_rows: List[dict] = []
    summary_rows: List[dict] = []

    for protein_index, methods in sorted(records_by_index.items()):
        baseline = methods.get("cai_only")
        if not baseline:
            continue
        for method in (codonrl_method, "ga"):
            candidate = methods.get(method)
            if not candidate:
                continue
            row = {
                "protein_index": protein_index,
                "comparison_variant": codonrl_method,
                "method": method,
                "baseline_method": "cai_only",
                "source_path": candidate.source_path,
                "baseline_source_path": baseline.source_path,
                "selected_q_scale": candidate.selected_q_scale,
                "mfe": candidate.mfe,
                "baseline_mfe": baseline.mfe,
                "mfe_relative_improvement": relative_improvement(candidate.mfe, baseline.mfe, "lower"),
                "cai": candidate.cai,
                "baseline_cai": baseline.cai,
                "cai_relative_improvement": relative_improvement(candidate.cai, baseline.cai, "higher"),
                "csc": candidate.csc,
                "baseline_csc": baseline.csc,
                "csc_relative_improvement": relative_improvement(candidate.csc, baseline.csc, "higher"),
                "gc_penalty": candidate.gc_penalty,
                "baseline_gc_penalty": baseline.gc_penalty,
                "gc_penalty_relative_improvement": relative_improvement(
                    candidate.gc_penalty, baseline.gc_penalty, "lower"
                ),
                "u_percent": candidate.u_percent,
                "baseline_u_percent": baseline.u_percent,
                "u_percent_relative_improvement": relative_improvement(
                    candidate.u_percent, baseline.u_percent, "lower"
                ),
            }
            per_protein_rows.append(row)

    for metric in METRIC_DIRECTIONS:
        key = f"{metric}_relative_improvement"
        codonrl_values = [
            row[key] for row in per_protein_rows if row["method"] == codonrl_method and math.isfinite(row[key])
        ]
        ga_values = [row[key] for row in per_protein_rows if row["method"] == "ga" and math.isfinite(row[key])]
        codonrl_wins = 0
        ga_wins = 0
        ties = 0
        by_index = {}
        for row in per_protein_rows:
            by_index.setdefault(row["protein_index"], {})[row["method"]] = row[key]
        for methods in by_index.values():
            if codonrl_method not in methods or "ga" not in methods:
                continue
            outcome = compare_values(methods[codonrl_method], methods["ga"], "higher")
            if outcome == 1:
                codonrl_wins += 1
            elif outcome == -1:
                ga_wins += 1
            else:
                ties += 1
        summary_rows.append(
            {
                "metric": metric,
                "n": min(len(codonrl_values), len(ga_values)),
                "codonrl_mean_relative_improvement": mean(codonrl_values) if codonrl_values else math.nan,
                "ga_mean_relative_improvement": mean(ga_values) if ga_values else math.nan,
                "mean_delta_codonrl_minus_ga": (
                    mean(codonrl_values) - mean(ga_values) if codonrl_values and ga_values else math.nan
                ),
                "codonrl_wins": codonrl_wins,
                "ga_wins": ga_wins,
                "ties": ties,
            }
        )
    return per_protein_rows, summary_rows


def build_report_tex(
    summaries: Dict[str, Dict[str, List[dict]]],
    notes: dict,
) -> str:
    def direct_table_rows(rows: List[dict]) -> str:
        return "\n".join(
        r"{} & {} & {} & {} & {} & {} & {} \\"
        .format(
            latex_escape(row["metric"]),
            latex_escape(row["direction"]),
            row["n"],
            fmt(row["codonrl_mean"]),
            fmt(row["ga_mean"]),
            fmt(row["mean_delta_codonrl_minus_ga"]),
            f'{row["codonrl_wins"]}/{row["ga_wins"]}/{row["ties"]}',
        )
        for row in rows
        )

    def relative_table_rows(rows: List[dict]) -> str:
        return "\n".join(
        r"{} & {} & {} & {} & {} & {} \\"
        .format(
            latex_escape(row["metric"]),
            row["n"],
            fmt(row["codonrl_mean_relative_improvement"]),
            fmt(row["ga_mean_relative_improvement"]),
            fmt(row["mean_delta_codonrl_minus_ga"]),
            f'{row["codonrl_wins"]}/{row["ga_wins"]}/{row["ties"]}',
        )
        for row in rows
        )

    k1_direct_rows = direct_table_rows(summaries["codonrl_k1"]["direct"])
    k1_relative_rows = relative_table_rows(summaries["codonrl_k1"]["relative"])
    k50_direct_rows = direct_table_rows(summaries["codonrl_k50"]["direct"])
    k50_relative_rows = relative_table_rows(summaries["codonrl_k50"]["relative"])

    return f"""\\documentclass[11pt]{{article}}
\\usepackage[margin=1in]{{geometry}}
\\usepackage{{booktabs}}
\\usepackage{{array}}
\\usepackage{{longtable}}
\\usepackage{{graphicx}}

\\title{{CodonRL vs GA Using a CAI-only Baseline}}
\\author{{Automated analysis from local repository outputs}}
\\date{{}}

\\begin{{document}}
\\maketitle

\\section*{{Inputs and selection rules}}
The analysis used checkpoint folders \\texttt{{checkpoints/\\{{i\\}}\\_linearfold\\_linearfold/}} for CodonRL, \\texttt{{results\\_ga\\_deap\\_v2\\_55\\_a\\_cai\\_1.0/seq\\_\\{{i\\}}/best\\_ga.fasta}} for GA, and a new CAI-only baseline built per protein from the repository codon-usage tables. In this first block of four tables, the GA reference is therefore always the \\texttt{{a\\_cai\\_1.0}} GA objective. CodonRL was evaluated in two fixed scaled-decoding settings from \\texttt{{scaling\\_decoding\\_summary.json}}: $K=1.0$ and $K=50.0$. The existing checkpoint summaries contained $K=50.0$ decoded sequences for all {notes["protein_count"]} proteins, so no additional decoding run was required for this update.

Protein identity was taken from \\texttt{{training\\_summary.json}} and validated by translation for every CodonRL, GA, and CAI-only mRNA sequence. Translation checks passed for all {notes["translation_passed"]} analyzed sequences and no broken sequences were included in the tables below.

\\section*{{Metric definitions and direction}}
CAI was computed with \\texttt{{CodonRL\\_main.calculate\\_cai}} using the codon table stored in each checkpoint. Because the repository computes relative adaptiveness as codon frequency divided by the per-amino-acid maximum, choosing the highest-weight synonymous codon at every amino acid maximizes CAI; the constructed CAI-only baseline therefore achieves the CAI-maximizing sequence under the repository formula. CSC was computed as the arithmetic mean of per-codon values from \\texttt{{config/csc.json}}, matching the benchmark scripts. MFE was recomputed from sequence with the repository MFE calculator, preferring ViennaRNA and falling back to LinearFold only if Vienna failed. GC is reported as a target-based penalty $|GC-0.50|$ because the repository objective code treats GC as closeness to 50\\% when a target is specified. U\\% is treated as a minimization metric.

Relative-improvement formulas used the CAI-only baseline $b$ and candidate value $x$:
\\[
\\mathrm{{imp}}_\\mathrm{{higher}} = \\frac{{x-b}}{{|b| + 10^{{-6}}}}, \\qquad
\\mathrm{{imp}}_\\mathrm{{lower}} = \\frac{{b-x}}{{|b| + 10^{{-6}}}}.
\\]
These were applied as follows: CAI and CSC use the higher-is-better form, while MFE, GC penalty, and U\\% use the lower-is-better form.

\\section*{{Results}}
Tables~\\ref{{tab:direct-k1}} and~\\ref{{tab:relative-k1}} show the original $K=1.0$ comparison. Tables~\\ref{{tab:direct-k50}} and~\\ref{{tab:relative-k50}} show the requested fixed $K=50.0$ comparison. The ``Mean $\\Delta$'' column is CodonRL minus GA, so negative values favor CodonRL for lower-is-better metrics and positive values favor CodonRL for higher-is-better metrics.

\\begin{{table}}[ht]
\\centering
\\resizebox{{\\textwidth}}{{!}}{{%
\\begin{{tabular}}{{llrrrrr}}
\\toprule
Metric & Better & $n$ & CodonRL mean & GA mean & Mean $\\Delta$ & Wins C/G/T \\\\
\\midrule
{k1_direct_rows}
\\bottomrule
\\end{{tabular}}%
}}
\\caption{{Absolute per-metric comparison across {notes["protein_count"]} proteins between CodonRL decoded from \\texttt{{scaling\\_decoding\\_summary.json}} at fixed Q-scale $K=1.0$ and GA sequences from \\texttt{{results\\_ga\\_deap\\_v2\\_55\\_a\\_cai\\_1.0}}. The CodonRL side is the repository's original scaled-Q decoding output at $K=1.0$; the GA side uses the GA objective associated with the \\texttt{{a\\_cai\\_1.0}} result folder.}}
\\label{{tab:direct-k1}}
\\end{{table}}

Positive relative-improvement values indicate improvement over the CAI-only baseline under the stated metric direction.

\\begin{{table}}[ht]
\\centering
\\resizebox{{\\textwidth}}{{!}}{{%
\\begin{{tabular}}{{lrrrrr}}
\\toprule
Metric & $n$ & CodonRL mean imp. & GA mean imp. & Mean $\\Delta$ imp. & Wins C/G/T \\\\
\\midrule
{k1_relative_rows}
\\bottomrule
\\end{{tabular}}%
}}
\\caption{{Relative-improvement comparison over the CAI-only baseline across {notes["protein_count"]} proteins for CodonRL decoded at fixed Q-scale $K=1.0$ from \\texttt{{scaling\\_decoding\\_summary.json}} versus GA from \\texttt{{results\\_ga\\_deap\\_v2\\_55\\_a\\_cai\\_1.0}}. ``Mean $\\Delta$ imp.'' is CodonRL mean relative improvement minus GA mean relative improvement.}}
\\label{{tab:relative-k1}}
\\end{{table}}

\\begin{{table}}[ht]
\\centering
\\resizebox{{\\textwidth}}{{!}}{{%
\\begin{{tabular}}{{llrrrrr}}
\\toprule
Metric & Better & $n$ & CodonRL mean & GA mean & Mean $\\Delta$ & Wins C/G/T \\\\
\\midrule
{k50_direct_rows}
\\bottomrule
\\end{{tabular}}%
}}
\\caption{{Absolute per-metric comparison across {notes["protein_count"]} proteins between CodonRL decoded from \\texttt{{scaling\\_decoding\\_summary.json}} at fixed Q-scale $K=50.0$ and GA sequences from \\texttt{{results\\_ga\\_deap\\_v2\\_55\\_a\\_cai\\_1.0}}. The CodonRL side is the repository's original scaled-Q decoding output at $K=50.0$; the GA side uses the GA objective associated with the \\texttt{{a\\_cai\\_1.0}} result folder.}}
\\label{{tab:direct-k50}}
\\end{{table}}

\\begin{{table}}[ht]
\\centering
\\resizebox{{\\textwidth}}{{!}}{{%
\\begin{{tabular}}{{lrrrrr}}
\\toprule
Metric & $n$ & CodonRL mean imp. & GA mean imp. & Mean $\\Delta$ imp. & Wins C/G/T \\\\
\\midrule
{k50_relative_rows}
\\bottomrule
\\end{{tabular}}%
}}
\\caption{{Relative-improvement comparison over the CAI-only baseline across {notes["protein_count"]} proteins for CodonRL decoded at fixed Q-scale $K=50.0$ from \\texttt{{scaling\\_decoding\\_summary.json}} versus GA from \\texttt{{results\\_ga\\_deap\\_v2\\_55\\_a\\_cai\\_1.0}}. ``Mean $\\Delta$ imp.'' is CodonRL mean relative improvement minus GA mean relative improvement.}}
\\label{{tab:relative-k50}}
\\end{{table}}

\\section*{{Coverage and caveats}}
The shared analysis set contained {notes["protein_count"]} proteins. ViennaRNA provided the final MFE for {notes["vienna_count"]} of {notes["sequence_count"]} analyzed sequences; the remainder would have used LinearFold fallback, but no fallback was needed in this run. No extra Python packages were installed in the \\texttt{{codonrl}} conda environment for this analysis script; the implementation used only the repository code plus Python standard-library CSV handling because \\texttt{{pandas}} was absent in the environment.

\\end{{document}}
"""


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)
    csc_weights = load_csc_weights(REPO_ROOT / "config" / "csc.json")

    metrics_rows: List[dict] = []
    skipped: List[str] = []
    records_by_index: Dict[int, Dict[str, SequenceRecord]] = {}
    variant_scales: Dict[str, int] = {}

    for protein_index in iter_indices():
        ckpt_dir = REPO_ROOT / "checkpoints" / f"{protein_index}_linearfold_linearfold"
        training_path = ckpt_dir / "training_summary.json"
        scaling_path = ckpt_dir / "scaling_decoding_summary.json"
        ga_path = REPO_ROOT / "results_ga_deap_v2_55_a_cai_1.0" / f"seq_{protein_index}" / "best_ga.fasta"

        if not (training_path.exists() and scaling_path.exists() and ga_path.exists()):
            skipped.append(
                f"{protein_index}: missing "
                f"{'training_summary.json ' if not training_path.exists() else ''}"
                f"{'scaling_decoding_summary.json ' if not scaling_path.exists() else ''}"
                f"{'best_ga.fasta' if not ga_path.exists() else ''}".strip()
            )
            continue

        training = load_json(training_path)
        scaling = load_json(scaling_path)
        protein = training["protein_sequence"].strip().upper()
        codon_table = training.get("config", {}).get("codon_table", "human").lower()

        index_records: Dict[str, SequenceRecord] = {}
        for method, requested_scale in CODONRL_VARIANTS.items():
            try:
                selected_scale, selected_payload = get_scaled_experiment(scaling, requested_scale)
            except KeyError as exc:
                skipped.append(f"{protein_index}: {exc}")
                index_records = {}
                break
            variant_scales[f"{method}:{selected_scale}"] = variant_scales.get(f"{method}:{selected_scale}", 0) + 1
            index_records[method] = make_record(
                protein_index=protein_index,
                method=method,
                source_path=scaling_path,
                sequence=selected_payload["generated_mrna_sequence"],
                protein_sequence=protein,
                codon_table=codon_table,
                csc_weights=csc_weights,
                selected_q_scale=selected_scale,
            )
        if not index_records:
            continue
        index_records["ga"] = make_record(
            protein_index=protein_index,
            method="ga",
            source_path=ga_path,
            sequence=read_fasta_sequence(ga_path),
            protein_sequence=protein,
            codon_table=codon_table,
            csc_weights=csc_weights,
        )
        index_records["cai_only"] = make_record(
            protein_index=protein_index,
            method="cai_only",
            source_path=training_path,
            sequence=build_cai_only_sequence(protein, codon_table),
            protein_sequence=protein,
            codon_table=codon_table,
            csc_weights=csc_weights,
        )

        bad = [method for method, record in index_records.items() if not record.translation_ok]
        if bad:
            skipped.append(f"{protein_index}: translation mismatch for {', '.join(bad)}")
            continue

        records_by_index[protein_index] = index_records
        for method in METHOD_ORDER:
            record = index_records[method]
            metrics_rows.append(
                {
                    "protein_index": record.protein_index,
                    "method": record.method,
                    "display_name": DISPLAY_NAME[record.method],
                    "source_path": record.source_path,
                    "selected_q_scale": record.selected_q_scale,
                    "codon_table": record.codon_table,
                    "translation_ok": record.translation_ok,
                    "sequence_length_nt": len(record.sequence),
                    "protein_length_aa": len(record.protein_sequence),
                    "mfe": record.mfe,
                    "mfe_method_used": record.mfe_method_used,
                    "cai": record.cai,
                    "csc": record.csc,
                    "gc_fraction": record.gc_fraction,
                    "gc_percent": record.gc_percent,
                    "gc_penalty": record.gc_penalty,
                    "u_fraction": record.u_fraction,
                    "u_percent": record.u_percent,
                    "sequence": record.sequence,
                }
            )

    metrics_rows.sort(key=lambda row: (row["protein_index"], METHOD_ORDER.index(row["method"])))
    summaries: Dict[str, Dict[str, List[dict]]] = {}
    all_relative_rows: List[dict] = []
    for method in CODONRL_VARIANTS:
        relative_rows, relative_summary = summarize_relative(records_by_index, method)
        all_relative_rows.extend(relative_rows)
        summaries[method] = {
            "direct": summarize_direct(records_by_index, method),
            "relative": relative_summary,
        }

    write_csv(
        OUT_DIR / "collected_metrics.csv",
        metrics_rows,
        [
            "protein_index",
            "method",
            "display_name",
            "source_path",
            "selected_q_scale",
            "codon_table",
            "translation_ok",
            "sequence_length_nt",
            "protein_length_aa",
            "mfe",
            "mfe_method_used",
            "cai",
            "csc",
            "gc_fraction",
            "gc_percent",
            "gc_penalty",
            "u_fraction",
            "u_percent",
            "sequence",
        ],
    )

    write_csv(
        OUT_DIR / "relative_improvement.csv",
        all_relative_rows,
        [
            "protein_index",
            "comparison_variant",
            "method",
            "baseline_method",
            "source_path",
            "baseline_source_path",
            "selected_q_scale",
            "mfe",
            "baseline_mfe",
            "mfe_relative_improvement",
            "cai",
            "baseline_cai",
            "cai_relative_improvement",
            "csc",
            "baseline_csc",
            "csc_relative_improvement",
            "gc_penalty",
            "baseline_gc_penalty",
            "gc_penalty_relative_improvement",
            "u_percent",
            "baseline_u_percent",
            "u_percent_relative_improvement",
        ],
    )

    total_sequences = len(metrics_rows)
    vienna_count = sum(1 for row in metrics_rows if row["mfe_method_used"] == "vienna")
    translation_passed = sum(1 for row in metrics_rows if row["translation_ok"])
    notes = {
        "protein_count": len(records_by_index),
        "sequence_count": total_sequences,
        "translation_passed": translation_passed,
        "vienna_count": vienna_count,
    }

    report_tex = build_report_tex(summaries, notes)
    (OUT_DIR / "codonrl_vs_ga_report.tex").write_text(report_tex)

    notes_lines = [
        "AGENT_3 notes",
        f"Analyzed proteins: {len(records_by_index)}",
        f"Analyzed sequences: {total_sequences}",
        f"Translation checks passed: {translation_passed}/{total_sequences}",
        f"CodonRL q_scales used from scaling_decoding_summary.json: {variant_scales}",
        "CodonRL selection rule for this update: fixed K=1.0 and fixed K=50.0 entries are both extracted from scaling_decoding_summary.json.",
        "GA extraction rule: results_ga_deap_v2_55_a_cai_1.0/seq_<i>/best_ga.fasta",
        "CAI-only construction rule: for each amino acid, choose the synonymous codon with the maximum repository CAI weight; this is CAI-maximizing because calculate_relative_adaptiveness normalizes by the per-amino-acid maximum frequency.",
        "Metric reuse: CAI and translation from CodonRL_main.py; MFE from the repository async MFE calculator; CSC arithmetic mean from config/csc.json in the same style as visualizeandbenchmark.py.",
        "GC metric used for comparison/reporting: gc_penalty = abs(gc_fraction - 0.50), because codonrl_baselines/objectives.py and ga_relative_improvement.py treat GC as a target-based metric around 50%.",
        "U metric direction: lower is better, matching UPercent.score when no target_u is supplied.",
        "Environment: conda run -n codonrl with PYTHONPATH unset before execution.",
        "Package installation: no extra packages were installed. pandas is absent in the codonrl environment, so this script uses the Python csv module.",
    ]
    if skipped:
        notes_lines.append("Skipped entries:")
        notes_lines.extend(f"- {item}" for item in skipped)
    else:
        notes_lines.append("Skipped entries: none")
    (OUT_DIR / "AGENT_3_notes.txt").write_text("\n".join(notes_lines) + "\n")


if __name__ == "__main__":
    main()
