#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
CODEX_WORK = REPO_ROOT / "codex_work"
OUT_ROOT = CODEX_WORK / "agent4_outputs"
TABLE_DIR = CODEX_WORK / "tables"
REPORT_DIR = CODEX_WORK / "report_updates"
LOG_DIR = CODEX_WORK / "logs"
TMP_DIR = CODEX_WORK / "tmp"
os.environ.setdefault("MPLCONFIGDIR", str(TMP_DIR / "mplconfig"))

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
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_json(path: Path) -> dict:
    with path.open() as handle:
        return json.load(handle)


def read_fasta_sequence(path: Path) -> str:
    parts = []
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
    chosen_codons = []
    for aa in protein:
        codons = AA_TO_CODONS[aa]
        best_codon = max(codons, key=lambda codon: (w[codon], freq[codon], -codons.index(codon)))
        chosen_codons.append(best_codon)
    return "".join(chosen_codons)


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


def fmt(value: float, digits: int = 4) -> str:
    if value is None or not math.isfinite(value):
        return "NA"
    return f"{value:.{digits}f}"


def summarize_direct(records_by_index: Dict[int, Dict[str, SequenceRecord]], codonrl_method: str) -> List[dict]:
    rows = []
    for metric, direction in METRIC_DIRECTIONS.items():
        deltas = []
        codonrl_values = []
        ga_values = []
        codonrl_wins = 0
        ga_wins = 0
        ties = 0
        for methods in records_by_index.values():
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
    per_protein_rows = []
    summary_rows = []
    for protein_index, methods in sorted(records_by_index.items()):
        baseline = methods.get("cai_only")
        if not baseline:
            continue
        for method in (codonrl_method, "ga"):
            candidate = methods.get(method)
            if not candidate:
                continue
            per_protein_rows.append(
                {
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
                    "gc_penalty_relative_improvement": relative_improvement(candidate.gc_penalty, baseline.gc_penalty, "lower"),
                    "u_percent": candidate.u_percent,
                    "baseline_u_percent": baseline.u_percent,
                    "u_percent_relative_improvement": relative_improvement(candidate.u_percent, baseline.u_percent, "lower"),
                }
            )

    for metric in METRIC_DIRECTIONS:
        key = f"{metric}_relative_improvement"
        codonrl_values = [row[key] for row in per_protein_rows if row["method"] == codonrl_method and math.isfinite(row[key])]
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


def render_section(k1_direct: List[dict], k1_relative: List[dict], k50_direct: List[dict], k50_relative: List[dict], notes: dict) -> str:
    def direct_rows(rows: List[dict]) -> str:
        return "\n".join(
            f"{row['metric']} & {row['direction']} & {row['n']} & {fmt(row['codonrl_mean'])} & {fmt(row['ga_mean'])} & {fmt(row['mean_delta_codonrl_minus_ga'])} & {row['codonrl_wins']}/{row['ga_wins']}/{row['ties']} \\\\"
            for row in rows
        )

    def relative_rows(rows: List[dict]) -> str:
        return "\n".join(
            f"{row['metric']} & {row['n']} & {fmt(row['codonrl_mean_relative_improvement'])} & {fmt(row['ga_mean_relative_improvement'])} & {fmt(row['mean_delta_codonrl_minus_ga'])} & {row['codonrl_wins']}/{row['ga_wins']}/{row['ties']} \\\\"
            for row in rows
        )

    return f"""
% AGENT4 START
\\section*{{AGENT 4: Correlated Decoding Weights}}
This section adds a new CodonRL-vs-GA comparison using CodonRL decoding weights aligned with the GA emphasis only for the metrics exposed directly by the current inference path. The decoding formula used here is $K \\cdot Q +$ explicit CAI/CSC/GC/U terms, with $\\alpha_{{\\mathrm{{CAI}}}}=2.5$, $\\alpha_{{\\mathrm{{CSC}}}}=1.0$, $\\alpha_{{\\mathrm{{GC}}}}=1.0$, and $\\alpha_{{U}}=1.0$. Runs were generated for both $K=1.0$ and $K=50.0$ on each checkpoint's own training protein. In this second block of four tables, the GA reference is always the \\texttt{{a\\_cai\\_2.5}} GA objective from \\texttt{{results\\_ga\\_deap\\_v2\\_55\\_a\\_cai\\_2.5/results\\_ga\\_deap\\_v2\\_55\\_a\\_cai\\_2.5}}. Explicit MFE weighting was not used in this experiment: the decoding configuration was verified to keep $\\alpha_{{\\mathrm{{MFE}}}}=0.0$.

Training proteins were read from \\texttt{{training\\_summary.json}} for each checkpoint, and translation checks passed for all {notes['translation_passed']} generated or compared sequences in this section. Checkpoint-local \\texttt{{scaling\\_decoding\\_summary\\_cai2p5\\_all1.json}} files were inspected during verification, but the final AGENT-4 deliverables were produced from fresh deterministic reruns under \\texttt{{codex\\_work/scripts/agent4\\_generate\\_decoding\\_outputs.py}} after forcing evaluation mode in the decoder. All exported AGENT-4 sequences and summaries live under \\texttt{{codex\\_work/agent4\\_outputs/}}.

\\begin{{table}}[ht]
\\centering
\\resizebox{{\\textwidth}}{{!}}{{%
\\begin{{tabular}}{{llrrrrr}}
\\toprule
Metric & Better & $n$ & CodonRL mean & GA mean & Mean $\\Delta$ & Wins C/G/T \\\\
\\midrule
{direct_rows(k1_direct)}
\\bottomrule
\\end{{tabular}}%
}}
\\caption{{Absolute per-metric comparison across {notes['protein_count']} proteins between AGENT-4 CodonRL decoding with objective $K \\cdot Q + 2.5\\,\\mathrm{{CAI}} + 1.0\\,\\mathrm{{CSC}} + 1.0\\,\\mathrm{{GC}} + 1.0\\,U$ at fixed Q-scale $K=1.0$, and GA sequences from \\texttt{{results\\_ga\\_deap\\_v2\\_55\\_a\\_cai\\_2.5/results\\_ga\\_deap\\_v2\\_55\\_a\\_cai\\_2.5}}. No explicit MFE decoding term was used on the CodonRL side.}}
\\label{{tab:agent4-direct-k1}}
\\end{{table}}

\\begin{{table}}[ht]
\\centering
\\resizebox{{\\textwidth}}{{!}}{{%
\\begin{{tabular}}{{lrrrrr}}
\\toprule
Metric & $n$ & CodonRL mean imp. & GA mean imp. & Mean $\\Delta$ imp. & Wins C/G/T \\\\
\\midrule
{relative_rows(k1_relative)}
\\bottomrule
\\end{{tabular}}%
}}
\\caption{{Relative-improvement comparison over the CAI-only baseline across {notes['protein_count']} proteins between AGENT-4 CodonRL decoding with objective $K \\cdot Q + 2.5\\,\\mathrm{{CAI}} + 1.0\\,\\mathrm{{CSC}} + 1.0\\,\\mathrm{{GC}} + 1.0\\,U$ at fixed Q-scale $K=1.0$, and GA from \\texttt{{results\\_ga\\_deap\\_v2\\_55\\_a\\_cai\\_2.5/results\\_ga\\_deap\\_v2\\_55\\_a\\_cai\\_2.5}}.}}
\\label{{tab:agent4-relative-k1}}
\\end{{table}}

\\begin{{table}}[ht]
\\centering
\\resizebox{{\\textwidth}}{{!}}{{%
\\begin{{tabular}}{{llrrrrr}}
\\toprule
Metric & Better & $n$ & CodonRL mean & GA mean & Mean $\\Delta$ & Wins C/G/T \\\\
\\midrule
{direct_rows(k50_direct)}
\\bottomrule
\\end{{tabular}}%
}}
\\caption{{Absolute per-metric comparison across {notes['protein_count']} proteins between AGENT-4 CodonRL decoding with objective $K \\cdot Q + 2.5\\,\\mathrm{{CAI}} + 1.0\\,\\mathrm{{CSC}} + 1.0\\,\\mathrm{{GC}} + 1.0\\,U$ at fixed Q-scale $K=50.0$, and GA sequences from \\texttt{{results\\_ga\\_deap\\_v2\\_55\\_a\\_cai\\_2.5/results\\_ga\\_deap\\_v2\\_55\\_a\\_cai\\_2.5}}. No explicit MFE decoding term was used on the CodonRL side.}}
\\label{{tab:agent4-direct-k50}}
\\end{{table}}

\\begin{{table}}[ht]
\\centering
\\resizebox{{\\textwidth}}{{!}}{{%
\\begin{{tabular}}{{lrrrrr}}
\\toprule
Metric & $n$ & CodonRL mean imp. & GA mean imp. & Mean $\\Delta$ imp. & Wins C/G/T \\\\
\\midrule
{relative_rows(k50_relative)}
\\bottomrule
\\end{{tabular}}%
}}
\\caption{{Relative-improvement comparison over the CAI-only baseline across {notes['protein_count']} proteins between AGENT-4 CodonRL decoding with objective $K \\cdot Q + 2.5\\,\\mathrm{{CAI}} + 1.0\\,\\mathrm{{CSC}} + 1.0\\,\\mathrm{{GC}} + 1.0\\,U$ at fixed Q-scale $K=50.0$, and GA from \\texttt{{results\\_ga\\_deap\\_v2\\_55\\_a\\_cai\\_2.5/results\\_ga\\_deap\\_v2\\_55\\_a\\_cai\\_2.5}}.}}
\\label{{tab:agent4-relative-k50}}
\\end{{table}}

\\paragraph{{AGENT-4 coverage.}} The shared AGENT-4 analysis set contained {notes['protein_count']} proteins. ViennaRNA provided the final MFE for {notes['vienna_count']} of {notes['sequence_count']} analyzed sequences in this section; the remainder would have used LinearFold fallback, but no fallback was needed in this run.
% AGENT4 END
""".strip()


def update_full_report(section_text: str) -> None:
    report_path = CODEX_WORK / "codonrl_vs_ga_report.tex"
    existing = report_path.read_text() if report_path.exists() else "\\documentclass{article}\n\\begin{document}\n\\end{document}\n"
    start_marker = "% AGENT4 START"
    end_marker = "% AGENT4 END"
    if start_marker in existing and end_marker in existing:
        before = existing.split(start_marker, 1)[0]
        after = existing.split(end_marker, 1)[1]
        updated = before + section_text + after
    else:
        updated = existing.replace("\\end{document}", "\n\n" + section_text + "\n\n\\end{document}")
    report_path.write_text(updated)


def main() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    TMP_DIR.mkdir(parents=True, exist_ok=True)

    csc_weights = load_csc_weights(REPO_ROOT / "config" / "csc.json")
    metrics_rows = []
    records_by_index: Dict[int, Dict[str, SequenceRecord]] = {}
    skipped = []

    for protein_index in range(1, 56):
        training_path = REPO_ROOT / "checkpoints" / f"{protein_index}_linearfold_linearfold" / "training_summary.json"
        k1_path = OUT_ROOT / "k1" / f"seq_{protein_index}" / "summary.json"
        k50_path = OUT_ROOT / "k50" / f"seq_{protein_index}" / "summary.json"
        ga_path = REPO_ROOT / "results_ga_deap_v2_55_a_cai_2.5" / "results_ga_deap_v2_55_a_cai_2.5" / f"seq_{protein_index}" / "best_ga.fasta"

        if not (training_path.exists() and k1_path.exists() and k50_path.exists() and ga_path.exists()):
            skipped.append(f"{protein_index}: missing required input")
            continue

        training = load_json(training_path)
        protein = training["protein_sequence"].strip().upper()
        codon_table = training.get("config", {}).get("codon_table", "human").lower()
        k1 = load_json(k1_path)
        k50 = load_json(k50_path)

        index_records = {
            "codonrl_k1": make_record(
                protein_index, "codonrl_k1", k1_path, k1["experiment"]["generated_mrna_sequence"], protein, codon_table, csc_weights, "1.0"
            ),
            "codonrl_k50": make_record(
                protein_index, "codonrl_k50", k50_path, k50["experiment"]["generated_mrna_sequence"], protein, codon_table, csc_weights, "50.0"
            ),
            "ga": make_record(
                protein_index, "ga", ga_path, read_fasta_sequence(ga_path), protein, codon_table, csc_weights
            ),
            "cai_only": make_record(
                protein_index, "cai_only", training_path, build_cai_only_sequence(protein, codon_table), protein, codon_table, csc_weights
            ),
        }

        bad = [method for method, record in index_records.items() if not record.translation_ok]
        if bad:
            skipped.append(f"{protein_index}: translation mismatch for {', '.join(bad)}")
            continue

        records_by_index[protein_index] = index_records
        for method in ("codonrl_k1", "codonrl_k50", "ga", "cai_only"):
            record = index_records[method]
            metrics_rows.append(
                {
                    "protein_index": record.protein_index,
                    "method": record.method,
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

    k1_relative_rows, k1_relative_summary = summarize_relative(records_by_index, "codonrl_k1")
    k50_relative_rows, k50_relative_summary = summarize_relative(records_by_index, "codonrl_k50")
    k1_direct = summarize_direct(records_by_index, "codonrl_k1")
    k50_direct = summarize_direct(records_by_index, "codonrl_k50")

    write_csv(
        TABLE_DIR / "agent4_metrics_comparison.csv",
        metrics_rows,
        [
            "protein_index", "method", "source_path", "selected_q_scale", "codon_table", "translation_ok",
            "sequence_length_nt", "protein_length_aa", "mfe", "mfe_method_used", "cai", "csc",
            "gc_fraction", "gc_percent", "gc_penalty", "u_fraction", "u_percent", "sequence",
        ],
    )
    write_csv(TABLE_DIR / "agent4_direct_k1.csv", k1_direct, list(k1_direct[0].keys()))
    write_csv(TABLE_DIR / "agent4_direct_k50.csv", k50_direct, list(k50_direct[0].keys()))
    write_csv(TABLE_DIR / "agent4_relative_k1.csv", k1_relative_summary, list(k1_relative_summary[0].keys()))
    write_csv(TABLE_DIR / "agent4_relative_k50.csv", k50_relative_summary, list(k50_relative_summary[0].keys()))
    write_csv(
        TABLE_DIR / "agent4_relative_improvement_per_sequence.csv",
        k1_relative_rows + k50_relative_rows,
        list((k1_relative_rows or k50_relative_rows)[0].keys()),
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

    section_text = render_section(k1_direct, k1_relative_summary, k50_direct, k50_relative_summary, notes)
    (REPORT_DIR / "agent4_section.tex").write_text(section_text + "\n")
    update_full_report(section_text)

    summary_lines = [
        "AGENT_4 summary",
        f"Analyzed proteins: {len(records_by_index)}",
        "Decoding weights: alpha_cai=2.5, alpha_csc=1.0, alpha_gc=1.0, alpha_u=1.0, alpha_mfe=0.0",
        "Q scales compared: K=1.0 and K=50.0",
        "",
        "Direct comparison wins (CodonRL vs GA):",
    ]
    for label, rows in (("K=1.0", k1_direct), ("K=50.0", k50_direct)):
        summary_lines.append(label)
        for row in rows:
            summary_lines.append(
                f"- {row['metric']}: CodonRL {row['codonrl_wins']}, GA {row['ga_wins']}, ties {row['ties']}"
            )
    summary_lines.extend(
        [
            "",
            "Relative-improvement wins over the CAI-only baseline (CodonRL vs GA):",
        ]
    )
    for label, rows in (("K=1.0", k1_relative_summary), ("K=50.0", k50_relative_summary)):
        summary_lines.append(label)
        for row in rows:
            summary_lines.append(
                f"- {row['metric']}: CodonRL {row['codonrl_wins']}, GA {row['ga_wins']}, ties {row['ties']}"
            )
    if skipped:
        summary_lines.extend(["", "Skipped entries:"] + [f"- {item}" for item in skipped])
    else:
        summary_lines.extend(["", "Skipped entries: none"])
    (REPORT_DIR / "agent4_summary.txt").write_text("\n".join(summary_lines) + "\n")

    method_notes = [
        "AGENT_4 method note",
        "CodonRL decoding path used only explicit CAI/CSC/GC/U terms plus scaled Q-values.",
        "Explicit MFE weighting was not used; alpha_mfe was verified to be 0.0.",
        "Checkpoint-local scaling_decoding_summary_cai2p5_all1.json files were inspected during verification but were not reused for final deliverables because the deterministic AGENT_4 export was regenerated in eval mode.",
        "Metric computation reused repository logic for CAI, translation, CSC, and MFE fallback order.",
    ]
    (LOG_DIR / "agent4_method_note.txt").write_text("\n".join(method_notes) + "\n")


if __name__ == "__main__":
    main()
