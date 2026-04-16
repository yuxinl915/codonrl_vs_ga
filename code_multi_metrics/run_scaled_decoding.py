#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(REPO_ROOT / "codex_work" / "tmp" / "mplconfig"))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from CodonRL_main import (  # noqa: E402
    AA_TO_CODONS,
    CODON_TO_INT,
    INT_TO_CODON,
    CodonRL,
    ECOLLI_K12_FREQ_PER_THOUSAND,
    HUMAN_FREQ_PER_THOUSAND,
    calculate_cai,
    calculate_relative_adaptiveness,
    configure_target_w_table,
    get_mfe_calculator,
    shutdown_mfe_calculator,
)


def to_rna(seq: str) -> str:
    return (seq or "").strip().upper().replace("T", "U")


def to_dna(seq: str) -> str:
    return (seq or "").strip().upper().replace("U", "T")


def split_codons(seq: str) -> list[str]:
    seq = (seq or "").strip().upper()
    if not seq or len(seq) % 3 != 0:
        return []
    return [seq[i : i + 3] for i in range(0, len(seq), 3)]


def gc_fraction(seq: str) -> float:
    seq = (seq or "").strip().upper()
    if not seq:
        return math.nan
    return (seq.count("G") + seq.count("C")) / len(seq)


def u_fraction(seq: str) -> float:
    seq = (seq or "").strip().upper()
    if not seq:
        return math.nan
    return seq.count("U") / len(seq)


def normalize_csc_weights(raw_weights: dict[str, float]) -> dict[str, float]:
    return {to_rna(codon): float(value) for codon, value in raw_weights.items()}


def calculate_csc(mrna: str, csc_weights_rna: dict[str, float]) -> float:
    codons = split_codons(to_rna(mrna))
    if not codons:
        return math.nan
    vals: list[float] = []
    for codon in codons:
        value = csc_weights_rna.get(codon)
        if value is None:
            return math.nan
        vals.append(float(value))
    return sum(vals) / len(vals)


def load_csc_weights(csc_path: Path) -> dict[str, float]:
    with csc_path.open() as handle:
        raw = json.load(handle)
    return normalize_csc_weights(raw)


def load_summary(summary_path: Path) -> dict[str, Any]:
    with summary_path.open() as handle:
        return json.load(handle)


def load_cfg_and_w(summary: dict[str, Any]) -> tuple[dict[str, Any], dict[str, float]]:
    cfg = dict(summary["config"])
    table = (cfg.get("codon_table", "human") or "human").lower()
    if table == "human":
        freq = HUMAN_FREQ_PER_THOUSAND
    elif table in ("ecolik12", "ecoli", "ecolik-12", "e.coli"):
        freq = ECOLLI_K12_FREQ_PER_THOUSAND
    else:
        raise ValueError(f"Unsupported codon table: {table}")
    w = calculate_relative_adaptiveness(AA_TO_CODONS, freq)
    configure_target_w_table(w)
    return cfg, w


def build_agent(cfg: dict[str, Any], device: str) -> CodonRL:
    cfg = dict(cfg)
    cfg["device"] = device
    cfg["use_amp"] = False
    cfg["eps_start"] = 0.0
    cfg["eps_end"] = 0.0
    cfg["eps_decay"] = 1
    return CodonRL(cfg)


def _gc_term(
    prefix_mrna: str,
    codon: str,
    alpha_gc: float,
    target_gc: float | None,
) -> float:
    if alpha_gc == 0.0:
        return 0.0
    codon_gc = gc_fraction(codon)
    if target_gc is None:
        return alpha_gc * codon_gc
    current_codons = len(prefix_mrna) / 3
    current_gc = gc_fraction(prefix_mrna) if prefix_mrna else 0.5
    new_gc = ((current_gc * current_codons) + codon_gc) / (current_codons + 1)
    return alpha_gc * (-abs(new_gc - target_gc)) * 10.0


def _u_term(
    prefix_mrna: str,
    codon: str,
    alpha_u: float,
    target_u: float | None,
) -> float:
    if alpha_u == 0.0:
        return 0.0
    codon_u = u_fraction(codon)
    if target_u is None:
        return alpha_u * (-codon_u)
    current_codons = len(prefix_mrna) / 3
    current_u = u_fraction(prefix_mrna) if prefix_mrna else 0.25
    new_u = ((current_u * current_codons) + codon_u) / (current_codons + 1)
    return alpha_u * (-abs(new_u - target_u)) * 10.0


def _mfe_term(
    mfe_calc: Any,
    prefix_mrna: str,
    codon: str,
    alpha_mfe: float,
) -> float:
    if alpha_mfe == 0.0 or mfe_calc is None or len(prefix_mrna) < 30:
        return 0.0
    try:
        mfe = mfe_calc.calculate_vienna_async(prefix_mrna + codon).result(timeout=0.1)
    except Exception:
        return 0.0
    if mfe is None:
        return 0.0
    return alpha_mfe * float(mfe)


def scaled_multiobjective_decode(
    agent: CodonRL,
    protein: str,
    w: dict[str, float],
    csc_weights: dict[str, float],
    q_scale: float,
    alpha_cai: float,
    alpha_csc: float,
    alpha_gc: float,
    alpha_u: float,
    target_gc: float | None,
    target_u: float | None,
    alpha_mfe: float,
    mfe_calc: Any,
    include_step_trace: bool,
) -> tuple[str, dict[str, Any]]:
    logw = {codon: math.log(max(weight, 1e-12)) for codon, weight in w.items()}
    logcsc = {
        codon: math.log(max(weight, 1e-12))
        for codon, weight in csc_weights.items()
        if weight > 0.0
    }

    agent._precompute_protein_memory(protein)
    mrna = ""
    cumulative_raw_q = 0.0
    cumulative_scaled_q = 0.0
    cumulative_metric_bonus = 0.0
    cumulative_total_score = 0.0
    step_trace: list[dict[str, Any]] | None = [] if include_step_trace else None

    for position, aa in enumerate(protein):
        state = agent._get_state(mrna, position)
        with torch.no_grad():
            q_values = agent.policy_net.decode_mrna(
                state["mrna"],
                state["pos"],
                agent.protein_memory_cache,
                agent.protein_pad_mask_cache,
            )[0]

        candidates: list[dict[str, Any]] = []
        for codon in AA_TO_CODONS[aa]:
            idx = CODON_TO_INT[codon]
            raw_q = float(q_values[idx].item())
            cai_term = alpha_cai * logw.get(codon, 0.0) if alpha_cai != 0.0 else 0.0
            csc_term = alpha_csc * logcsc.get(codon, 0.0) if alpha_csc != 0.0 else 0.0
            gc_term = _gc_term(mrna, codon, alpha_gc, target_gc)
            u_term = _u_term(mrna, codon, alpha_u, target_u)
            mfe_term = _mfe_term(mfe_calc, mrna, codon, alpha_mfe)
            metric_bonus = cai_term + csc_term + gc_term + u_term + mfe_term
            total_score = (q_scale * raw_q) + metric_bonus
            candidates.append(
                {
                    "codon": codon,
                    "raw_q": raw_q,
                    "scaled_q": q_scale * raw_q,
                    "cai_term": cai_term,
                    "csc_term": csc_term,
                    "gc_term": gc_term,
                    "u_term": u_term,
                    "mfe_term": mfe_term,
                    "metric_bonus": metric_bonus,
                    "total_score": total_score,
                }
            )

        best = max(candidates, key=lambda item: item["total_score"])
        mrna += best["codon"]
        cumulative_raw_q += best["raw_q"]
        cumulative_scaled_q += best["scaled_q"]
        cumulative_metric_bonus += best["metric_bonus"]
        cumulative_total_score += best["total_score"]
        if step_trace is not None:
            step_trace.append(
                {
                    "position": position,
                    "amino_acid": aa,
                    "chosen_codon": best["codon"],
                    "raw_q": best["raw_q"],
                    "scaled_q": best["scaled_q"],
                    "metric_bonus": best["metric_bonus"],
                    "total_score": best["total_score"],
                    "runner_up_total_score_gap": (
                        best["total_score"]
                        - sorted(
                            (candidate["total_score"] for candidate in candidates), reverse=True
                        )[1]
                        if len(candidates) > 1
                        else None
                    ),
                }
            )

    num_steps = len(protein)
    decode_stats = {
        "num_steps": num_steps,
        "cumulative_raw_q": cumulative_raw_q,
        "cumulative_scaled_q": cumulative_scaled_q,
        "cumulative_metric_bonus": cumulative_metric_bonus,
        "cumulative_total_score": cumulative_total_score,
        "mean_raw_q": cumulative_raw_q / num_steps if num_steps else math.nan,
        "mean_scaled_q": cumulative_scaled_q / num_steps if num_steps else math.nan,
        "mean_metric_bonus": cumulative_metric_bonus / num_steps if num_steps else math.nan,
        "mean_total_score": cumulative_total_score / num_steps if num_steps else math.nan,
    }
    if step_trace is not None:
        decode_stats["step_trace"] = step_trace
    return mrna, decode_stats


def compute_mfe_comparison(mrna: str) -> dict[str, Any]:
    mfe_calc = get_mfe_calculator()
    try:
        linearfold_mfe = mfe_calc.calculate_linearfold_async(mrna).result()
    except Exception:
        linearfold_mfe = None
    try:
        vienna_mfe = mfe_calc.calculate_vienna_async(mrna).result()
    except Exception:
        vienna_mfe = None

    if linearfold_mfe is not None and vienna_mfe is not None and abs(vienna_mfe) > 1e-12:
        difference = float(vienna_mfe - linearfold_mfe)
        relative_difference = float(difference / abs(vienna_mfe))
    else:
        difference = None
        relative_difference = None

    return {
        "linearfold_mfe": linearfold_mfe,
        "vienna_mfe": vienna_mfe,
        "mfe_difference": difference,
        "mfe_relative_difference": relative_difference,
    }


def summarize_experiment(
    *,
    task_id: str,
    k_value: float,
    mrna: str,
    decode_stats: dict[str, Any],
    w: dict[str, float],
    csc_weights: dict[str, float],
    cfg: dict[str, Any],
    source_summary: dict[str, Any],
) -> dict[str, Any]:
    mrna = to_rna(mrna)
    cds_dna = to_dna(mrna)
    cai_value = calculate_cai(mrna, w)
    csc_value = calculate_csc(mrna, csc_weights)
    mfe_comparison = compute_mfe_comparison(mrna)
    log_sum_w = sum(math.log(max(w[codon], 1e-12)) for codon in split_codons(mrna))
    objective_mfe = (
        mfe_comparison["linearfold_mfe"]
        if (cfg.get("final_mfe_method", "vienna") == "linearfold")
        else mfe_comparison["vienna_mfe"]
    )
    objective_value = (
        None
        if objective_mfe is None
        else float(objective_mfe - (cfg.get("lambda_val", 0.0) * log_sum_w))
    )

    return {
        "task_id": task_id,
        "q_scale": k_value,
        "sequence": mrna,
        "generated_mrna_sequence": mrna,
        "generated_cds_dna_sequence": cds_dna,
        "protein_length": len(source_summary["protein_sequence"]),
        "score": decode_stats["cumulative_total_score"],
        "decoding_stats": decode_stats,
        "objective": objective_value,
        "objective_mfe": objective_mfe,
        "cai": cai_value,
        "csc": csc_value,
        "gc_content": gc_fraction(cds_dna) * 100.0,
        "gc_content_fraction": gc_fraction(cds_dna),
        "u_percent": u_fraction(mrna) * 100.0,
        "u_fraction": u_fraction(mrna),
        "mfe_comparison": mfe_comparison,
        "source_checkpoint_metrics": {
            "best_objective": source_summary.get("best_objective", {}),
            "best_mfe": source_summary.get("best_mfe", {}),
            "best_cai": source_summary.get("best_cai", {}),
            "final_metrics": source_summary.get("final_metrics", {}),
        },
    }


def checkpoint_dirs(ckpt_root: Path, indices: list[int] | None) -> list[Path]:
    if indices:
        return [ckpt_root / f"{idx}_linearfold_linearfold" for idx in indices]
    paths = sorted(ckpt_root.glob("*_linearfold_linearfold"), key=lambda p: int(p.name.split("_", 1)[0]))
    return [path for path in paths if path.is_dir()]


def run_for_checkpoint(
    ckpt_dir: Path,
    k_values: list[float],
    args: argparse.Namespace,
    csc_weights: dict[str, float],
) -> dict[str, Any]:
    summary_path = ckpt_dir / "training_summary.json"
    ckpt_path = ckpt_dir / "ckpt_best_objective.pth"
    source_summary = load_summary(summary_path)
    cfg, w = load_cfg_and_w(source_summary)
    protein = source_summary["protein_sequence"].strip().upper()
    device = args.device
    agent = build_agent(cfg, device)
    state_dict = torch.load(ckpt_path, map_location=device)
    agent.policy_net.load_state_dict(state_dict)
    agent.target_net.load_state_dict(state_dict)
    mfe_calc = get_mfe_calculator() if args.alpha_mfe != 0.0 else None

    scaling_experiments: OrderedDict[str, Any] = OrderedDict()
    for k_value in k_values:
        mrna, decode_stats = scaled_multiobjective_decode(
            agent=agent,
            protein=protein,
            w=w,
            csc_weights=csc_weights,
            q_scale=k_value,
            alpha_cai=args.alpha_cai,
            alpha_csc=args.alpha_csc,
            alpha_gc=args.alpha_gc,
            alpha_u=args.alpha_u,
            target_gc=args.target_gc,
            target_u=args.target_u,
            alpha_mfe=args.alpha_mfe,
            mfe_calc=mfe_calc,
            include_step_trace=args.include_step_trace,
        )
        scaling_experiments[str(k_value)] = summarize_experiment(
            task_id=source_summary["task_id"],
            k_value=k_value,
            mrna=mrna,
            decode_stats=decode_stats,
            w=w,
            csc_weights=csc_weights,
            cfg=cfg,
            source_summary=source_summary,
        )

    output = {
        "task_id": source_summary["task_id"],
        "protein_sequence": protein,
        "protein_length": source_summary.get("protein_length", len(protein)),
        "source_checkpoint_dir": str(ckpt_dir.relative_to(REPO_ROOT)),
        "created_at": datetime.now().isoformat(),
        "decoding_formula": {
            "original": "score = Q + explicit_metric_terms",
            "scaled": "score = K * Q + explicit_metric_terms",
        },
        "decoding_config": {
            "k_values": k_values,
            "alpha_cai": args.alpha_cai,
            "alpha_csc": args.alpha_csc,
            "alpha_gc": args.alpha_gc,
            "alpha_u": args.alpha_u,
            "alpha_mfe": args.alpha_mfe,
            "target_gc": args.target_gc,
            "target_u": args.target_u,
            "device": device,
            "csc_file": str(args.csc_file),
            "source_checkpoint": ckpt_path.name,
            "source_training_summary": summary_path.name,
            "script": str(Path(__file__).resolve().relative_to(REPO_ROOT)),
        },
        "scaling_experiments": scaling_experiments,
    }

    output_path = ckpt_dir / args.summary_name
    with output_path.open("w") as handle:
        json.dump(output, handle, indent=2)
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run scaled-Q CodonRL decoding for checkpoint folders.")
    parser.add_argument("--ckpt-root", type=Path, default=REPO_ROOT / "checkpoints")
    parser.add_argument("--indices", type=int, nargs="*", default=None)
    parser.add_argument("--k-values", type=float, nargs="+", default=[1.0, 20.0, 50.0, 100.0])
    parser.add_argument("--alpha-cai", type=float, default=2.5)
    parser.add_argument("--alpha-csc", type=float, default=0.0)
    parser.add_argument("--alpha-gc", type=float, default=0.0)
    parser.add_argument("--alpha-u", type=float, default=0.0)
    parser.add_argument("--alpha-mfe", type=float, default=0.0)
    parser.add_argument("--target-gc", type=float, default=None)
    parser.add_argument("--target-u", type=float, default=None)
    parser.add_argument("--csc-file", type=Path, default=REPO_ROOT / "config" / "csc.json")
    parser.add_argument("--device", default=("cuda:0" if torch.cuda.is_available() else "cpu"))
    parser.add_argument("--summary-name", default="scaling_decoding_summary.json")
    parser.add_argument("--manifest-path", type=Path, default=REPO_ROOT / "codex_work" / "scaled_decoding_manifest.json")
    parser.add_argument("--include-step-trace", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    csc_weights = load_csc_weights(args.csc_file)
    outputs = []
    failures = []
    try:
        for ckpt_dir in checkpoint_dirs(args.ckpt_root, args.indices):
            summary_path = ckpt_dir / "training_summary.json"
            ckpt_path = ckpt_dir / "ckpt_best_objective.pth"
            if not summary_path.exists() or not ckpt_path.exists():
                failures.append({"checkpoint_dir": str(ckpt_dir), "reason": "missing summary or checkpoint"})
                continue
            try:
                output = run_for_checkpoint(ckpt_dir, args.k_values, args, csc_weights)
                outputs.append(
                    {
                        "task_id": output["task_id"],
                        "checkpoint_dir": str(ckpt_dir.relative_to(REPO_ROOT)),
                        "summary_path": str((ckpt_dir / args.summary_name).relative_to(REPO_ROOT)),
                    }
                )
                print(f"Wrote {ckpt_dir / args.summary_name}")
            except Exception as exc:
                failures.append({"checkpoint_dir": str(ckpt_dir), "reason": repr(exc)})
                print(f"Failed {ckpt_dir}: {exc}", file=sys.stderr)
    finally:
        shutdown_mfe_calculator()

    manifest = {
        "created_at": datetime.now().isoformat(),
        "device": args.device,
        "k_values": args.k_values,
        "decode_weights": {
            "alpha_cai": args.alpha_cai,
            "alpha_csc": args.alpha_csc,
            "alpha_gc": args.alpha_gc,
            "alpha_u": args.alpha_u,
            "alpha_mfe": args.alpha_mfe,
            "target_gc": args.target_gc,
            "target_u": args.target_u,
        },
        "completed": outputs,
        "failures": failures,
    }
    args.manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with args.manifest_path.open("w") as handle:
        json.dump(manifest, handle, indent=2)

    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
