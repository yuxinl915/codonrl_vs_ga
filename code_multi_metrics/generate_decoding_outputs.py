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

REPO_ROOT = Path(__file__).resolve().parents[2]
CODEX_WORK = REPO_ROOT / "codex_work"
OUT_ROOT = CODEX_WORK / "agent4_outputs"
LOG_DIR = CODEX_WORK / "logs"
TMP_DIR = CODEX_WORK / "tmp"
os.environ.setdefault("MPLCONFIGDIR", str(TMP_DIR / "mplconfig"))

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from CodonRL_main import (  # noqa: E402
    AA_TO_CODONS,
    CODON_TO_INT,
    CodonRL,
    ECOLLI_K12_FREQ_PER_THOUSAND,
    HUMAN_FREQ_PER_THOUSAND,
    calculate_relative_adaptiveness,
    configure_target_w_table,
    shutdown_mfe_calculator,
)


EXPECTED_CONFIG = {
    "alpha_cai": 2.5,
    "alpha_csc": 1.0,
    "alpha_gc": 1.0,
    "alpha_u": 1.0,
    "alpha_mfe": 0.0,
    "target_gc": None,
    "target_u": None,
}


def to_rna(seq: str) -> str:
    return (seq or "").strip().upper().replace("T", "U")


def to_dna(seq: str) -> str:
    return (seq or "").strip().upper().replace("U", "T")


def gc_fraction(seq: str) -> float:
    seq = (seq or "").strip().upper()
    if not seq:
        return math.nan
    return (seq.count("G") + seq.count("C")) / len(seq)


def u_fraction(seq: str) -> float:
    seq = to_rna(seq)
    if not seq:
        return math.nan
    return seq.count("U") / len(seq)


def split_codons(seq: str) -> list[str]:
    seq = to_rna(seq)
    if not seq or len(seq) % 3 != 0:
        return []
    return [seq[i : i + 3] for i in range(0, len(seq), 3)]


def load_json(path: Path) -> dict[str, Any]:
    with path.open() as handle:
        return json.load(handle)


def load_csc_weights(csc_path: Path) -> dict[str, float]:
    raw = load_json(csc_path)
    return {to_rna(codon): float(value) for codon, value in raw.items()}


def calculate_csc(mrna: str, csc_weights: dict[str, float]) -> float:
    codons = split_codons(mrna)
    if not codons:
        return math.nan
    values = []
    for codon in codons:
        value = csc_weights.get(codon)
        if value is None:
            return math.nan
        values.append(float(value))
    return sum(values) / len(values)


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


def gc_term(prefix_mrna: str, codon: str, alpha_gc: float, target_gc: float | None) -> float:
    if alpha_gc == 0.0:
        return 0.0
    codon_gc = gc_fraction(codon)
    if target_gc is None:
        return alpha_gc * codon_gc
    current_codons = len(prefix_mrna) / 3
    current_gc = gc_fraction(prefix_mrna) if prefix_mrna else 0.5
    new_gc = ((current_gc * current_codons) + codon_gc) / (current_codons + 1)
    return alpha_gc * (-abs(new_gc - target_gc)) * 10.0


def u_term(prefix_mrna: str, codon: str, alpha_u: float, target_u: float | None) -> float:
    if alpha_u == 0.0:
        return 0.0
    codon_u = u_fraction(codon)
    if target_u is None:
        return alpha_u * (-codon_u)
    current_codons = len(prefix_mrna) / 3
    current_u = u_fraction(prefix_mrna) if prefix_mrna else 0.25
    new_u = ((current_u * current_codons) + codon_u) / (current_codons + 1)
    return alpha_u * (-abs(new_u - target_u)) * 10.0


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
            cai_bonus = alpha_cai * logw.get(codon, 0.0)
            csc_bonus = alpha_csc * logcsc.get(codon, 0.0)
            gc_bonus = gc_term(mrna, codon, alpha_gc, target_gc)
            u_bonus = u_term(mrna, codon, alpha_u, target_u)
            metric_bonus = cai_bonus + csc_bonus + gc_bonus + u_bonus
            total_score = (q_scale * raw_q) + metric_bonus
            candidates.append(
                {
                    "codon": codon,
                    "raw_q": raw_q,
                    "scaled_q": q_scale * raw_q,
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

    num_steps = len(protein)
    return mrna, {
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


def existing_summary_path(ckpt_dir: Path) -> Path:
    return ckpt_dir / "scaling_decoding_summary_cai2p5_all1.json"


def summary_matches(summary: dict[str, Any], k_values: list[float]) -> bool:
    config = summary.get("decoding_config", {})
    for key, expected_value in EXPECTED_CONFIG.items():
        if config.get(key) != expected_value:
            return False
    actual_k = sorted(float(v) for v in config.get("k_values", []))
    expected_k = sorted(float(v) for v in k_values)
    if actual_k != expected_k:
        return False
    experiments = summary.get("scaling_experiments", {})
    for k in k_values:
        if str(float(k)) not in experiments and str(k) not in experiments:
            return False
    return True


def load_existing_summary(ckpt_dir: Path, k_values: list[float]) -> dict[str, Any] | None:
    path = existing_summary_path(ckpt_dir)
    if not path.exists():
        return None
    summary = load_json(path)
    if not summary_matches(summary, k_values):
        return None
    return summary


def generate_summary(
    ckpt_dir: Path,
    k_values: list[float],
    csc_weights: dict[str, float],
    device: str,
) -> dict[str, Any]:
    summary_path = ckpt_dir / "training_summary.json"
    ckpt_path = ckpt_dir / "ckpt_best_objective.pth"
    source_summary = load_json(summary_path)
    cfg, w = load_cfg_and_w(source_summary)
    protein = source_summary["protein_sequence"].strip().upper()
    agent = build_agent(cfg, device)
    state_dict = torch.load(ckpt_path, map_location=device)
    agent.policy_net.load_state_dict(state_dict)
    agent.target_net.load_state_dict(state_dict)
    agent.policy_net.eval()
    agent.target_net.eval()

    experiments: OrderedDict[str, Any] = OrderedDict()
    for k_value in k_values:
        mrna, decode_stats = scaled_multiobjective_decode(
            agent=agent,
            protein=protein,
            w=w,
            csc_weights=csc_weights,
            q_scale=float(k_value),
            alpha_cai=EXPECTED_CONFIG["alpha_cai"],
            alpha_csc=EXPECTED_CONFIG["alpha_csc"],
            alpha_gc=EXPECTED_CONFIG["alpha_gc"],
            alpha_u=EXPECTED_CONFIG["alpha_u"],
            target_gc=EXPECTED_CONFIG["target_gc"],
            target_u=EXPECTED_CONFIG["target_u"],
        )
        experiments[str(float(k_value))] = {
            "task_id": source_summary["task_id"],
            "q_scale": float(k_value),
            "sequence": to_rna(mrna),
            "generated_mrna_sequence": to_rna(mrna),
            "generated_cds_dna_sequence": to_dna(mrna),
            "protein_length": len(protein),
            "score": decode_stats["cumulative_total_score"],
            "decoding_stats": decode_stats,
            "cai": None,
            "csc": calculate_csc(mrna, csc_weights),
            "gc_content": gc_fraction(mrna) * 100.0,
            "gc_content_fraction": gc_fraction(mrna),
            "u_percent": u_fraction(mrna) * 100.0,
            "u_fraction": u_fraction(mrna),
        }

    return {
        "task_id": source_summary["task_id"],
        "protein_sequence": protein,
        "protein_length": len(protein),
        "source_checkpoint_dir": str(ckpt_dir.relative_to(REPO_ROOT)),
        "created_at": datetime.now().isoformat(),
        "decoding_formula": {
            "original": "score = Q + explicit_metric_terms",
            "scaled": "score = K * Q + explicit_metric_terms",
        },
        "decoding_config": {
            "k_values": [float(v) for v in k_values],
            **EXPECTED_CONFIG,
            "device": device,
            "csc_file": str(REPO_ROOT / "config" / "csc.json"),
            "source_checkpoint": "ckpt_best_objective.pth",
            "source_training_summary": "training_summary.json",
            "script": str(Path(__file__).resolve().relative_to(REPO_ROOT)),
        },
        "scaling_experiments": experiments,
    }


def export_per_k_outputs(
    protein_index: int,
    summary: dict[str, Any],
    origin: str,
    source_summary_path: Path | None,
) -> list[dict[str, Any]]:
    exported = []
    task_id = summary["task_id"]
    protein = summary["protein_sequence"]
    for k_str, label in (("1.0", "k1"), ("50.0", "k50")):
        payload = summary["scaling_experiments"][k_str]
        out_dir = OUT_ROOT / label / f"seq_{protein_index}"
        out_dir.mkdir(parents=True, exist_ok=True)
        json_path = out_dir / "summary.json"
        rna_path = out_dir / "predicted_mrna.fasta"
        dna_path = out_dir / "predicted_dna.fasta"
        export_payload = {
            "protein_index": protein_index,
            "task_id": task_id,
            "protein_sequence": protein,
            "source_checkpoint_dir": summary["source_checkpoint_dir"],
            "decoding_formula": summary["decoding_formula"],
            "decoding_config": summary["decoding_config"],
            "output_origin": origin,
            "origin_summary_path": (
                str(source_summary_path.relative_to(REPO_ROOT))
                if source_summary_path is not None and source_summary_path.exists()
                else None
            ),
            "experiment": payload,
        }
        json_path.write_text(json.dumps(export_payload, indent=2) + "\n")
        rna_path.write_text(f">seq_{protein_index}_codonrl_k{int(float(k_str))}\n{payload['generated_mrna_sequence']}\n")
        dna_path.write_text(f">seq_{protein_index}_codonrl_k{int(float(k_str))}_dna\n{payload['generated_cds_dna_sequence']}\n")
        exported.append(
            {
                "protein_index": protein_index,
                "k_label": label,
                "q_scale": float(k_str),
                "json_path": str(json_path.relative_to(REPO_ROOT)),
                "rna_fasta": str(rna_path.relative_to(REPO_ROOT)),
                "dna_fasta": str(dna_path.relative_to(REPO_ROOT)),
                "origin": origin,
            }
        )
    return exported


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate AGENT_4 decoding outputs under codex_work.")
    parser.add_argument("--indices", type=int, nargs="*", default=None)
    parser.add_argument("--device", default=("cuda:0" if torch.cuda.is_available() else "cpu"))
    parser.add_argument("--reuse-existing", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    TMP_DIR.mkdir(parents=True, exist_ok=True)

    k_values = [1.0, 50.0]
    csc_weights = load_csc_weights(REPO_ROOT / "config" / "csc.json")
    if args.indices:
        indices = sorted(args.indices)
    else:
        indices = list(range(1, 56))

    manifest = {
        "created_at": datetime.now().isoformat(),
        "device": args.device,
        "k_values": k_values,
        "decode_weights": EXPECTED_CONFIG,
        "completed": [],
        "failures": [],
    }

    try:
        for protein_index in indices:
            ckpt_dir = REPO_ROOT / "checkpoints" / f"{protein_index}_linearfold_linearfold"
            try:
                if args.reuse_existing:
                    reused = load_existing_summary(ckpt_dir, k_values)
                else:
                    reused = None
                if reused is not None:
                    summary = reused
                    origin = "reused_existing_checkpoint_summary"
                    origin_path = existing_summary_path(ckpt_dir)
                else:
                    summary = generate_summary(ckpt_dir, k_values, csc_weights, args.device)
                    origin = "generated_now"
                    origin_path = None
                exported = export_per_k_outputs(protein_index, summary, origin, origin_path)
                manifest["completed"].append(
                    {
                        "protein_index": protein_index,
                        "task_id": summary["task_id"],
                        "source_checkpoint_dir": summary["source_checkpoint_dir"],
                        "origin": origin,
                        "exports": exported,
                    }
                )
                print(f"Processed seq_{protein_index} ({origin})")
            except Exception as exc:
                manifest["failures"].append(
                    {
                        "protein_index": protein_index,
                        "source_checkpoint_dir": str(ckpt_dir.relative_to(REPO_ROOT)),
                        "reason": repr(exc),
                    }
                )
                print(f"Failed seq_{protein_index}: {exc}", file=sys.stderr)
    finally:
        shutdown_mfe_calculator()

    manifest_path = LOG_DIR / "agent4_generation_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    if manifest["failures"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
