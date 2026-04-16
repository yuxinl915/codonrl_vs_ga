# -*- coding: utf-8 -*-
# """
# Codon Optimization using CodonRl with Milestone Rewards.
# Modified for MFE ablation study - LinearFold only with Vienna comparison
#
# Enhanced with comprehensive training history recording and visualization.
#
# Optimized with:
# > epsilon for short sequences
# > Multi-GPU parallel computation for improved throughput.
# > Multi-parallel computation for improved throughput.
# > Flash-Attention-2 & Automatic Mixed Precision (AMP) for improved throughput.
# > Cached Protein Transformer Encoder to reduce redundant computations.
# > Asynchronous MFE calculation with LRU cache to reduce CPU wait time.
# > Complete training history tracking and visualization.
# > MFE method selection for ablation studies
#
# Allows selection of codon usage tables, processes multiple tasks from a JSON file,
# and supports replay buffer pre-population from a guide mRNA.
# """

# --- Core Imports ---
import multiprocessing as mp
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import deque, defaultdict
import math
import time
import argparse
import os
import json
import csv
from typing import Optional, Dict, List, Tuple, Set, Any
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor, Future

# --- Plotting ---
try:
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("WARNING: matplotlib not found. Training curves will not be generated.")

# --- Weights & Biases Integration ---
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("WARNING: wandb library not found. Logging disabled.")

# --- External Tool Integration (ViennaRNA & LinearFold) ---
try:
    import RNA
    VIENNA_RNA_AVAILABLE = True
except ImportError:
    VIENNA_RNA_AVAILABLE = False
    print("\nWARNING: ViennaRNA library not found. Final MFE calculation disabled.\n")

try:
    import linearfold
    LINEARFOLD_AVAILABLE = True
except ImportError:
    LINEARFOLD_AVAILABLE = False
    print("\nWARNING: linearfold-unofficial library not found. Milestone MFE will be disabled.\n")

# --- Data Definitions ---
AMINO_ACIDS: str = "ACDEFGHIKLMNPQRSTVWY*"
AA_TO_INT: Dict[str, int] = {aa: i for i, aa in enumerate(AMINO_ACIDS)}
INT_TO_AA: Dict[int, str] = {i: aa for aa, i in AA_TO_INT.items()}
N_AMINO_ACIDS: int = len(AMINO_ACIDS)

CODONS: List[str] = sorted([
    'UUU', 'UUC', 'UUA', 'UUG', 'UCU', 'UCC', 'UCA', 'UCG', 'AGU', 'AGC',
    'UAU', 'UAC', 'UAA', 'UAG', 'UGA', 'UGU', 'UGC', 'UGG', 'CUU', 'CUC',
    'CUA', 'CUG', 'CCU', 'CCC', 'CCA', 'CCG', 'CAU', 'CAC', 'CAA', 'CAG',
    'CGU', 'CGC', 'CGA', 'CGG', 'AGA', 'AGG', 'AUU', 'AUC', 'AUA', 'AUG',
    'ACU', 'ACC', 'ACA', 'ACG', 'AAU', 'AAC', 'AAA', 'AAG', 'GUU', 'GUC',
    'GUA', 'GUG', 'GCU', 'GCC', 'GCA', 'GCG', 'GAU', 'GAC', 'GAA', 'GAG',
    'GGU', 'GGC', 'GGA', 'GGG'
])
CODON_TO_INT: Dict[str, int] = {codon: i for i, codon in enumerate(CODONS)}
INT_TO_CODON: Dict[int, str] = {i: codon for codon, i in CODON_TO_INT.items()}
N_CODONS: int = len(CODONS)
PAD_CODON_IDX: int = N_CODONS

AA_TO_CODONS: Dict[str, List[str]] = {
    'F': ['UUU', 'UUC'], 'L': ['UUA', 'UUG', 'CUU', 'CUC', 'CUA', 'CUG'], 'I': ['AUU', 'AUC', 'AUA'], 'M': ['AUG'],
    'V': ['GUU', 'GUC', 'GUA', 'GUG'], 'S': ['UCU', 'UCC', 'UCA', 'UCG', 'AGU', 'AGC'], 'P': ['CCU', 'CCC', 'CCA', 'CCG'],
    'T': ['ACU', 'ACC', 'ACA', 'ACG'], 'A': ['GCU', 'GCC', 'GCA', 'GCG'], 'Y': ['UAU', 'UAC'], '*': ['UAA', 'UAG', 'UGA'],
    'H': ['CAU', 'CAC'], 'Q': ['CAA', 'CAG'], 'N': ['AAU', 'AAC'], 'K': ['AAA', 'AAG'], 'D': ['GAU', 'GAC'],
    'E': ['GAA', 'GAG'], 'C': ['UGU', 'UGC'], 'W': ['UGG'], 'R': ['CGU', 'CGC', 'CGA', 'CGG', 'AGA', 'AGG'],
    'G': ['GGU', 'GGC', 'GGA', 'GGG']
}
CODON_TO_AA: Dict[str, str] = {codon: aa for aa, codon_list in AA_TO_CODONS.items() for codon in codon_list}

# --- Codon Usage Tables ---
TARGET_W_TABLE: Dict[str, float] = {}

MFE_CALCULATOR: Optional['AsyncMFECalculator'] = None
MFE_MAX_WORKERS: int = 4

def configure_target_w_table(table: Dict[str, float]):
    global TARGET_W_TABLE
    TARGET_W_TABLE = table

def set_mfe_max_workers(value: Optional[int]):
    global MFE_MAX_WORKERS
    if value and value > 0:
        shutdown_mfe_calculator()
        MFE_MAX_WORKERS = value

def get_mfe_calculator() -> 'AsyncMFECalculator':
    global MFE_CALCULATOR
    if MFE_CALCULATOR is None:
        MFE_CALCULATOR = AsyncMFECalculator(max_workers=MFE_MAX_WORKERS)
    return MFE_CALCULATOR

def shutdown_mfe_calculator():
    global MFE_CALCULATOR
    if MFE_CALCULATOR is not None:
        MFE_CALCULATOR.shutdown()
        MFE_CALCULATOR = None

ECOLLI_K12_FREQ_PER_THOUSAND: Dict[str, float] = {
    'UUU': 19.9, 'UUC': 14.8, 'UUA': 13.8, 'UUG': 13.0, 'UCU': 9.0, 'UCC': 8.9, 'UCA': 7.6, 'UCG': 9.1, 'UAU': 11.8, 'UAC': 8.7, 'UAA': 2.0, 'UAG': 0.3, 'UGU': 5.3, 'UGC': 6.7, 'UGA': 1.0, 'UGG': 14.0, 'CUU': 12.1, 'CUC': 11.6, 'CUA': 4.0, 'CUG': 51.7, 'CCU': 6.7, 'CCC': 5.1, 'CCA': 8.0, 'CCG': 24.2, 'CAU': 13.0, 'CAC': 8.8, 'CAA': 13.8, 'CAG': 28.8, 'CGU': 17.8, 'CGC': 19.6, 'CGA': 3.4, 'CGG': 6.0, 'AUU': 27.6, 'AUC': 21.8, 'AUA': 4.9, 'AUG': 27.8, 'ACU': 9.4, 'ACC': 22.4, 'ACA': 6.9, 'ACG': 15.7, 'AAU': 18.9, 'AAC': 21.7, 'AAA': 34.5, 'AAG': 11.8, 'AGU': 9.0, 'AGC': 15.7, 'AGA': 2.2, 'AGG': 1.2, 'GUU': 17.7, 'GUC': 14.1, 'GUA': 10.4, 'GUG': 22.4, 'GCU': 15.8, 'GCC': 25.8, 'GCA': 19.8, 'GCG': 34.2, 'GAU': 33.0, 'GAC': 19.4, 'GAA': 49.6, 'GAG': 19.8, 'GGU': 26.5, 'GGC': 29.4, 'GGA': 8.3, 'GGG': 11.2
}
HUMAN_FREQ_PER_THOUSAND: Dict[str, float] = {
    'UUU': 17.6, 'UUC': 20.3, 'UUA': 7.7,  'UUG': 12.9, 'UCU': 15.2, 'UCC': 17.7, 'UCA': 12.2, 'UCG': 4.4,  'UAU': 12.2, 'UAC': 15.3, 'UAA': 1.0,  'UAG': 0.8, 'UGU': 10.6, 'UGC': 12.6, 'UGA': 1.6,  'UGG': 13.2, 'CUU': 13.2, 'CUC': 19.6, 'CUA': 7.2,  'CUG': 39.6, 'CCU': 17.5, 'CCC': 19.9, 'CCA': 16.9, 'CCG': 6.9, 'CAU': 10.9, 'CAC': 15.1, 'CAA': 12.3, 'CAG': 34.2, 'CGU': 4.5,  'CGC': 10.4, 'CGA': 6.2,  'CGG': 11.4, 'AUU': 16.0, 'AUC': 20.8, 'AUA': 7.5,  'AUG': 22.0, 'ACU': 13.1, 'ACC': 18.9, 'ACA': 15.1, 'ACG': 6.1,  'AAU': 17.0, 'AAC': 19.1, 'AAA': 24.4, 'AAG': 31.9, 'AGU': 12.1, 'AGC': 19.3, 'AGA': 12.2, 'AGG': 12.0, 'GUU': 11.0, 'GUC': 14.5, 'GUA': 7.1,  'GUG': 28.1, 'GCU': 18.4, 'GCC': 27.7, 'GCA': 15.8, 'GCG': 7.4,  'GAU': 21.8, 'GAC': 25.1, 'GAA': 29.0, 'GAG': 39.6, 'GGU': 10.8, 'GGC': 22.2, 'GGA': 16.5, 'GGG': 16.5
}

# --- Asynchronous MFE Calculator with method selection ---
class AsyncMFECalculator:
    def __init__(self, max_workers: int = 4):
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        print(f"Async MFE Calculator initialized with {max_workers} workers.")
        # Track MFE calculations for analysis
        self.vienna_calls = 0
        self.linearfold_calls = 0
        self.vienna_times = []
        self.linearfold_times = []

    @staticmethod
    @lru_cache(maxsize=4096)
    def _vienna_fold(mrna_sequence: str) -> float:
        if not VIENNA_RNA_AVAILABLE or not mrna_sequence: return 0.0
        try:
            _, mfe_val = RNA.fold(mrna_sequence)
            return float(mfe_val) if isinstance(mfe_val, (int, float)) else 0.0
        except Exception: return 0.0

    @staticmethod
    @lru_cache(maxsize=8192)
    def _linear_fold(mrna_sequence: str) -> float:
        if not LINEARFOLD_AVAILABLE or not mrna_sequence or len(mrna_sequence) < 4: return 0.0
        try:
            _, mfe_val = linearfold.fold(mrna_sequence) # type: ignore
            return float(mfe_val) if isinstance(mfe_val, (int, float)) else 0.0
        except Exception: return 0.0

    def calculate_vienna_async(self, mrna_sequence: str) -> Future:
        self.vienna_calls += 1
        start_time = time.time()
        future = self.executor.submit(self._vienna_fold, mrna_sequence.upper().replace('T', 'U'))
        # Track timing (note: this is approximate due to async nature)
        self.vienna_times.append(time.time() - start_time)
        return future

    def calculate_linearfold_async(self, mrna_sequence: str) -> Future:
        self.linearfold_calls += 1
        start_time = time.time()
        future = self.executor.submit(self._linear_fold, mrna_sequence.upper().replace('T', 'U'))
        self.linearfold_times.append(time.time() - start_time)
        return future
    
    def calculate_mfe_async(self, mrna_sequence: str, method: str = "vienna") -> Future:
        """Unified MFE calculation with method selection"""
        if method == "linearfold":
            return self.calculate_linearfold_async(mrna_sequence)
        else:  # default to vienna
            return self.calculate_vienna_async(mrna_sequence)

    def get_stats(self):
        """Get statistics about MFE calculations"""
        return {
            "vienna_calls": self.vienna_calls,
            "linearfold_calls": self.linearfold_calls,
            "avg_vienna_time": np.mean(self.vienna_times) if self.vienna_times else 0,
            "avg_linearfold_time": np.mean(self.linearfold_times) if self.linearfold_times else 0
        }

    def shutdown(self):
        print("Shutting down MFE calculator thread pool...")
        print(f"Statistics: {self.get_stats()}")
        self.executor.shutdown()

# --- Objective/Helper Functions ---
def load_codon_frequency_table(filepath: str) -> Optional[Dict[str, float]]:
    try:
        freq_table: Dict[str, float] = {}
        with open(filepath, 'r') as f:
            for line in f:
                parts = line.strip().upper().split()
                if len(parts) == 2 and parts[0] in CODONS:
                    try:
                        freq_table[parts[0].replace('T', 'U')] = float(parts[1])
                    except ValueError: continue
        for codon in CODONS: freq_table.setdefault(codon, 0.0)
        return freq_table
    except FileNotFoundError:
        print(f"Error: Codon frequency file not found: {filepath}"); return None

def calculate_relative_adaptiveness(aa_to_codons_map: Dict[str, List[str]], codon_freq_table: Dict[str, float]) -> Dict[str, float]:
    relative_adaptiveness: Dict[str, float] = {}
    aa_max_freq: Dict[str, float] = defaultdict(float)
    for aa, codons in aa_to_codons_map.items():
        if aa == '*': continue
        max_f: float = max((codon_freq_table.get(c, 0.0) for c in codons), default=0.0)
        aa_max_freq[aa] = max_f
    for codon in CODONS:
        aa = CODON_TO_AA.get(codon)
        if aa and aa != '*':
            max_f = aa_max_freq.get(aa, 0.0)
            relative_adaptiveness[codon] = (codon_freq_table.get(codon, 0.0) / max_f) if max_f > 1e-9 else 0.0
        else: relative_adaptiveness[codon] = 0.0
    return relative_adaptiveness

def calculate_cai(mrna_sequence_str: str, w_table: Dict[str, float]) -> float:
    if not mrna_sequence_str or not w_table: return 0.0
    log_sum_w, num_codons = 0.0, 0
    for i in range(0, len(mrna_sequence_str), 3):
        codon = mrna_sequence_str[i:i+3]
        if len(codon) < 3 or CODON_TO_AA.get(codon) == '*': break
        w = w_table.get(codon, 0.0)
        if w > 1e-9:
            log_sum_w += math.log(w)
            num_codons += 1
        else: return 0.0
    return math.exp(log_sum_w / num_codons) if num_codons > 0 else 0.0

def calculate_log_codon_weights_sum(mrna_sequence_str: str, w_table: Dict[str, float]) -> float:
    total_log_weight: float = 0.0
    for i in range(0, len(mrna_sequence_str), 3):
        codon = mrna_sequence_str[i:i+3]
        if len(codon) < 3 or CODON_TO_AA.get(codon) == '*': break
        w = w_table.get(codon, 0.0)
        if w <= 1e-9: return -float('inf')
        total_log_weight += math.log(w)
    return total_log_weight

def calculate_gc_content(sequence_str: str) -> float:
    if not sequence_str: return 0.0
    gc_count = sequence_str.count('G') + sequence_str.count('C')
    return (gc_count / len(sequence_str)) * 100.0 if len(sequence_str) > 0 else 0.0

def calculate_objective(mrna_sequence: str, config: Dict[str, Any]) -> Tuple[float, float]:
    """Calculate objective with configurable MFE method"""
    lambda_val = config.get('lambda_val', 0.0)
    mfe_method = config.get('final_mfe_method', 'vienna')  # Default to vienna for backward compatibility
    
    mfe_future = get_mfe_calculator().calculate_mfe_async(mrna_sequence, method=mfe_method)
    log_sum_w = calculate_log_codon_weights_sum(mrna_sequence, TARGET_W_TABLE)
    mfe = mfe_future.result()
    
    if log_sum_w == -float('inf'): return float('inf'), mfe
    return mfe - (lambda_val * log_sum_w), mfe

def calculate_milestone_objective(partial_mrna: str, config: Dict[str, Any]) -> float:
    """Calculate milestone objective with configurable MFE method"""
    current_length = len(partial_mrna)
    if current_length < 4: return 0.0
    
    lambda_val = config.get('lambda_val', 0.0)
    # Default to linearfold for backward compatibility
    mfe_method = config.get('milestone_mfe_method', 'linearfold')  
    
    mfe_future = get_mfe_calculator().calculate_mfe_async(partial_mrna, method=mfe_method)
    cai_partial = calculate_cai(partial_mrna, TARGET_W_TABLE)
    mfe_partial = mfe_future.result()
    
    mfe_per_len = mfe_partial / current_length
    log_cai_term = math.log(cai_partial) if cai_partial > 1e-9 else -20.0
    if lambda_val > 1e-9 and cai_partial <= 1e-9: return float('inf')
    milestone_obj = mfe_per_len - (lambda_val * log_cai_term)
    return milestone_obj if math.isfinite(milestone_obj) else float('inf')

def calculate_comparison_mfe(mrna_sequence: str, primary_method: str) -> Dict[str, float]:
    """Calculate MFE using both methods for comparison"""
    results = {}
    
    # Get primary MFE
    if primary_method == "linearfold":
        primary_future = get_mfe_calculator().calculate_linearfold_async(mrna_sequence)
        results['linearfold_mfe'] = primary_future.result()
        
        # Also calculate Vienna for comparison if available
        if VIENNA_RNA_AVAILABLE:
            vienna_future = get_mfe_calculator().calculate_vienna_async(mrna_sequence)
            results['vienna_mfe'] = vienna_future.result()
            results['mfe_difference'] = abs(results['linearfold_mfe'] - results['vienna_mfe'])
            results['mfe_relative_difference'] = results['mfe_difference'] / (abs(results['vienna_mfe']) + 1e-6)
    else:
        primary_future = get_mfe_calculator().calculate_vienna_async(mrna_sequence)
        results['vienna_mfe'] = primary_future.result()
        
        # Also calculate LinearFold for comparison if available
        if LINEARFOLD_AVAILABLE:
            linear_future = get_mfe_calculator().calculate_linearfold_async(mrna_sequence)
            results['linearfold_mfe'] = linear_future.result()
            results['mfe_difference'] = abs(results['vienna_mfe'] - results['linearfold_mfe'])
            results['mfe_relative_difference'] = results['mfe_difference'] / (abs(results['vienna_mfe']) + 1e-6)
    
    return results

# --- Network Architectures (unchanged) ---
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 2000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

class CachedAttentionQNetwork(nn.Module):
    def __init__(self, n_amino_acids_vocab: int, n_codons_vocab: int, d_model: int, nhead: int, num_encoder_layers: int, num_decoder_layers: int, dim_feedforward: int, dropout: float):
        super().__init__()
        self.d_model = d_model
        self.aa_embedding = nn.Embedding(n_amino_acids_vocab, d_model, padding_idx=n_amino_acids_vocab - 1)
        self.codon_embedding = nn.Embedding(n_codons_vocab, d_model, padding_idx=n_codons_vocab - 1)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, F.gelu, batch_first=True, norm_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers, nn.LayerNorm(d_model))
        decoder_layer = nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout, F.gelu, batch_first=True, norm_first=True)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_decoder_layers, nn.LayerNorm(d_model))
        self.output_head = nn.Linear(d_model, N_CODONS)
        print("Initialized CachedAttentionQNetwork (Encoder-Decoder). PyTorch >= 2.0 will enable Flash Attention automatically.")

    def encode_protein(self, protein_seq_int: torch.Tensor, protein_pad_mask: torch.Tensor) -> torch.Tensor:
        protein_embed = self.aa_embedding(protein_seq_int) * math.sqrt(self.d_model)
        protein_embed = self.pos_encoder(protein_embed)
        return self.transformer_encoder(protein_embed, src_key_padding_mask=protein_pad_mask)

    def decode_mrna(self, partial_mrna_int: torch.Tensor, current_pos_in_protein: torch.Tensor, memory: torch.Tensor, memory_pad_mask: torch.Tensor) -> torch.Tensor:
        mrna_embed = self.codon_embedding(partial_mrna_int) * math.sqrt(self.d_model)
        mrna_embed = self.pos_encoder(mrna_embed)
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(partial_mrna_int.size(1)).to(partial_mrna_int.device)
        output = self.transformer_decoder(mrna_embed, memory, tgt_mask=tgt_mask, memory_key_padding_mask=memory_pad_mask)
        batch_size = output.size(0)
        gather_idx = current_pos_in_protein.view(batch_size, 1, 1).expand(-1, -1, self.d_model)
        last_codon_output = torch.gather(output, 1, gather_idx).squeeze(1)
        return self.output_head(last_codon_output)

# --- Replay Buffer (unchanged) ---
class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buffer: deque = deque(maxlen=capacity)
    def push(self, state: Dict, action: int, reward: float, next_state: Optional[Dict], done: bool):
        state_cpu = {k: v.cpu() for k, v in state.items()}
        next_state_cpu = {k: v.cpu() for k, v in next_state.items()} if next_state else None
        self.buffer.append((state_cpu, action, reward, next_state_cpu, done))
    def sample(self, batch_size: int, device: torch.device) -> Optional[Tuple[Dict, torch.Tensor, torch.Tensor, Dict, torch.Tensor]]:
        if len(self.buffer) < batch_size: return None
        transitions = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*transitions)
        state_batch: Dict[str, torch.Tensor] = {key: torch.cat([s[key] for s in states]).to(device) for key in states[0]}
        non_final_mask = torch.tensor([s is not None for s in next_states], device=device, dtype=torch.bool)
        non_final_next_states = [s for s in next_states if s is not None]
        next_state_batch: Dict[str, torch.Tensor] = {}
        if non_final_next_states:
             next_state_batch = {key: torch.cat([s[key] for s in non_final_next_states]).to(device) for key in non_final_next_states[0]}
        action_batch = torch.tensor(actions, dtype=torch.long, device=device)
        reward_batch = torch.tensor(rewards, dtype=torch.float, device=device)
        return state_batch, action_batch, reward_batch, next_state_batch, non_final_mask
    def __len__(self) -> int: return len(self.buffer)


class CodonRL:
    def __init__(self, agent_config: Dict[str, Any]):
        self.config = agent_config
        self.device = torch.device(self.config["device"])
        self.use_amp = self.config.get("use_amp", False) and "cuda" in self.device.type
        self.aa_vocab_size, self.aa_pad_idx = N_AMINO_ACIDS + 1, N_AMINO_ACIDS
        self.codon_vocab_size, self.codon_pad_idx = N_CODONS + 1, N_CODONS

        self.policy_net = CachedAttentionQNetwork(self.aa_vocab_size, self.codon_vocab_size, self.config["embedding_dim"], self.config["n_head"], self.config["n_encoder_layer"], self.config["n_decoder_layer"], self.config["transformer_dim_feedforward"], self.config["transformer_dropout"]).to(self.device)
        self.target_net = CachedAttentionQNetwork(self.aa_vocab_size, self.codon_vocab_size, self.config["embedding_dim"], self.config["n_head"], self.config["n_encoder_layer"], self.config["n_decoder_layer"], self.config["transformer_dim_feedforward"], self.config["transformer_dropout"]).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict()); self.target_net.eval()
        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=self.config["learning_rate"], weight_decay=self.config.get("adamw_weight_decay", 0.01))
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)
        self.memory = ReplayBuffer(self.config["buffer_size"])
        self.steps_done = 0
        self.protein_memory_cache, self.protein_pad_mask_cache = None, None

    def _precompute_protein_memory(self, protein_str: str):
        if not self.config.get("use_protein_cache", True): return
        p_max_len = self.config["protein_max_len"]
        protein_int = [AA_TO_INT.get(aa, self.aa_pad_idx) for aa in protein_str]
        protein_int += [self.aa_pad_idx] * (p_max_len - len(protein_int))
        protein_tensor = torch.tensor([protein_int], dtype=torch.long, device=self.device)
        self.protein_pad_mask_cache = (protein_tensor == self.aa_pad_idx)
        with torch.no_grad(), torch.amp.autocast('cuda', enabled=self.use_amp):
            self.protein_memory_cache = self.policy_net.encode_protein(protein_tensor, self.protein_pad_mask_cache)
        print("Protein context memory has been pre-computed and cached.")

    def _get_state(self, partial_mrna_str: str, current_aa_index: int) -> Dict[str, torch.Tensor]:
        m_max_len = self.config["protein_max_len"]
        mrna_int = [CODON_TO_INT.get(partial_mrna_str[i:i+3], self.codon_pad_idx) for i in range(0, len(partial_mrna_str), 3)]
        mrna_int += [self.codon_pad_idx] * (m_max_len - len(mrna_int))
        mrna_tensor = torch.tensor([mrna_int], dtype=torch.long, device=self.device)
        pos_tensor = torch.tensor([current_aa_index], dtype=torch.long, device=self.device)
        return {"mrna": mrna_tensor, "pos": pos_tensor}

    def select_action(self, state: Dict, current_amino_acid: str) -> Tuple[int, str, float]:
        eps_threshold = self.config["eps_end"] + (self.config["eps_start"] - self.config["eps_end"]) * math.exp(-1. * self.steps_done / self.config["eps_decay"])
        self.steps_done += 1
        possible_codons = AA_TO_CODONS[current_amino_acid]
        possible_codon_indices = [CODON_TO_INT[c] for c in possible_codons]
        if random.random() > eps_threshold:
            self.policy_net.eval()
            with torch.no_grad(), torch.amp.autocast('cuda', enabled=self.use_amp):
                q_values = self.policy_net.decode_mrna(state["mrna"], state["pos"], self.protein_memory_cache, self.protein_pad_mask_cache)
                mask = torch.full_like(q_values, -float('inf'))
                mask[0, possible_codon_indices] = 0.0
                action_idx = (q_values + mask).argmax(dim=1).item()
            self.policy_net.train()
        else:
            action_idx = random.choice(possible_codon_indices)
        return action_idx, INT_TO_CODON[action_idx], eps_threshold

    def optimize_model(self) -> Optional[float]:
        if len(self.memory) < self.config["batch_size"]: return None
        state_batch, action_batch, reward_batch, next_state_batch, non_final_mask = self.memory.sample(self.config["batch_size"], self.device)
        self.optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast('cuda', enabled=self.use_amp):
            protein_memory = self.protein_memory_cache.expand(self.config["batch_size"], -1, -1)
            protein_mask = self.protein_pad_mask_cache.expand(self.config["batch_size"], -1)
            current_q_all = self.policy_net.decode_mrna(state_batch["mrna"], state_batch["pos"], protein_memory, protein_mask)
            current_q_selected = current_q_all.gather(1, action_batch.unsqueeze(1)).squeeze(1)
            next_state_q_values = torch.zeros(self.config["batch_size"], device=self.device)
            if non_final_mask.any():
                with torch.no_grad():
                    num_non_final = next_state_batch["mrna"].size(0)
                    protein_memory_nf = self.protein_memory_cache.expand(num_non_final, -1, -1)
                    protein_mask_nf = self.protein_pad_mask_cache.expand(num_non_final, -1)
                    target_next_q_all = self.target_net.decode_mrna(next_state_batch["mrna"], next_state_batch["pos"], protein_memory_nf, protein_mask_nf)
                    max_next_q = target_next_q_all.max(dim=1)[0]
                    next_state_q_values[non_final_mask] = max_next_q.to(next_state_q_values.dtype)

            target_q_selected = reward_batch + (self.config["gamma"] * next_state_q_values)
            loss = F.smooth_l1_loss(current_q_selected, target_q_selected)
        self.scaler.scale(loss).backward()
        if self.config.get("gradient_clipping_norm", 0) > 0:
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.config["gradient_clipping_norm"])
        self.scaler.step(self.optimizer)
        self.scaler.update()
        return loss.item()

# --- Helper functions for saving and visualization ---
def save_sequence_to_fasta(filepath: str, header_info: str, sequence: str):
    try:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f: f.write(f">{header_info}\n{sequence}\n")
        print(f"Saved sequence to: {filepath}")
    except Exception as e: print(f"Error saving sequence to {filepath}: {e}")

def save_checkpoint(filepath: str, state_dict: Optional[Dict]):
    if state_dict is None: return
    try:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        torch.save(state_dict, filepath)
        print(f"Saved checkpoint to: {filepath}")
    except Exception as e: print(f"Error saving checkpoint to {filepath}: {e}")

def save_training_history_csv(filepath: str, history: Dict[str, List]):
    """Save training history to CSV file"""
    try:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=history.keys())
            writer.writeheader()
            for i in range(len(history['episodes'])):
                row = {key: history[key][i] for key in history.keys()}
                writer.writerow(row)
        print(f"Saved training history to: {filepath}")
    except Exception as e:
        print(f"Error saving training history: {e}")

def save_training_summary_json(filepath: str, results: Dict, protein_sequence: str, task_id: str):
    """Save comprehensive training summary to JSON"""
    summary = {
        'task_id': task_id,
        'protein_sequence': protein_sequence,
        'protein_length': len(protein_sequence),
        'training_time': results.get('training_time', 0),
        'num_episodes': len(results['training_history']['episodes']) if 'training_history' in results else 0,
        'mfe_method_config': results.get('mfe_method_config', {}),
        'best_objective': {
            'score': results['best_objective']['score'],
            'sequence': results['best_objective']['mrna'],
            'episode': results['best_objective'].get('episode', 0),
            'gc_content': calculate_gc_content(results['best_objective']['mrna']) if results['best_objective']['mrna'] else 0,
            'mfe_comparison': results['best_objective'].get('mfe_comparison', {})
        },
        'best_mfe': {
            'score': results['best_mfe']['score'],
            'sequence': results['best_mfe']['mrna'],
            'cai': results['best_mfe']['cai'],
            'episode': results['best_mfe'].get('episode', 0),
            'gc_content': calculate_gc_content(results['best_mfe']['mrna']) if results['best_mfe']['mrna'] else 0,
            'mfe_comparison': results['best_mfe'].get('mfe_comparison', {})
        },
        'best_cai': {
            'score': results['best_cai']['score'],
            'sequence': results['best_cai']['mrna'],
            'mfe': results['best_cai']['mfe'],
            'episode': results['best_cai'].get('episode', 0),
            'gc_content': calculate_gc_content(results['best_cai']['mrna']) if results['best_cai']['mrna'] else 0,
            'mfe_comparison': results['best_cai'].get('mfe_comparison', {})
        },
        'final_metrics': {
            'objective': results['training_history']['objectives'][-1] if 'training_history' in results else 0,
            'mfe': results['training_history']['mfes'][-1] if 'training_history' in results else 0,
            'cai': results['training_history']['cais'][-1] if 'training_history' in results else 0
        },
        'mfe_calculator_stats': results.get('mfe_calculator_stats', {}),
        'config': results.get('config', {})
    }
    
    try:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"Saved training summary to: {filepath}")
    except Exception as e:
        print(f"Error saving summary: {e}")

def plot_training_curves(history: Dict, output_path: str, task_id: str):
    """Generate and save training curves"""
    if not MATPLOTLIB_AVAILABLE:
        print("Matplotlib not available, skipping plot generation")
        return
        
    try:
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle(f'Training Progress for {task_id}', fontsize=16)
        
        # Objective curve
        axes[0, 0].plot(history['episodes'], history['objectives'], label='Current', alpha=0.6)
        axes[0, 0].plot(history['episodes'], history['best_objective_history'], label='Best', linewidth=2)
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Objective')
        axes[0, 0].set_title('Objective Score Over Training')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # MFE curve
        axes[0, 1].plot(history['episodes'], history['mfes'], label='Current', alpha=0.6)
        axes[0, 1].plot(history['episodes'], history['best_mfe_history'], label='Best', linewidth=2)
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('MFE (kcal/mol)')
        axes[0, 1].set_title('Minimum Free Energy Over Training')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # CAI curve
        axes[0, 2].plot(history['episodes'], history['cais'], label='Current', alpha=0.6)
        axes[0, 2].plot(history['episodes'], history['best_cai_history'], label='Best', linewidth=2)
        axes[0, 2].set_xlabel('Episode')
        axes[0, 2].set_ylabel('CAI')
        axes[0, 2].set_title('Codon Adaptation Index Over Training')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
        
        # Loss curve
        axes[1, 0].plot(history['episodes'], history['losses'])
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].set_title('Training Loss Over Time')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Epsilon curve
        axes[1, 1].plot(history['episodes'], history['epsilons'])
        axes[1, 1].set_xlabel('Episode')
        axes[1, 1].set_ylabel('Epsilon')
        axes[1, 1].set_title('Exploration Rate Over Time')
        axes[1, 1].grid(True, alpha=0.3)
        
        # GC Content
        axes[1, 2].plot(history['episodes'], history['gc_contents'])
        axes[1, 2].set_xlabel('Episode')
        axes[1, 2].set_ylabel('GC Content (%)')
        axes[1, 2].set_title('GC Content Over Training')
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved training curves to: {output_path}")
    except Exception as e:
        print(f"Error saving plot: {e}")
        if 'fig' in locals():
            plt.close()

# --- Training Loop with History Recording and MFE method selection ---
def train_with_guidance(protein_sequence: str, agent_config: Dict[str, Any], guide_mrna: Optional[str] = None) -> Tuple[Optional[CodonRL], Dict[str, Any]]:
    agent = CodonRL(agent_config)
    start_time = time.time()
    run_id_str = agent_config.get('run_id_str', 'unknown_task')
    
    # Print MFE method configuration
    milestone_method = agent_config.get('milestone_mfe_method', 'linearfold')
    final_method = agent_config.get('final_mfe_method', 'vienna')
    print(f"MFE Configuration - Milestone: {milestone_method}, Final: {final_method}")
    
    best_objective_score, best_objective_mrna, best_objective_episode = float('inf'), "", 0
    best_mfe_score, best_mfe_mrna, best_mfe_cai, best_mfe_episode = float('inf'), "", 0.0, 0
    best_cai_score, best_cai_mrna, best_cai_mfe, best_cai_episode = 0.0, "", float('inf'), 0
    
    # Initialize training history
    training_history = {
        'episodes': [],
        'objectives': [],
        'mfes': [],
        'cais': [],
        'losses': [],
        'epsilons': [],
        'timestamps': [],
        'gc_contents': [],
        'best_objective_history': [],
        'best_mfe_history': [],
        'best_cai_history': []
    }

    run = None
    if WANDB_AVAILABLE and agent_config.get("wandb_log", False):
        try:
            if wandb.run is not None: wandb.finish()
            config_with_mfe = {**agent_config, 
                             "milestone_mfe_method": milestone_method,
                             "final_mfe_method": final_method}
            run = wandb.init(project=agent_config["wandb_project"], 
                           name=agent_config.get("wandb_run_name"), 
                           config=config_with_mfe, 
                           reinit=True)
            print(f"--- wandb logging initialized for Task {run_id_str} (Run Name: {run.name}) ---")
        except Exception as e: 
            print(f"WARNING (Task {run_id_str}): Failed to initialize wandb: {e}.")

    protein_len = len(protein_sequence)
    if protein_len == 0: return None, {}
    agent._precompute_protein_memory(protein_sequence)
    milestones = {math.floor(p * protein_len) - 1 for p in [0.25, 0.5, 0.75]}
    milestone_reward_weight = agent_config.get('milestone_reward_weight', 0.1)

    if agent_config.get("prepopulate_buffer") and guide_mrna:
        print(f"Pre-populating buffer for Task {run_id_str} with {protein_len} guide transitions...")
        guide_codons = [guide_mrna[i:i+3] for i in range(0, len(guide_mrna), 3)]
        temp_mrna = ""
        final_objective_guide, _ = calculate_objective(guide_mrna, agent_config)
        final_reward_guide = -final_objective_guide if math.isfinite(final_objective_guide) else -1e9
        for t in range(protein_len):
            state_t = agent._get_state(temp_mrna, t)
            action_codon = guide_codons[t]
            action_idx = CODON_TO_INT.get(action_codon, -1)
            if action_idx == -1: continue
            temp_mrna += action_codon
            done_t = (t == protein_len - 1)
            reward_t = agent_config.get('guide_reward_bonus', 0.0)
            if t in milestones:
                milestone_obj = calculate_milestone_objective(temp_mrna, agent_config)
                if math.isfinite(milestone_obj):
                    reward_t += -milestone_obj * milestone_reward_weight
            if done_t:
                reward_t += final_reward_guide
            next_state_t = agent._get_state(temp_mrna, t + 1) if not done_t else None
            agent.memory.push(state_t, action_idx, reward_t, next_state_t, done_t)
        print(f"Buffer pre-populated. Current size: {len(agent.memory)}")

    print(f"\nStarting training for Task {run_id_str}, protein '{protein_sequence[:30]}...' (Len: {protein_len})")
    for i_episode in range(agent_config['num_episodes']):
        episode_start_time = time.time()
        partial_mrna, loss_list = "", []
        final_objective, final_mfe, final_cai, current_eps = float('inf'), float('inf'), 0.0, 1.0

        for t in range(protein_len):
            state = agent._get_state(partial_mrna, t)
            action_idx, action_codon, current_eps = agent.select_action(state, protein_sequence[t])
            partial_mrna += action_codon
            done = (t == protein_len - 1)
            step_reward = 0.0
            if guide_mrna and action_codon == guide_mrna[t*3 : t*3+3]:
                step_reward += agent_config.get('guide_reward_bonus', 0.0)
            if t in milestones:
                milestone_obj = calculate_milestone_objective(partial_mrna, agent_config)
                if milestone_obj != float('inf'):
                    step_reward += -milestone_obj * milestone_reward_weight
            if done:
                final_objective, final_mfe = calculate_objective(partial_mrna, agent_config)
                final_cai = calculate_cai(partial_mrna, TARGET_W_TABLE)
                step_reward += -final_objective if final_objective != float('inf') else -1e9
            next_state = agent._get_state(partial_mrna, t + 1) if not done else None
            agent.memory.push(state, action_idx, step_reward, next_state, done)
            loss = agent.optimize_model()
            if loss is not None: loss_list.append(loss)
            if done: break
            
        avg_loss = np.mean(loss_list) if loss_list else 0.0
        gc_content = calculate_gc_content(partial_mrna) if len(partial_mrna) == protein_len * 3 else 0.0
        
        # Record episode metrics
        training_history['episodes'].append(i_episode + 1)
        training_history['objectives'].append(final_objective)
        training_history['mfes'].append(final_mfe)
        training_history['cais'].append(final_cai)
        training_history['losses'].append(avg_loss)
        training_history['epsilons'].append(current_eps)
        training_history['timestamps'].append(time.time() - start_time)
        training_history['gc_contents'].append(gc_content)
        training_history['best_objective_history'].append(best_objective_score)
        training_history['best_mfe_history'].append(best_mfe_score)
        training_history['best_cai_history'].append(best_cai_score)
        
        if len(partial_mrna) == protein_len * 3:
            if final_objective < best_objective_score:
                best_objective_score, best_objective_mrna, best_objective_episode = final_objective, partial_mrna, i_episode + 1
                print(f"💡 (Task {run_id_str}) New best Objective: {best_objective_score:.4f} at Ep {i_episode+1}")
            if final_mfe < best_mfe_score:
                best_mfe_score, best_mfe_mrna, best_mfe_cai, best_mfe_episode = final_mfe, partial_mrna, final_cai, i_episode + 1
                print(f"🧬 (Task {run_id_str}) New best MFE: {best_mfe_score:.4f} (CAI: {best_mfe_cai:.3f}) at Ep {i_episode+1}")
            if final_cai > best_cai_score:
                best_cai_score, best_cai_mrna, best_cai_mfe, best_cai_episode = final_cai, partial_mrna, final_mfe, i_episode + 1
                print(f"📈 (Task {run_id_str}) New best CAI: {best_cai_score:.4f} (MFE: {best_cai_mfe:.2f}) at Ep {i_episode+1}")
        
        print(f"(Task {run_id_str}) Ep {i_episode+1}/{agent_config['num_episodes']} | Obj: {final_objective:.2f} (Best: {best_objective_score:.2f}) | MFE: {final_mfe:.2f} (Best: {best_mfe_score:.2f}) | CAI: {final_cai:.3f} (Best: {best_cai_score:.3f}) | Eps: {current_eps:.3f} | Loss: {avg_loss:.4f} | Time: {(time.time() - episode_start_time):.2f}s")
        
        if run:
            run.log({"episode": i_episode + 1, "avg_loss": avg_loss, "epsilon": current_eps, 
                    "final_objective": final_objective, "final_mfe": final_mfe, "final_cai": final_cai, 
                    "best_objective_so_far": best_objective_score, "best_mfe_so_far": best_mfe_score, 
                    "best_cai_so_far": best_cai_score, "gc_content": gc_content})
        
        if i_episode > 0 and i_episode % agent_config["target_update_freq"] == 0:
            agent.target_net.load_state_dict(agent.policy_net.state_dict())
    
    # After training, calculate MFE comparisons for best sequences
    print(f"\nCalculating MFE comparisons for best sequences...")
    
    best_objective_comparison = calculate_comparison_mfe(best_objective_mrna, final_method) if best_objective_mrna else {}
    best_mfe_comparison = calculate_comparison_mfe(best_mfe_mrna, final_method) if best_mfe_mrna else {}
    best_cai_comparison = calculate_comparison_mfe(best_cai_mrna, final_method) if best_cai_mrna else {}
    
    # Print comparison results
    if best_mfe_comparison and 'vienna_mfe' in best_mfe_comparison and 'linearfold_mfe' in best_mfe_comparison:
        print(f"Best MFE sequence - LinearFold: {best_mfe_comparison['linearfold_mfe']:.2f}, Vienna: {best_mfe_comparison['vienna_mfe']:.2f}, Diff: {best_mfe_comparison['mfe_difference']:.2f}")
    
    print(f"\nTraining complete for Task {run_id_str}. Total time: {(time.time() - start_time):.2f} seconds")
    
    # Get MFE calculator statistics
    mfe_stats = get_mfe_calculator().get_stats()
    
    best_results = {
        "best_objective": {
            "score": best_objective_score,
            "mrna": best_objective_mrna,
            "episode": best_objective_episode,
            "mfe_comparison": best_objective_comparison,
            "model_state_dict": agent.policy_net.state_dict()
        },
        "best_mfe": {
            "score": best_mfe_score,
            "mrna": best_mfe_mrna,
            "cai": best_mfe_cai,
            "episode": best_mfe_episode,
            "mfe_comparison": best_mfe_comparison,
            "model_state_dict": agent.policy_net.state_dict()
        },
        "best_cai": {
            "score": best_cai_score,
            "mrna": best_cai_mrna,
            "mfe": best_cai_mfe,
            "episode": best_cai_episode,
            "mfe_comparison": best_cai_comparison,
            "model_state_dict": agent.policy_net.state_dict()
        },
        "training_history": training_history,
        "training_time": time.time() - start_time,
        "mfe_method_config": {
            "milestone_mfe_method": milestone_method,
            "final_mfe_method": final_method
        },
        "mfe_calculator_stats": mfe_stats,
        "config": agent_config
    }
    
    if run: run.finish()
    return agent, best_results

# --- Helper Functions for JSON and FASTA Reading ---
def load_protein_guide_pairs_from_json(filepath: str) -> List[Dict[str, str]]:
    try:
        with open(filepath, 'r') as f: data = json.load(f)
        pairs = []
        for i, item in enumerate(data):
            if isinstance(item, dict) and "protein_sequence" in item:
                pairs.append({"protein_sequence": item["protein_sequence"], "guide_mrna_sequence": item.get("mrna_sequence"), "id": str(item.get("seqn", f"item_{i+1}"))})
        print(f"Successfully loaded {len(pairs)} tasks from {filepath}.")
        return pairs
    except Exception as e: print(f"Error reading JSON file {filepath}: {e}"); return []

def read_fasta(filepath: str) -> Optional[Tuple[str, str]]:
    try:
        with open(filepath, 'r') as f:
            header, seq = None, ""
            for line in f:
                line = line.strip()
                if line.startswith('>'):
                    if header is not None: break
                    header = line[1:]
                elif header is not None: seq += line.upper()
            return (header, "".join(seq.split())) if header and seq else None
    except Exception as e: print(f"Error reading FASTA file {filepath}: {e}"); return None

def translate_mrna(mrna_seq: str) -> str:
    protein = [CODON_TO_AA.get(mrna_seq[i:i+3], '?') for i in range(0, len(mrna_seq), 3)]
    try:
        stop_index = protein.index('*')
        protein = protein[:stop_index]
    except ValueError: pass
    return "".join(protein)

def run_single_task(
    task_idx: int,
    total_tasks: int,
    task_content: Dict[str, str],
    base_config: Dict[str, Any],
    args_dict: Dict[str, Any],
    target_w_table: Dict[str, float],
    is_parallel: bool = False,
    run_suffix: Optional[str] = None,
):
    configure_target_w_table(target_w_table)
    current_task_config = base_config.copy()
    task_id = str(task_content['id']).replace(" ", "_").replace("/", "_")
    prefix = f"\n{'='*25} Starting Task {task_idx + 1}/{total_tasks} (ID: {task_id}) {'='*25}"
    print(prefix)

    target_protein = "".join(filter(lambda aa: aa in AA_TO_INT, task_content["protein_sequence"].upper()))
    if len(target_protein) > current_task_config['protein_max_len']:
        target_protein = target_protein[:current_task_config['protein_max_len']]
        print(f"Warning: Truncated protein to max length {current_task_config['protein_max_len']}.")
    if not target_protein:
        print(f"Error: Protein sequence for task {task_id} is empty. Skipping."); return

    protein_len = len(target_protein)
    current_task_config['eps_decay'] = int(protein_len * current_task_config['eps_decay_factor'])
    print(f"Protein length: {protein_len}, eps_decay_factor: {current_task_config['eps_decay_factor']:.1f} -> Calculated eps_decay: {current_task_config['eps_decay']}")

    guide_mrna = task_content.get("guide_mrna_sequence")
    if guide_mrna:
        guide_mrna = guide_mrna.upper().replace("T", "U")
        if len(guide_mrna) != len(target_protein) * 3 or translate_mrna(guide_mrna) != target_protein:
            print(f"Warning (Task {task_id}): Guide mRNA is invalid or does not translate to the target protein. Disabling guide.")
            guide_mrna = None
        else: print(f"Validated guide mRNA for Task {task_id}.")

    current_task_config['run_id_str'] = task_id
    if current_task_config["wandb_log"]:
        suffix = run_suffix or "main"
        mfe_suffix = f"{current_task_config.get('milestone_mfe_method', 'lf')}-{current_task_config.get('final_mfe_method', 'vn')}"
        current_task_config["wandb_run_name"] = f"{args_dict['wandb_run_name_prefix']}-{task_id}-{suffix}-{mfe_suffix}"
        if args_dict.get("wandb_group"):
            current_task_config["wandb_group"] = args_dict["wandb_group"]

    agent, best_results = train_with_guidance(protein_sequence=target_protein, agent_config=current_task_config, guide_mrna=guide_mrna)
    
    if agent and best_results:
        # Create output directory with MFE method in name
        mfe_method_str = f"{current_task_config.get('milestone_mfe_method', 'linearfold')}_{current_task_config.get('final_mfe_method', 'vienna')}"
        task_output_dir = os.path.join(args_dict['output_dir'], f"{task_id}_{mfe_method_str}")
        os.makedirs(task_output_dir, exist_ok=True)
        print(f"\n--- Saving results for Task {task_id} to '{task_output_dir}' ---")
        
        # Save training history CSV
        if 'training_history' in best_results:
            history_csv_path = os.path.join(task_output_dir, "training_history.csv")
            save_training_history_csv(history_csv_path, best_results['training_history'])
            
            # Save training curves plot
            plot_path = os.path.join(task_output_dir, "training_curves.png")
            plot_training_curves(best_results['training_history'], plot_path, task_id)
        
        # Save comprehensive JSON summary
        summary_json_path = os.path.join(task_output_dir, "training_summary.json")
        save_training_summary_json(summary_json_path, best_results, target_protein, task_id)

        # Save best objective results
        res_obj = best_results["best_objective"]
        if res_obj["mrna"]:
            _, mfe_val = calculate_objective(res_obj["mrna"], current_task_config)
            cai_val = calculate_cai(res_obj["mrna"], TARGET_W_TABLE)
            comparison_info = ""
            if res_obj.get('mfe_comparison') and 'vienna_mfe' in res_obj['mfe_comparison']:
                comparison_info = f" LinearFold={res_obj['mfe_comparison'].get('linearfold_mfe', 'N/A'):.2f} Vienna={res_obj['mfe_comparison'].get('vienna_mfe', 'N/A'):.2f}"
            header = f"{task_id}_best_objective | Score={res_obj['score']:.4f} MFE={mfe_val:.2f} CAI={cai_val:.3f} Episode={res_obj.get('episode', 0)}{comparison_info}"
            save_sequence_to_fasta(os.path.join(task_output_dir, "best_objective.fasta"), header, res_obj["mrna"])
            save_checkpoint(os.path.join(task_output_dir, "ckpt_best_objective.pth"), res_obj["model_state_dict"])
        
        # Save best MFE results with comparison
        res_mfe = best_results["best_mfe"]
        if res_mfe["mrna"]:
            comparison_info = ""
            if res_mfe.get('mfe_comparison') and 'vienna_mfe' in res_mfe['mfe_comparison']:
                comparison_info = f" | LinearFold={res_mfe['mfe_comparison'].get('linearfold_mfe', 'N/A'):.2f} Vienna={res_mfe['mfe_comparison'].get('vienna_mfe', 'N/A'):.2f}"
            header = f"{task_id}_best_mfe | MFE={res_mfe['score']:.4f} CAI={res_mfe['cai']:.3f} Episode={res_mfe.get('episode', 0)}{comparison_info}"
            save_sequence_to_fasta(os.path.join(task_output_dir, "best_mfe.fasta"), header, res_mfe["mrna"])

        # Save best CAI results
        res_cai = best_results["best_cai"]
        if res_cai["mrna"]:
            comparison_info = ""
            if res_cai.get('mfe_comparison') and 'vienna_mfe' in res_cai['mfe_comparison']:
                comparison_info = f" | LinearFold={res_cai['mfe_comparison'].get('linearfold_mfe', 'N/A'):.2f} Vienna={res_cai['mfe_comparison'].get('vienna_mfe', 'N/A'):.2f}"
            header = f"{task_id}_best_cai | CAI={res_cai['score']:.4f} MFE={res_cai['mfe']:.2f} Episode={res_cai.get('episode', 0)}{comparison_info}"
            save_sequence_to_fasta(os.path.join(task_output_dir, "best_cai.fasta"), header, res_cai["mrna"])

def worker_main(
    device_label: str,
    worker_rank: int,
    task_bundle: List[Tuple[int, int, Dict[str, str]]],
    base_config: Dict[str, Any],
    args_dict: Dict[str, Any],
    target_w_table: Dict[str, float],
    mfe_workers: Optional[int],
):
    configure_target_w_table(target_w_table)
    set_mfe_max_workers(mfe_workers)
    device_to_use = device_label if device_label != "cuda" else "cuda:0"
    if device_to_use.startswith("cuda") and not torch.cuda.is_available():
        print(f"Warning: CUDA device '{device_to_use}' not available in worker. Falling back to CPU.")
        device_to_use = "cpu"
    for bundle_idx, (task_idx, total_tasks, task) in enumerate(task_bundle):
        worker_config = base_config.copy()
        worker_config["device"] = device_to_use
        run_single_task(
            task_idx,
            total_tasks,
            task,
            worker_config,
            args_dict,
            target_w_table,
            is_parallel=True,
            run_suffix=f"worker{worker_rank}-job{bundle_idx}"
        )
    shutdown_mfe_calculator()

# --- Main Execution Block ---
if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    parser = argparse.ArgumentParser(description="Optimized Codon Design using CodonRL with MFE ablation support.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    input_group = parser.add_argument_group('Input Sequences')
    protein_input_group = input_group.add_mutually_exclusive_group(required=False)
    protein_input_group.add_argument("-p", "--protein_seq", type=str, help="Target amino acid sequence string.")
    protein_input_group.add_argument("-pf", "--protein_file", type=str, help="Path to FASTA file with target amino acid sequence.")
    guide_input_group = input_group.add_mutually_exclusive_group(required=False)
    guide_input_group.add_argument("--guide_mrna_seq", type=str, default=None, help="Optional: Guide mRNA sequence string.")
    guide_input_group.add_argument("--guide_mrna_file", type=str, default=None, help="Optional: Path to FASTA file with guide mRNA.")
    input_group.add_argument("-jf", "--json_input_file", type=str, default=None, help="Path to JSON file with protein sequences and optional guide mRNAs.")

    model_group = parser.add_argument_group('Model Architecture')
    model_group.add_argument("--embedding_dim", type=int, default=128)
    model_group.add_argument("--n_head", type=int, default=8)
    model_group.add_argument("--n_encoder_layer", type=int, default=3, help="Number of Transformer encoder layers.")
    model_group.add_argument("--n_decoder_layer", type=int, default=3, help="Number of Transformer decoder layers.")
    model_group.add_argument("--transformer_dim_feedforward", type=int, default=512)
    model_group.add_argument("--transformer_dropout", type=float, default=0.1)

    train_group = parser.add_argument_group('Training Parameters')
    train_group.add_argument("-e", "--num_episodes", type=int, default=250)
    train_group.add_argument("--protein_max_len", type=int, default=700)
    train_group.add_argument("--learning_rate", type=float, default=5e-5)
    train_group.add_argument("--adamw_weight_decay", type=float, default=0.01)
    train_group.add_argument("--gradient_clipping_norm", type=float, default=1.0)
    train_group.add_argument("--batch_size", type=int, default=64)
    train_group.add_argument("--buffer_size", type=int, default=10000)
    train_group.add_argument("--gamma", type=float, default=0.99)
    train_group.add_argument("--eps_start", type=float, default=1.0)
    train_group.add_argument("--eps_end", type=float, default=0.01)
    train_group.add_argument("--eps_decay_factor", type=float, default=50.0, help="Factor to multiply by protein length to get the epsilon decay rate. Higher value means slower decay.")
    train_group.add_argument("--target_update_freq", type=int, default=50)

    guidance_group = parser.add_argument_group('Guidance Parameters')
    guidance_group.add_argument("--prepopulate_buffer", action='store_true', help="Enable pre-populating the replay buffer with the guide mRNA.")
    guidance_group.add_argument("--guide_reward_bonus", type=float, default=0.5, help="Reward bonus for choosing a codon from the guide mRNA.")
    
    reward_group = parser.add_argument_group('Reward & Objective')
    reward_group.add_argument("--lambda_val", type=float, default=4.0)
    reward_group.add_argument("--milestone_reward_weight", type=float, default=0.3)

    # MFE Method Selection for Ablation
    mfe_group = parser.add_argument_group('MFE Method Selection (for ablation studies)')
    mfe_group.add_argument("--milestone_mfe_method", type=str, default="linearfold", 
                          choices=["linearfold", "vienna"], 
                          help="MFE calculation method for milestones")
    mfe_group.add_argument("--final_mfe_method", type=str, default="vienna", 
                          choices=["linearfold", "vienna"], 
                          help="MFE calculation method for final objective")

    optim_group = parser.add_argument_group('Performance Optimizations')
    optim_group.add_argument("--device", type=str, default="auto", help="Device ('cuda', 'cpu', 'auto').")
    optim_group.add_argument("--use_amp", action='store_true', help="Enable Automatic Mixed Precision (AMP) for training.")
    optim_group.add_argument("--no_protein_cache", action='store_false', dest='use_protein_cache', help="Disable the protein encoder cache.")

    parallel_group = parser.add_argument_group('Parallel Execution')
    parallel_group.add_argument("--parallel_devices", type=str, default="", help="Comma-separated device list for parallel execution, e.g. 'cuda:0,cuda:1'.")
    parallel_group.add_argument("--max_workers", type=int, default=None, help="Cap the number of parallel worker processes (<= number of devices).")
    parallel_group.add_argument("--mfe_workers", type=int, default=4, help="Thread-pool size per process for MFE calculations.")

    io_group = parser.add_argument_group('Input/Output & Logging')
    io_group.add_argument("--codon_table", type=str, default="human", help="Codon usage table: 'human', 'ecolik12', or path to file.")
    io_group.add_argument("--output_dir", type=str, default="results", help="Directory to save all output files.")
    io_group.add_argument("--wandb_log", action='store_true', help="Enable Weights & Biases logging.")
    io_group.add_argument("--wandb_project", type=str, default="CodonRL")
    io_group.add_argument("--wandb_run_name_prefix", type=str, default="opt-run", help="Prefix for Wandb run names.")
    
    args = parser.parse_args()
    config_template = vars(args).copy()

    # Add MFE method configuration
    config_template["milestone_mfe_method"] = args.milestone_mfe_method
    config_template["final_mfe_method"] = args.final_mfe_method

    if args.device == "auto": 
        config_template["device"] = "cuda" if torch.cuda.is_available() else "cpu"
    if "cuda" in config_template["device"] and not torch.cuda.is_available():
        print(f"Warning: CUDA device '{config_template['device']}' not available. Falling back to CPU.")
        config_template["device"] = "cpu"
    
    print(f"--- Global Settings ---\nUsing device: {config_template['device']}")
    print(f"MFE Methods - Milestone: {config_template['milestone_mfe_method']}, Final: {config_template['final_mfe_method']}")
    if config_template.get('use_amp'): 
        print("Automatic Mixed Precision (AMP): Enabled")
    if config_template.get('use_protein_cache'): 
        print("Protein Encoder Cache: Enabled")

    codon_table_source = args.codon_table.lower()
    if codon_table_source == "ecolik12": 
        target_freq_table = ECOLLI_K12_FREQ_PER_THOUSAND
    elif codon_table_source == "human": 
        target_freq_table = HUMAN_FREQ_PER_THOUSAND
    else:
        target_freq_table = load_codon_frequency_table(args.codon_table)
        if target_freq_table is None: 
            exit(1)
    
    target_w_table = calculate_relative_adaptiveness(AA_TO_CODONS, target_freq_table)
    configure_target_w_table(target_w_table)
    set_mfe_max_workers(args.mfe_workers)

    print(f"Using codon table: {codon_table_source}")
    
    tasks_to_process: List[Dict[str, str]] = []
    if args.json_input_file:
        tasks_to_process = load_protein_guide_pairs_from_json(args.json_input_file)
    else:
        prot_record = read_fasta(args.protein_file) if args.protein_file else None
        prot_str = args.protein_seq or (prot_record[1] if prot_record else None)
        if not prot_str: 
            print("Error: No protein sequence provided. Exiting.")
            exit(1)
        guide_record = read_fasta(args.guide_mrna_file) if args.guide_mrna_file else None
        guide_str = args.guide_mrna_seq or (guide_record[1] if guide_record else None)
        tasks_to_process.append({"protein_sequence": prot_str, "guide_mrna_sequence": guide_str, "id": "cli_task"})

    task_packages = [(idx, len(tasks_to_process), task) for idx, task in enumerate(tasks_to_process)]
    args_dict = {
        "output_dir": args.output_dir,
        "wandb_run_name_prefix": args.wandb_run_name_prefix,
        "wandb_log": args.wandb_log,
        "wandb_group": os.getenv("WANDB_RUN_GROUP")
    }

    parallel_devices = [dev.strip() for dev in args.parallel_devices.split(",") if dev.strip()]
    if not parallel_devices:
        for package in task_packages:
            run_single_task(package[0], package[1], package[2], config_template, args_dict, target_w_table, is_parallel=False)
        shutdown_mfe_calculator()
    else:
        if args.max_workers:
            active_devices = parallel_devices[:max(1, min(args.max_workers, len(parallel_devices)))]
        else:
            active_devices = parallel_devices
        print(f"Launching parallel execution across devices: {active_devices}")
        workers: List[mp.Process] = []
        for worker_idx, device_label in enumerate(active_devices):
            bundled_tasks = task_packages[worker_idx::len(active_devices)]
            if not bundled_tasks:
                continue
            process = mp.Process(
                target=worker_main,
                args=(device_label, worker_idx, bundled_tasks, config_template, args_dict, target_w_table, args.mfe_workers),
                daemon=False
            )
            process.start()
            workers.append(process)
        for proc in workers:
            proc.join()
        shutdown_mfe_calculator()
    
    print("\nScript finished processing all tasks.")