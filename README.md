
# CodonRL


## Installation

### Prerequisites

- **OS**: Linux (tested on Ubuntu 20.04+)
- **GPU**: NVIDIA GPU with CUDA 12.1+ support (recommended: H100, A100 for training)
- **Conda**: Anaconda or Miniconda

### Setup
```bash
# Clone the repository
git clone git@github.com:Kingsford-Group/codonrl.git

cd codonrl

# Create conda environment
conda env create -f environment.yml

# Activate environment
conda activate codonrl
```

## Checkpoints

We provide 55 model checkpoints trained on different protein sequences from the UniProt dataset.

### Download All Checkpoints

Download all 55 checkpoints using the provided script:

```bash

# Download all checkpoints
bash download_checkpoint.sh
```

---

## Quick Start

After downloading checkpoints, run inference with pre-trained models:

```bash
# Simple: test 5 alpha values (1.0, 1.5, 2.0, 2.5, 3.0)
bash run_decoding_multiobjective.sh

# Advanced: multi-objective optimization (e.g., U content minimization)
bash run_decoding_multiobjective_multialpha.sh
```

---

## Inference & Benchmarking

### Configuration

Before running, edit paths in the scripts:

```bash
OUT_DIR=./benchmark_multialpha
CSV_PATH=./datasets/gemorna_with_all_metrics.csv
CKPT_ROOT=./checkpoints
```

**Important**: Update the path in `visualizeandbenchmark_multialpha.py` and `visualizeandbenchmark.py`:

```python
sys.path.append('/path/to/codonrl')  # Change to  actual CodonRL path
```

### 1. Simple Multi-Objective Benchmarking

**Script**: `run_decoding_multiobjective.sh`

Tests 5 different alpha values with balanced evaluation weights.

```bash
bash run_decoding_multiobjective.sh
```

**Monitor progress:**
```bash
tail -f benchmark_multialpha/logs/*.log
```

**Outputs** (for each alpha, e.g., `alpha=2.5`):
- `*.csv` - Detailed metrics (CAI, MFE, CSC, GC, U)
- `*_rna.fasta` / `*_dna.fasta` - Generated mRNA sequences
- `*_viz.png` - Parity plots
- `summary.txt` - Quick statistics

### 2. Advanced Multi-Alpha Benchmarking

**Script**: `run_decoding_multiobjective_multialpha.sh`

Fine-grained control over multiple optimization objectives.

**Current experiment**: U content minimization
```bash
bash run_decoding_multiobjective_multialpha.sh
```

**Key parameters:**
- `--alpha_cai` : CAI weight
- `--alpha_csc` : Codon stability coefficient
- `--alpha_gc` : GC content weight
- `--alpha_u` : U content weight (negative = minimize)
- `--target_gc` / `--target_u`: Target content values

**Custom experiments:**

```bash
# CAI optimization
python visualizeandbenchmark_multialpha.py \
  --alpha_cai 2.5 --run_name "high_cai" ...

# GC content targeting (55%)
python visualizeandbenchmark_multialpha.py \
  --alpha_gc 0.5 --target_gc 0.55 --run_name "gc_target" ...

# Multi-objective balance
python visualizeandbenchmark_multialpha.py \
  --alpha_cai 1.0 --alpha_csc 0.3 --alpha_gc 0.2 --alpha_u -0.3 \
  --run_name "balanced" ...
```

---

## Training (For Advanced Users)

Want to train on your own data? Follow this guide.

### Basic Usage

```bash
# Batch training from JSON file
python CodonRL_main.py --jf datasets/proteins.json
```

### Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--jf` / `--json_input_file` | Required | Training dataset (JSON format) |
| `--codon_table` | `human` | Codon table: `human` or `ecolik12` |
| `--lambda_val` | `4.0` | CAI-MFE tradeoff (0=MFE only, 10=CAI priority) |
| `--protein_max_len` | `700` | Maximum protein length to process |
| `--batch_size` | `64` | Training batch size |
| `-e` / `--num_episodes` | `250` | Total training episodes |
| `--learning_rate` | `5e-5` | Learning rate |
| `--buffer_size` | `10000` | Experience replay buffer size |
| `--target_update_freq` | `50` | Target network update frequency (steps) |
| `--max_workers` | `None` | Max parallel workers (auto-detected if None) |
| `--mfe_workers` | `4` | Thread-pool size per process for MFE calculations |
| `--milestone_mfe_method` | `linearfold` | MFE method during training: `linearfold` or `vienna` |
| `--final_mfe_method` | `vienna` | MFE method for final evaluation |
| `--output_dir` | `results` | Directory for checkpoints and logs |

### GPU Configuration

**Single GPU (50 workers):**
```bash
export DEVICES=$(python3 -c "print(','.join(['cuda:0']*50))")
```

**Multi-GPU (e.g., 4 GPUs with 12-13 workers each):**
```bash
export DEVICES=$(python3 -c "
devices = []
for i in range(4):
    devices.extend([f'cuda:{i}']*13)
print(','.join(devices[:50]))
")
```

**Manual configuration:**
```bash
export DEVICES="cuda:0,cuda:0,cuda:1,cuda:1,cuda:2,cuda:2,cuda:3,cuda:3"
```

### Optional Flags

```bash
--use_amp                    # Enable automatic mixed precision (recommended for modern GPUs)
--prepopulate_buffer         # Pre-fill replay buffer before training starts
--wandb_log                  # Enable Weights & Biases logging
--wandb_project <name>       # W&B project name
--wandb_run_name_prefix <p>  # W&B run name prefix for experiment tracking
```

### Production Example

```bash
DEVICES=$(python3 -c "print(','.join(['cuda:0']*50))")

nohup python CodonRL_main.py \
  --jf ./datasets/uniprot_le_500/uniprot_with_guidance_l0.json \
  --codon_table human \
  --lambda_val 4 \
  --protein_max_len 501 \
  --batch_size 64 \
  -e 500 \
  --buffer_size 100000 \
  --learning_rate 2e-5 \
  --target_update_freq 150 \
  --parallel_devices $DEVICES \
  --max_workers 55 \
  --mfe_workers 4 \
  --milestone_mfe_method linearfold \
  --final_mfe_method linearfold \
  --use_amp \
  --prepopulate_buffer \
  --wandb_log \
  --wandb_project CodonRL \
  --wandb_run_name_prefix run \
  --output_dir results
```


## Citation

TBD

---

## License

This project is licensed under the [CodonRL Software License Agreement](./license.txt).

---
## Acknowledgements

TBD

---

## Contributing

TBD

---

## Contact

TBD
