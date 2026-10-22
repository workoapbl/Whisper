#!/bin/bash
#SBATCH --job-name=sae
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=results/slurm/%x-%j.out

# SAE-calibrated pruning on Llama-3.2-1B. Run from the repository root:
#
#   mkdir -p results/slurm                                 (once, Slurm won't create it)
#   sbatch sae/run.sh baseline [ratio] [type] [seed]       dense and SparseGPT perplexity
#   sbatch sae/run.sh smoke                                short SAE run on layers 0 and 1
#   sbatch sae/run.sh train [sae.train args]               SAEs for every layer
#   sbatch sae/run.sh check                                per-layer FVU of the trained SAEs
#   sbatch sae/run.sh prune [ratio] [type] [seed]          SAE-SparseGPT perplexity
#
# ratio defaults to 0.5, type to unstructured (or n:m, e.g. 2:4), seed to 0.
# Only training needs the H200; other stages fit on an L40S (sbatch --gres=gpu:l40s:1 ...).

set -euo pipefail

MODEL=meta-llama/Llama-3.2-1B
SAE_PATH=$HOME/orcd/pool/sae/llama-3.2-1b
SMOKE_PATH=$HOME/orcd/scratch/sae/llama-3.2-1b-smoke
RESULTS=results/Llama-3.2-1B

STAGE=$1
shift

cd "$SLURM_SUBMIT_DIR"
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate whisper

echo "stage $STAGE on $(hostname), job ${SLURM_JOB_ID:-none}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# Runs one pruning method through main.py, saving under results/ like the README examples.
prune() {
    local method=$1 ratio=$2 type=$3 seed=$4
    shift 4
    local tag=$ratio
    if [ "$type" != "unstructured" ]; then
        tag=${type/:/-}
    fi
    python main.py --model "$MODEL" --prune_method "$method" --seed "$seed" \
        --sparsity_ratio "$ratio" --sparsity_type "$type" \
        --save "$RESULTS/$method/sparsity_$tag" "$@"
}

case $STAGE in
    baseline)
        prune sparsegpt 0 unstructured 0
        prune sparsegpt "${1:-0.5}" "${2:-unstructured}" "${3:-0}"
        ;;
    smoke)
        python -m sae.train --model "$MODEL" --output_path "$SMOKE_PATH" \
            --layers 0 1 --training_tokens 2000000
        ;;
    train)
        python -m sae.train --model "$MODEL" --output_path "$SAE_PATH" "$@"
        ;;
    check)
        python -m sae.check --model "$MODEL" --sae_path "$SAE_PATH"
        ;;
    prune)
        prune sae-sparsegpt "${1:-0.5}" "${2:-unstructured}" "${3:-0}" --sae_path "$SAE_PATH"
        ;;
    *)
        echo "unknown stage $STAGE, expected baseline, smoke, train, check or prune"
        exit 1
        ;;
esac
