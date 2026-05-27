#!/bin/bash
#SBATCH --job-name=validate_anti_loss
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=2:00:00
#SBATCH --partition=fengl2
set -xeuo pipefail

source /seu_share/home/fenglei/213243847/miniconda3/etc/profile.d/conda.sh
conda activate verl-agent

cd /seu_share2/home/fenglei/213243847/Anti_lossRL


echo "=============================================================================="
echo ""
echo "开始时间: $(date)"

### =========================================================================
### User-adjustable parameters — override via env vars
### =========================================================================

# Model
MODEL_PATH=/seu_share2/home/fenglei/sharedata/Qwen2.5-7B-Instruct

# Data (GSM8K parquet)
GSM8K_TEST_FILE=/seu_share2/home/fenglei/213243847/Anti_lossRL/data/grade-school-math/grade_school_math/data/test.jsonl

# Phase 1 validation knobs
NUM_PROMPTS=${NUM_PROMPTS:-10}
ROLLOUTS_PER_PROMPT=${ROLLOUTS_PER_PROMPT:-8}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-512}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-1024}
TEMPERATURE=${TEMPERATURE:-1.0}
TOP_P=${TOP_P:-0.95}

# Anti-loss
ANTI_MARGIN=${ANTI_MARGIN:-}              # empty = no margin; set to e.g. -4.0 for margin mode

# Gradient / update checks
NUM_GRADIENT_CHECKS=${NUM_GRADIENT_CHECKS:-5}
NUM_UPDATE_CHECKS=${NUM_UPDATE_CHECKS:-3}
UPDATE_LR=${UPDATE_LR:-1e-5}

# Output
OUTPUT_DIR=${OUTPUT_DIR:-./validation_results}
SEED=${SEED:-42}

# Device (auto-detect: cuda if available, else cpu)
DEVICE=${DEVICE:-}

### =========================================================================
### Derived defaults
### =========================================================================

# Resolve device if not explicitly set
if [[ -z "$DEVICE" ]]; then
    if python3 -c 'import torch; assert torch.cuda.is_available()' 2>/dev/null; then
        DEVICE=cuda
    else
        DEVICE=cpu
    fi
fi

PROJECT_ROOT="/seu_share2/home/fenglei/213243847/Anti_lossRL/verl"

ANTI_MARGIN_ARG=()
if [[ -n "$ANTI_MARGIN" ]]; then
    ANTI_MARGIN_ARG=(--anti_margin "$ANTI_MARGIN")
fi

DEVICE_ARG=()
if [[ -n "$DEVICE" ]]; then
    DEVICE_ARG=(--device "$DEVICE")
fi

### =========================================================================
### Phase 1 — Offline validation
### =========================================================================

echo "=============================================="
echo "Phase 1 — Validate Anti-Loss Mechanism"
echo "=============================================="
echo "  Model:            $MODEL_PATH"
echo "  Test data:        $GSM8K_TEST_FILE"
echo "  Prompts:          $NUM_PROMPTS"
echo "  Rollouts/prompt:  $ROLLOUTS_PER_PROMPT"
echo "  Temperature:      $TEMPERATURE"
echo "  Anti margin:      ${ANTI_MARGIN:-none (basic mode)}"
echo "  Device:           $DEVICE"
echo "  Output dir:       $OUTPUT_DIR"
echo "=============================================="

export PYTHONPATH="${PYTHONPATH:-}"
PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

# Build args array — avoids backslash-continuation fragility (trailing whitespace breaks \)
VALIDATE_ARGS=(
    --model_path "$MODEL_PATH"
    --test_data_path "$GSM8K_TEST_FILE"
    --num_prompts "$NUM_PROMPTS"
    --rollouts_per_prompt "$ROLLOUTS_PER_PROMPT"
    --max_response_length "$MAX_RESPONSE_LENGTH"
    --max_prompt_length "$MAX_PROMPT_LENGTH"
    --temperature "$TEMPERATURE"
    --top_p "$TOP_P"
    --num_gradient_checks "$NUM_GRADIENT_CHECKS"
    --num_update_checks "$NUM_UPDATE_CHECKS"
    --update_lr "$UPDATE_LR"
    --seed "$SEED"
    --output_dir "$OUTPUT_DIR"
    "${ANTI_MARGIN_ARG[@]}"
    "${DEVICE_ARG[@]}"
)
python3 "$PROJECT_ROOT/scripts/validate_anti_loss.py" "${VALIDATE_ARGS[@]}"

echo ""
echo "Phase 1 validation complete. Results saved to $OUTPUT_DIR"
echo ""
echo "=============================================="
echo "After Phase 1 passes, run Phase 4 ablations:"
echo ""
echo "  # Baseline (no suppression)"
echo "  MODEL_PATH=$MODEL_PATH bash $0 --baseline"
echo ""
echo "  # With rollout suppression"
echo "  MODEL_PATH=$MODEL_PATH bash $0 --suppress"
echo ""
echo "  # With rollout suppression + margin"
echo "  MODEL_PATH=$MODEL_PATH bash $0 --suppress-margin"
echo "=============================================="



echo "结束时间: $(date)"
echo "退出码: $?"