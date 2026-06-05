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

source /home/herring/Anti_lossRL/.venv/bin/activate

cd /home/herring/Anti_lossRL


echo "=============================================================================="
echo ""
echo "开始时间: $(date)"

### =========================================================================
### User-adjustable parameters — override via env vars
### =========================================================================

# Model
MODEL_PATH=/home/herring/Anti_lossRL/model/Qwen2.5-0.5B-Instruct

# Data (GSM8K parquet)
GSM8K_TEST_FILE=/home/herring/Anti_lossRL/data/gsm8k_processed/test.jsonl

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

SCORING_METHOD=${SCORING_METHOD:-flexible}

# Gradient direction check
DIRECTION_THRESHOLD=${DIRECTION_THRESHOLD:-0.5}
MAX_CONFLICT_RATE=${MAX_CONFLICT_RATE:-0.5}
NORMALIZE_ADVANTAGES=${NORMALIZE_ADVANTAGES:-true}

# Constrained anti-loss budget
TARGET_ANTI_RATIO=${TARGET_ANTI_RATIO:-0.2}
LAMBDA_ANTI_MAX=${LAMBDA_ANTI_MAX:-1.0}

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

PROJECT_ROOT="/home/herring/Anti_lossRL/verl"

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
    --data_source "openai/gsm8k"
    --scoring_method "$SCORING_METHOD"
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
    --direction_threshold "$DIRECTION_THRESHOLD"
    --max_conflict_rate "$MAX_CONFLICT_RATE"
    --normalize_advantages "$NORMALIZE_ADVANTAGES"
    --target_anti_ratio "$TARGET_ANTI_RATIO"
    --lambda_anti_max "$LAMBDA_ANTI_MAX"
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