#!/bin/bash
# Run experiments from JSON config files
# Usage: ./run_config.sh <dataset> <potential> [num_seeds] [date_tag]
# Example: ./run_config.sh singlecell W2 5 26Jan24
#
# Post-training mode (re-run evaluation on trained models):
# Usage: ./run_config.sh --post-training <dataset> <potential> [date_tag]
# Example: ./run_config.sh --post-training singlecell W2 26Jan24

set -e

# Check for post-training mode
POST_TRAINING=false
if [[ "$1" == "--post-training" ]]; then
    POST_TRAINING=true
    shift
fi

DATASET=${1:?Usage: $0 [--post-training] <dataset> <potential> [num_seeds] [date_tag]}
POTENTIAL=${2:?Usage: $0 [--post-training] <dataset> <potential> [num_seeds] [date_tag]}

if [[ "$POST_TRAINING" == true ]]; then
    DATE_TAG=${3:-$(date +%d%b%y)}
    NUM_SEEDS=0  # Not used in post-training mode
else
    NUM_SEEDS=${3:-5}
    DATE_TAG=${4:-$(date +%d%b%y)}
fi

CONFIG_FILE="configs/${DATASET}/${POTENTIAL}.json"

if [[ ! -f "$CONFIG_FILE" ]]; then
    echo "Error: Config file not found: $CONFIG_FILE"
    echo "Available configs:"
    ls -1 configs/${DATASET}/*.json 2>/dev/null || echo "  No configs for dataset '$DATASET'"
    exit 1
fi

echo "============================================================"
echo "Dataset: $DATASET"
echo "Potential: $POTENTIAL"
echo "Config: $CONFIG_FILE"
if [[ "$POST_TRAINING" == true ]]; then
    echo "Mode: POST-TRAINING (re-run evaluation)"
else
    echo "Seeds: $NUM_SEEDS"
fi
echo "Date tag: $DATE_TAG"
echo "============================================================"

# Parse JSON config into command-line arguments
# Uses python to handle JSON parsing robustly
EXTRA_ARGS=$(python3 << EOF
import json
with open("$CONFIG_FILE") as f:
    config = json.load(f)

args = []
for key, value in config.items():
    if key == "description":
        continue
    # Convert snake_case to kebab-case for CLI
    cli_key = key.replace("_", "-")
    if isinstance(value, bool):
        if value:
            args.append(f"--{cli_key}")
    elif isinstance(value, list):
        args.append(f"--{cli_key}")
        args.extend(str(v) for v in value)
    else:
        args.append(f"--{cli_key}")
        args.append(str(value))

print(" ".join(args))
EOF
)

echo "Extra args from config: $EXTRA_ARGS"
echo ""

# Dataset-specific base arguments and module
case $DATASET in
    singlecell)
        BASE_ARGS="--pca-dim 100"
        MODULE="fm_explore.singlecell.train_singlecell"
        ;;
    gulfofmexico)
        BASE_ARGS=""
        MODULE="fm_explore.gulfofmexico.train_gom"
        ;;
    *)
        echo "Error: Unknown dataset '$DATASET'"
        echo "Supported datasets: singlecell, gulfofmexico"
        exit 1
        ;;
esac

# Change to project directory
cd /home/ubuntu/fm-explore

# Use conda environment (pixi doesn't have CUDA pytorch yet)
source /home/ubuntu/miniconda3/etc/profile.d/conda.sh
conda activate env_3msbm

# Post-training mode: run evaluation on already trained models
if [[ "$POST_TRAINING" == true ]]; then
    echo ""
    echo "Running post-training evaluation..."

    BASE_DIR="results_local/${DATE_TAG}/${POTENTIAL}"
    if [[ ! -d "$BASE_DIR" ]]; then
        echo "Error: Directory not found: $BASE_DIR"
        exit 1
    fi

    # Find all seed directories
    for seed_dir in "$BASE_DIR"/seed_*; do
        if [[ ! -d "$seed_dir" ]]; then
            continue
        fi

        # Find the run directory inside seed_N
        RUN_DIR=$(ls -d "$seed_dir"/*/ 2>/dev/null | head -1)
        if [[ -z "$RUN_DIR" ]]; then
            echo "  $(basename $seed_dir): no run directory found"
            continue
        fi

        # Check if already has final epoch metrics in losses.csv
        # (Just having losses.csv isn't enough - need final epoch metrics)
        if [[ -f "${RUN_DIR}losses.csv" ]]; then
            # Check if final epoch metrics exist by looking for the highest metric_epochs value
            final_epoch=$(python3 -c "
import csv, sys
with open('${RUN_DIR}losses.csv') as f:
    reader = csv.DictReader(f)
    epochs = [float(r.get('metric_epochs', 0) or 0) for r in reader if r.get('fgd_t1')]
    print(int(max(epochs)) if epochs else 0)
" 2>/dev/null)
            # Get expected final epoch from args.json
            expected_epoch=$(python3 -c "
import json
with open('${RUN_DIR}args.json') as f:
    print(json.load(f).get('epochs', 0))
" 2>/dev/null)
            if [[ "$final_epoch" == "$expected_epoch" ]]; then
                echo "  $(basename $seed_dir): ✓ already complete (epoch $final_epoch)"
                continue
            fi
        fi

        # Check if has a model checkpoint
        CHECKPOINT=$(ls "$RUN_DIR"/models/model_epoch_*.pt 2>/dev/null | sort -V | tail -1)
        if [[ -z "$CHECKPOINT" ]]; then
            echo "  $(basename $seed_dir): no checkpoint found"
            continue
        fi

        echo ""
        echo "============================================================"
        echo "Post-training: $(basename $seed_dir)"
        echo "============================================================"

        PYTHONPATH=/home/ubuntu/fm-explore/src python -m $MODULE \
            --post-training-only "$RUN_DIR"
    done

    echo ""
    echo "Post-training complete!"
    exit 0
fi

# Run experiments with different seeds sequentially launched with delays
echo "Starting $NUM_SEEDS experiments..."
PIDS=()

for seed in $(seq 1 $NUM_SEEDS); do
    # Create unique save directory: results_local/<date>/<potential>/seed_<N>
    SAVE_DIR="results_local/${DATE_TAG}/${POTENTIAL}/seed_${seed}"
    mkdir -p "$SAVE_DIR"

    echo "  Launching seed=$seed -> $SAVE_DIR"
    PYTHONPATH=/home/ubuntu/fm-explore/src python -m $MODULE \
        $BASE_ARGS --save-dir "$SAVE_DIR" --tag run $EXTRA_ARGS --seed $seed &

    PIDS+=($!)

    # Add delay between launches to prevent race conditions
    if [[ $seed -lt $NUM_SEEDS ]]; then
        sleep 3
    fi
done

echo ""
echo "All $NUM_SEEDS experiments launched (PIDs: ${PIDS[*]})"
echo "Results will be saved to: results_local/${DATE_TAG}/${POTENTIAL}/seed_*"
echo "Waiting for completion..."
wait
echo "Done!"
