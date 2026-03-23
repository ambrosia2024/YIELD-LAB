#!/bin/bash
#SBATCH --job-name=cybench
#SBATCH --time=12:00:00
#SBATCH -p gpu_a100
#SBATCH -n 1
#SBATCH --gpus=1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=v.saxena@maastrichtuniversity.nl
#SBATCH --output=log/%x_%j.out
#SBATCH --error=log/%x_%j.err

source ~/miniconda3/etc/profile.d/conda.sh
conda activate cybench
export CUDA_VISIBLE_DEVICES=0

echo "Running on node: $(hostname)"
echo "Start time: $(date)"

crops=("maize" "wheat")
countries=("US" "DE")
hf_models=("autoformer" "informer" "patchtst" "tst" "tsmixer")
linear_models=("nlinear" "dlinear" "xlinear" "rlinear")

# -----------------------------
# Concurrency control
# Tune MAX_PARALLEL based on GPU
# memory. Safe default is 2.
# HF transformers: try 2-3
# Linear models: try 4-6
# -----------------------------
MAX_PARALLEL=4

# Create semaphore pipe
PIPE=$(mktemp -u)
mkfifo "$PIPE"
exec 3<>"$PIPE"
rm "$PIPE"

# Fill pipe with tokens
for i in $(seq 1 $MAX_PARALLEL); do
    echo >&3
done

mkdir -p checkpoints/results log

# -----------------------------
# run_model: acquires a token,
# launches process, releases
# token when done
# -----------------------------
run_model() {
    local log_file=$1
    shift
    local cmd=("$@")

    # Acquire token (blocks if MAX_PARALLEL already running)
    read -u 3

    {
        "${cmd[@]}" > "$log_file" 2>&1
        # Release token when process finishes
        echo >&3
    } &
}

# -----------------------------
# Merge helper
# -----------------------------
merge_results() {
    echo "Merging results..."
    for metric in nrmse mape r2 rmse mae mse smape; do
        final_csv="checkpoints/results/${metric}.csv"
        first=1
        for tmp_dir in checkpoints/results/tmp_*/; do
            src="${tmp_dir}${metric}.csv"
            if [ -f "$src" ]; then
                if [ $first -eq 1 ]; then
                    cp "$src" "$final_csv"
                    first=0
                else
                    tail -n +2 "$src" >> "$final_csv"
                fi
            fi
        done
        echo "Merged $metric.csv"
    done
    rm -rf checkpoints/results/tmp_*/
}

# -----------------------------
# HF models
# -----------------------------
echo "--------------------------------------"
echo "Running HF models"
echo "--------------------------------------"

for crop in "${crops[@]}"; do
    for country in "${countries[@]}"; do
        for model in "${hf_models[@]}"; do
            tmp_dir="checkpoints/results/tmp_${model}_${country}_${crop}"
            mkdir -p "$tmp_dir"

            echo "Starting $model $country $crop"

            cmd=(
                python tstBaselines.py
                --crop $crop
                --country $country
                --model_type $model
                --aggregation daily
                --batch_size 64
                --epochs 50
                --lag_years 0
                # --use_sota_features
                # --use_residual_trend
                # --use_recursive_lags
                --use_cwb_feature
                --test_years 5
                # --include_spatial_features
                # --use_gdd               
                # --use_heat_stress_days  
                # --use_rue               
                # --use_farquhar      
                --save_checkpoint_dir checkpoints/$crop/
                --results_dir "$tmp_dir"
            )

            run_model \
                "checkpoints/results/${model}_${country}_${crop}.txt" \
                "${cmd[@]}"

        done
    done
done

# -----------------------------
# Linear models
# -----------------------------
echo "--------------------------------------"
echo "Running Linear models"
echo "--------------------------------------"

for crop in "${crops[@]}"; do
    for country in "${countries[@]}"; do
        for model in "${linear_models[@]}"; do
            tmp_dir="checkpoints/results/tmp_${model}_${country}_${crop}"
            mkdir -p "$tmp_dir"

            echo "Starting $model $country $crop"

            cmd=(
                python linearBaselines.py
                --crop $crop
                --country $country
                --model_type $model
                --aggregation daily
                --batch_size 64
                --epochs 50
                --lag_years 0
                --test_years 5
                # --use_sota_features
                # --use_residual_trend
                # --use_recursive_lags
                --use_cwb_feature
                # --include_spatial_features
                --use_revIN
                # --use_gdd        
                # --use_heat_stress_days  
                # --use_rue            
                # --use_farquhar    
                --save_checkpoint_dir checkpoints/$crop/
                --results_dir "$tmp_dir"
            )

            run_model \
                "checkpoints/results/${model}_${country}_${crop}.txt" \
                "${cmd[@]}"

        done
    done
done

# Wait for all remaining jobs
wait

# Merge all CSVs into final files
merge_results

echo "End time: $(date)"
echo "All jobs finished."