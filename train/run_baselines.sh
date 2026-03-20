#!/bin/bash
#SBATCH --job-name=cybench
#SBATCH --time=03:00:00
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

run_group () {

    crop=$1
    country=$2

    echo "--------------------------------------"
    echo "Running HF models for $country $crop"
    echo "--------------------------------------"

    pids=()

    for model in "${hf_models[@]}"; do

        echo "Starting $model $country $crop"

        python tstBaselines.py \
        --crop $crop \
        --country $country \
        --model_type $model \
        --aggregation daily \
        --batch_size 64 \
        --epochs 50 \
        --lag_years 0 \
        --use_sota_features \
        --use_residual_trend \
        --use_recursive_lags \
        --use_cwb_feature \
        --aggregation daily\
        --include_spatial_features \
        --save_checkpoint_dir checkpoints/$crop/ \
        > checkpoints/results/${model}_${country}_${crop}.txt 2>&1 &

        pids+=($!)
        sleep 2

    done

    for pid in "${pids[@]}"; do
        wait $pid
    done

    echo "Finished HF group $country $crop"
}

# -----------------------------
# HF groups
# -----------------------------

run_group maize US
run_group wheat US
run_group maize DE
run_group wheat DE


# -----------------------------
# Linear models
# -----------------------------

echo "--------------------------------------"
echo "Running Linear models"
echo "--------------------------------------"

pids=()

for crop in "${crops[@]}"; do
for country in "${countries[@]}"; do
for model in "${linear_models[@]}"; do

    echo "Starting $model $country $crop"

    python linearBaselines.py \
    --crop $crop \
    --country $country \
    --model_type $model \
    --aggregation daily \
    --batch_size 64 \
    --epochs 50 \
    --lag_years 0 \
    --use_sota_features \
    --use_residual_trend \
    --use_recursive_lags \
    --use_cwb_feature \
    --aggregation daily\
    --include_spatial_features \
    --use_revIN \
    --save_checkpoint_dir checkpoints/$crop/ \
    > checkpoints/results/${model}_${country}_${crop}.txt 2>&1 &

    pids+=($!)
    sleep 2

done
done
done

for pid in "${pids[@]}"; do
    wait $pid
done

echo "End time: $(date)"
echo "All jobs finished."