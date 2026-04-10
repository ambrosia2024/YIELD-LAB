#!/bin/bash
#SBATCH --job-name=cybench
#SBATCH --time=6:00:00
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

# ==================== CONFIGURATION ====================
# Minimum number of years required for training
MIN_YEARS=8

# Countries to exclude (already trained)
EXCLUDED_COUNTRIES=""
# =======================================================

# Embed the years_info.txt dictionary directly
YEARS_DICT=$(cat <<'EOF'
{
'maize':
    {'AO': [2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017],
    'AR': [2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023],
    'AT': [2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020],
    'BE': [2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020],
    'BF': [2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2016, 2017, 2018, 2019],
    'BG': [2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020],
    'BR': [2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023],
    'CN': [2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022],
    'CZ': [2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020],
    'DE': [2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021],
    'DK': [2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020],
    'EE': [],
    'EL': [2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019],
    'ES': [2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020],
    'ET': [2003, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2016, 2017, 2020],
    'FI': [],
    'FR': [2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020],
    'HR': [2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020],
    'HU': [2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020],
    'IE': [],
    'IN': [2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017],
    'IT': [2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020],
    'LS': [2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021],
    'LT': [2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020],
    'LV': [],
    'MG': [2005, 2006, 2007, 2008, 2009, 2010],
    'ML': [2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017],
    'MW': [2018, 2019, 2020, 2021, 2022, 2023],
    'MX': [2014, 2017, 2019, 2022],
    'MZ': [2004, 2005, 2006, 2007, 2008, 2009, 2010, 2012, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022],
    'NE': [2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2019, 2020, 2021],
    'NL': [2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020],
    'PL': [2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020],
    'PT': [2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020],
    'RO': [2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020],
    'SE': [2007, 2011, 2012, 2013, 2014, 2016, 2017, 2018, 2019, 2020],
    'SK': [2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018],
    'SN': [2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015],
    'TD': [2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017],
    'US': [2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023],
    'ZA': [2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022],
    'ZM': [2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017]},
'wheat': {
    'AR': [2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023],
    'AT': [2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020],
    'AU': [2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023],
    'BE': [2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020],
    'BG': [2010, 2011, 2012, 2013, 2014, 2016, 2017, 2018, 2019, 2020],
    'BR': [2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022],
    'CN': [2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022],
    'CZ': [2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020],
    'DE': [2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021],
    'DK': [2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020],
    'EE': [2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020],
    'EL': [2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019],
    'ES': [2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020],
    'FI': [2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020],
    'FR': [2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020],
    'HR': [2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020],
    'HU': [2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020],
    'IE': [2010, 2011, 2012, 2019, 2020],
    'IN': [2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017],
    'IT': [2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020],
    'LT': [2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020],
    'LV': [2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018],
    'NL': [2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020],
    'PL': [2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020],
    'PT': [2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020],
    'RO': [2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020],
    'SE': [2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020],
    'SK': [2017, 2018],
    'US': [2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023]}}
EOF
)

# Dynamically extract crops and filtered countries
crops=($(python3 -c "
import ast
import sys

data = ast.literal_eval('''$YEARS_DICT''')
MIN_YEARS = $MIN_YEARS
EXCLUDED = '$EXCLUDED_COUNTRIES'.split()

# Get all crops
for crop in data.keys():
    print(crop)
"))

# Get countries for each crop with >= MIN_YEARS
declare -A MAIZE_COUNTRIES
declare -A WHEAT_COUNTRIES

while IFS='|' read -r crop country; do
    if [ "$crop" = "maize" ]; then
        MAIZE_COUNTRIES[$country]=1
    elif [ "$crop" = "wheat" ]; then
        WHEAT_COUNTRIES[$country]=1
    fi
done < <(python3 -c "
import ast

data = ast.literal_eval('''$YEARS_DICT''')
MIN_YEARS = $MIN_YEARS
EXCLUDED = '$EXCLUDED_COUNTRIES'.split()

for crop in data.keys():
    for country, years in data[crop].items():
        if len(years) >= MIN_YEARS and country not in EXCLUDED:
            print(f'{crop}|{country}')
")

echo "Configuration:"
echo "  MIN_YEARS: $MIN_YEARS"
echo "  EXCLUDED_COUNTRIES: $EXCLUDED_COUNTRIES"
echo "  Crops: ${crops[@]}"
echo "  Maize countries with >=$MIN_YEARS years: ${!MAIZE_COUNTRIES[@]}"
echo "  Wheat countries with >=$MIN_YEARS years: ${!WHEAT_COUNTRIES[@]}"

hf_models=("autoformer" "informer" "patchtst" "tsmixer" "tst" "itransformer" "timexer")
linear_models=("nlinear" "dlinear" "xlinear" "rlinear")

# -----------------------------
# Concurrency control
# Tune MAX_PARALLEL based on GPU
# memory. Safe default is 2.
# HF transformers: try 2-3
# Linear models: try 4-6
# -----------------------------
MAX_PARALLEL=3

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

# Function to get countries for a crop
get_countries_for_crop() {
    local crop=$1
    if [ "$crop" = "maize" ]; then
        echo "${!MAIZE_COUNTRIES[@]}"
    elif [ "$crop" = "wheat" ]; then
        echo "${!WHEAT_COUNTRIES[@]}"
    fi
}

# Sort countries alphabetically for consistent ordering
sort_countries() {
    echo $1 | tr ' ' '\n' | sort | tr '\n' ' '
}

# -----------------------------
# NEW: Check if result file exists and is non-empty
# -----------------------------
is_completed() {
    local result_file="checkpoints/results/${1}_${2}_${3}.txt"
    if [ -f "$result_file" ] && [ -s "$result_file" ]; then
        return 0  # Completed
    else
        return 1  # Not completed
    fi
}

# -----------------------------
# HF models
# -----------------------------
echo "--------------------------------------"
echo "Running HF models"
echo "--------------------------------------"

# Uncomment one at a time to experiment:
# cmd+=(--use_sota_features)
# cmd+=(--use_residual_trend)
# cmd+=(--use_recursive_lags)
# cmd+=(--use_gdd)
# cmd+=(--use_heat_stress_days)
# cmd+=(--use_rue)
# cmd+=(--use_farquhar)
# cmd+= (--include_spatial_features)

for crop in "${crops[@]}"; do
   countries=$(sort_countries "$(get_countries_for_crop $crop)")
   for country in $countries; do
       for model in "${hf_models[@]}"; do
           # Check if already completed
           if is_completed "$model" "$country" "$crop"; then
               echo "Skipping $model $country $crop (already completed)"
               continue
           fi

           tmp_dir="checkpoints/results/tmp_${model}_${country}_${crop}"
           mkdir -p "$tmp_dir"

           echo "Starting $model $country $crop"

           cmd=(
               python tstBaselines.py
               --crop $crop
               --country $country
               --model_type $model
               --aggregation daily
               --batch_size 32
               --epochs 50
               --lag_years 0
               --use_cwb_feature
               --test_years 5
               --save_checkpoint_dir checkpoints/yield-$model-cybench/$country/$crop/
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
# echo "--------------------------------------"
# echo "Running Linear models"
# echo "--------------------------------------"

for crop in "${crops[@]}"; do
   countries=$(sort_countries "$(get_countries_for_crop $crop)")
   for country in $countries; do
       for model in "${linear_models[@]}"; do
           # Check if already completed
           if is_completed "$model" "$country" "$crop"; then
               echo "Skipping $model $country $crop (already completed)"
               continue
           fi

           tmp_dir="checkpoints/results/tmp_${model}_${country}_${crop}"
           mkdir -p "$tmp_dir"

           echo "Starting $model $country $crop"

          cmd=(
               python linearBaselines.py
               --crop $crop
               --country $country
               --model_type $model
               --aggregation daily
               --batch_size 32
               --epochs 50
               --lag_years 0
               --test_years 5
               --use_cwb_feature
               --save_checkpoint_dir checkpoints/yield-$model-cybench/$country/$crop/
               --results_dir "$tmp_dir"
           )

           run_model \
               "checkpoints/results/${model}_${country}_${crop}.txt" \
               "${cmd[@]}"

       done
   done
done

# -----------------------------
# Trend model
# -----------------------------
echo "--------------------------------------"
echo "Running Trend model"
echo "--------------------------------------"

for crop in "${crops[@]}"; do
    countries=$(sort_countries "$(get_countries_for_crop $crop)")
    for country in $countries; do
        # Check if already completed
        if is_completed "trend" "$country" "$crop"; then
            echo "Skipping trend $country $crop (already completed)"
            continue
        fi

        tmp_dir="checkpoints/results/tmp_trend_${country}_${crop}"
        mkdir -p "$tmp_dir"

        echo "Starting trend $country $crop"

        cmd=(
            python trendBaseline.py
            --crop $crop
            --country $country
            --epochs 50
            --aggregation daily
            --test_years 5
            --lag_years 2
            --include_spatial_features
            --use_cwb_feature
            --save_checkpoint_dir checkpoints/yield-trend-cybench/$country/$crop/
            --results_dir "$tmp_dir"
        )

        run_model \
            "checkpoints/results/trend_${country}_${crop}.txt" \
            "${cmd[@]}"

    done
done

# Wait for all remaining jobs
wait

# Merge all CSVs into final files
merge_results

echo "End time: $(date)"
echo "All jobs finished."
