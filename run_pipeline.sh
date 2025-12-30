#!/bin/bash
# Pipeline script for topic modeling
# Usage:
#   ./run_pipeline.sh              # full corpus
#   ./run_pipeline.sh --sample     # downsample to 1000 docs for testing

set -e  # exit on error

DOWNSAMPLE=0
if [ "$1" == "--sample" ]; then
    DOWNSAMPLE=1000
    echo "Running in sample mode (${DOWNSAMPLE} docs)"
fi

#preprocess zst files to csv
echo "Preprocessing zst to csv"
python src/zst_to_csv.py

# fit topic model
echo "Fitting topic model "
python src/model_and_topic_data_fitting.py --downsample $DOWNSAMPLE

# Step 3: extract and compress topic data for sharing
echo "Extracting and compressing topic data"
python src/extract_topic_csvs.py

# Step 4: zip everything
echo " Zipping results and meta_data"
META_FILE=$(ls meta_data_*.csv 2>/dev/null | head -1)

zip -r results.zip topic_csvs/ $META_FILE fitted_models/

echo "Results saved to results.zip"
