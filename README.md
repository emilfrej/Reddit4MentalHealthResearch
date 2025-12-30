# Topic Modeling Mental Health Reddit Posts

Topic modeling of ~1.9 million mental health-related Reddit posts using Semantic Signal Separation (S3).

## Project Structure

```
.
├── setup.sh                     # Creates venv and installs dependencies
├── run_pipeline.sh              # Main pipeline script
├── requirements.txt             # Python dependencies
│
├── src/
│   ├── zst_to_csv.py            # Decompress raw Reddit data to CSV
│   ├── model_and_topic_data_fitting.py  # Fit topic models
│   ├── S3_hyperparamtuning.py   # Hyperparameter tuning for S3
│   ├── fit_time_testing.py      # Benchmark fitting times
│   ├── extract_topic_csvs.py    # Export topic data to compressed format
│   ├── log_reg.py               # Logistic regression case study
│   └── turftopic_benchmarking.py  # Evaluation metrics
│
├── paper/                       
│   ├── report_pdf.Rmd           # Main report
│   ├── make_figures_for_paper.R # Generate all figures
│   
│
├── reddit_data/                 # Place .zst files here
├── fitted_models/               # Saved models
├── evaluations/                 # Model evaluation results
└── figures/                     # Generated figures
```

## Installation (UCloud)

1. Clone project to UCloud

2. Run the setup script:
```
bash setup.sh
```
3. Activate the virtual environment:
```
source .venv/bin/activate
```

## Running the Pipeline

### Full pipeline

Place Reddit `.zst` dump files in `reddit_data/`, then change the topic number in `model_and_topic_data_fitting.py`.

```bash
./run_pipeline.sh
```

This will:
1. Decompress and preprocess Reddit data
2. Fit the S3 topic model
3. Extract topic data to compressed format
4. Zip results

To test the reproducibility of this project, I suggest running pipeline --sample on subset of the data on subsample of the data. I suggest r/psychosis, since this is the smallest of the used subreddits. 

### Individual scripts
The above pipeline doesn't emperical fit time testing for corpus subsambles nor hyperparameter tune a model like reported in the paper. If you wish do any of this, activate venv first, then the following scripts. Note that you will have to adapt, model types, subsamples size, and topic numbers. 

```bash
# 1. Preprocess data
python src/zst_to_csv.py

# 2. Fit topic model
python src/model_and_topic_data_fitting.py

# 3. Hyperparameter investigation
python src/S3_hyperparamtuning.py

# 4. Export topic data
python src/extract_topic_csvs.py

# 5. Run case study
python src/log_reg.py
```

### Generate figures
To generate the figures for the paper do. Obv this will require the full data set. This can then be used to knit report.rmd
```bash
cd paper/
Rscript make_figures_for_paper.R
```
populates
- `paper/figures/` - All figures for the paper
