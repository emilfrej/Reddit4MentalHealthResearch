# Topic Modeling Mental Health Reddit Posts

Topic modeling of ~2 million mental health-related Reddit posts using Semantic Signal Separation (S3). See this [dropbox](https://www.dropbox.com/scl/fo/7pkofmtii9oflsi21dv0b/AHOwC3Q2lyBe2_hhvFqhhYI?rlkey=51u8ni3cxtowrp1v3chh7jh07&e=1&st=jrt3je13&dl=0) for the produced data and notebook on how to use it. 

## Project Structure

```
.
├── setup.sh  # creates venv and installs dependencies
├── run_pipeline.sh   # main pipeline script
├── requirements.txt   # dependencies to be used by setup.sh
│
├── src/ #scripts for the pipeline
│
├── paper/  #contains report in .rmd, scripts, data for plotting, and made plots redo for the paper
│
├── reddit_data/  #for .zst subreddit dumps
├── fitted_models/ #fitted models
├── evaluations/   #model evals
└── figures/  #non-paper ready figures
```

## Installation

1. Clone the project:
```bash
git clone https://github.com/emilfrej/Reddit4MentalHealthResearch.git
```
2. Run the setup script:
```
cd Reddit4MentalHealthResearch
bash setup.sh
```
3. Activate the virtual environment:
```
source .venv/bin/activate
```
This should make you ready to use the scripts.

## Running the Pipeline

### Full pipeline
To play around with the code,  I suggest downloading an single subreddit "submissions" file from the [pushshift dump](https://the-eye.eu/redarcs/). For example, by downloading the relatively small [r/pyschosis](https://the-eye.eu/redarcs/files/Psychosis_submissions.zst)
This will break the logistic regression since there no positive targets given that posts in r/SuicideWatch are not included, and no positive targets can therefore be found. To work around this you could, also download r/SuicideSubmissions.

If you want to run the full pipeline and recreate the results, you will need at least one subreddit in `.zst` from the pushshift archive. Place Reddit `.zst` files in `reddit_data/`, then change the topic number in `model_and_topic_data_fitting.py` to desired number. Recreating similiar results to the paper would require downloading the entire corpus and running S3 model with n_topics = 300. This would take a bunch of time to setup and waiting for runs to finish even with GPU access and high-performance computers. To this end I suggest getting a torrent [academic torrents](https://academictorrents.com/) and use the CLI tool aria2.

```bash
bash run_pipeline.sh
```

This will:
1. Decompress and preprocess Reddit data
2. Fit the S3 topic model
3. Extract topic data to compressed format
4. Zip results

To test the reproducibility of this project, I suggest running pipeline --sample on subset of the data on subsample of the data. I suggest r/psychosis, since this is the smallest of the used subreddits. 

### Individual scripts
The above pipeline doesn't emperical fit time testing for corpus subsambles nor hyperparameter tune a model like reported in the paper. If you wish do any of this, activate venv first, then the following scripts. Note that you will have to adapt, model types, subsamples size, and topic numbers.

The topic model evaluation metrics (coherence, diversity) used in `S3_hyperparamtuning.py` are from [turftopic](https://github.com/x-tabdeveloping/turftopic) and are therefore not distributed here. If you want to run the script you will need to place these functions in the placeholder script `turftopic_benchmarking.py`, refer to turftopic's benchmarking code. 

```bash
# 1. Preprocess data
python src/zst_to_csv.py

# 2. Fit topic model
python src/model_and_topic_data_fitting.py

# 3. Hyperparameter investigation
python src/S3_hyperparamtuning.py

# 4. Export topic data
python src/topic_data_extraction.py

# 5. Run case study
python src/log_reg.py
```

### Generate figures
To generate the figures for the paper run the following. This will require the full data set. This can then be used to knit report.rmd
```bash
cd paper/
Rscript make_figures_for_paper.R
```



