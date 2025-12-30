import json
import time
import pandas as pd
import numpy as np
from glob import glob
from pathlib import Path
from turftopic.vectorizers.snowball import StemmingCountVectorizer

from turftopic import (SemanticSignalSeparation, ClusteringTopicModel, KeyNMF, SensTopic)
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer

#batched is python 3.12+, so we need a fallback
try:
    from itertools import batched
except ImportError:
    def batched(iterable, n):
        from itertools import islice
        it = iter(iterable)
        while batch := list(islice(it, n)):
            yield batch

### You need to run the following to use spacy vectorizers. See turftopic documentation for info
# python -m spacy download en_core_web_sm


def get_corpus(downsample: int = 0):
    '''
    Read in data and downsample if given. Returns as a lists of all texts
    '''

    #get path of .csv with filtered corpus
    path = glob("*corpus*.csv")[0]

    #read in df
    df = pd.read_csv(path)
    
    #downsample if arg is given
    if downsample != 0:
        df = df.sample(downsample)

    #do last second clean of corpus 
    corpus = df["selftext"].tolist()
    
    return corpus

### ADAPTED FROM TURFTOPIC MODEL BENCHMARKING CODE ###
def fit_topic_models(
    corpus,
    encoder_name: str = "all-MiniLM-L6-v2", #default turftopic choice
    out_dir: str = "fitted_models"
):
    #check if output folder exists
    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True)

    # make embeddings outside fitting loop
    encoder = SentenceTransformer(encoder_name, device="cpu")
    embeddings = encoder.encode(corpus, show_progress_bar=True)

    #setup vectorizers
    #lemma_vectorizer = LemmaCountVectorizer("en_core_web_sm")
    stem_vectorizer = StemmingCountVectorizer(language="english")
    count_vectorizer = CountVectorizer(ngram_range=(2,3), stop_words="english")



    #prep models
    topic_models = [
        {"model_name": "SemanticSignalSeparation", "model": SemanticSignalSeparation(encoder = encoder, vectorizer=stem_vectorizer)},
        #{"model_name": "SensTopic", "model": SensTopic(encoder = encoder, vectorizer = stem_vectorizer)},
        #{"model_name": "KeyNMF", "model": KeyNMF(100, top_n=25, encoder = encoder, vectorizer= stem_vectorizer)},
        #{"model_name": "ClusteringModel_BERT", "model": ClusteringTopicModel(n_reduce_to=30, reduction_method="average", encoder = encoder, vectorizer = count_vectorizer)},
    ]

    #for keeping track of fitting
    counter = 1
    corpus_size = len(corpus)

    fit_times = []

    #loop over models
    for topic_model in topic_models:
        
        #keep track of time
        start_time = time.time()
        
        model = topic_model["model"]
        model_name = topic_model["model_name"]

        #print status
        print(f"Fitting {model_name}: #{counter} of {len(topic_models)} models." )

        #do online fitting for KeyNMF
        if model_name == "KeyNMF":
            for batch in batched(corpus, 1000):
                batch = list(batch)
                model.partial_fit(batch)

        #fit the model
        else:
            model.fit(corpus, embeddings = embeddings)

        #prep topic data
        topic_data = model.prepare_topic_data(corpus = corpus, embeddings = embeddings)
        
        #save models and data 
        model_dir = out_dir / f"{model_name}_model"
        model_dir.mkdir(exist_ok = True)
        model.to_disk(model_dir)
        topic_data.to_disk(out_dir / f"{model_name}_topic_data.joblib")
        
        #increment for keeping track
        counter = counter + 1

        #elapsed time
        elapsed_time = time.time() - start_time

        #store fit times
        fit_times.append({"model_name": model_name, "time": elapsed_time})

    #store fit times to csv
    pd.DataFrame(fit_times).to_csv(out_dir / f"fit_times_{corpus_size}.csv")



def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--downsample", type=int, default=0, help="downsample corpus to n docs (0 = no downsample)")
    args = parser.parse_args()

    #read in data
    corpus = get_corpus(downsample=args.downsample)

    #fit and store models
    fit_topic_models(corpus)

if __name__ == "__main__":
    main()
