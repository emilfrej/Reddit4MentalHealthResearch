"""
Fit S3 model to corpus with configurable n_topics.
Used for running on ucloud with GPU
"""
import time
import pandas as pd
import numpy as np
from glob import glob
from pathlib import Path
from turftopic.vectorizers.spacy import LemmaCountVectorizer
from turftopic.vectorizers.snowball import StemmingCountVectorizer

from turftopic import (SemanticSignalSeparation, ClusteringTopicModel, KeyNMF, SensTopic)
from sentence_transformers import SentenceTransformer
from itertools import batched
from sklearn.feature_extraction.text import CountVectorizer

import torch


def get_corpus(downsample: int = 0):
    path = glob("*corpus*.csv")[0]
    df = pd.read_csv(path)

    full_corpus_flag = True

    if downsample != 0:
        df = df.sample(downsample, random_state=42)
        full_corpus_flag = False

    corpus = df["selftext"].tolist()


    return corpus, full_corpus_flag

### ADAPTED FROM TURFTOPIC MODEL BENCHMARKING CODE ### 
def fit_topic_models(
    corpus,
    full_corpus_flag,
    n_topics,
    encoder_name: str = "all-MiniLM-L6-v2", #default turftopic choice
    out_dir: str = "fitted_models"

):


    #check if output folders exists
    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True)


    #use cuda, onnx doesn't seem to work on ucloud, but should be faster according to Tt documentation
    backend = "torch"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    

    #if embeddings exists and downsample not true, load them
    emb_path = Path(f"embeddings/corpus_embeddings.npy")
    if emb_path.exists() and full_corpus_flag:
        
        print(f"Embeddings found at {emb_path}. being loaded...")
        embeddings = np.load(emb_path)
    else:
        #print
        print(f"Embedding corpus using {backend} and {device}")

        encoder = SentenceTransformer(encoder_name,device=device, backend=backend)
        embeddings = encoder.encode(
            corpus,
            show_progress_bar=True,
        )
        emb_path.parent.mkdir(parents=True, exist_ok=True)

        #try
        np.save(emb_path, embeddings)


    #setup vectorizer
    stem_vectorizer = StemmingCountVectorizer(language="english")
    
    #prep models
    topic_models = [
        {"model_name": "SemanticSignalSeparation", "model": SemanticSignalSeparation(vectorizer=stem_vectorizer, n_components=n_topics)},
        #{"model_name": "SensTopic", "model": SensTopic(encoder = encoder, vectorizer = stem_vectorizer)},
        #{"model_name": "KeyNMF", "model": KeyNMF(100, top_n=25, encoder = encoder, vectorizer= stem_vectorizer)},
        #{"model_name": "ClusteringModel_BERT", "model": ClusteringTopicModel(n_reduce_to=30, reduction_method="average", encoder = encoder, vectorizer = stem_vectorizer)},
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
        model_name = f"{topic_model['model_name']}_{n_topics}"

        #print status
        print(f"Fitting {model_name}: #{counter} of {len(topic_models)} models, with {n_topics} to full corpus: {full_corpus_flag}" )

        #do online fitting for KeyNMF
        if model_name == "KeyNMF":
            for batch in batched(corpus, 1000):
                batch = list(batch)
                model.partial_fit(batch)

        #fit the model
        else:
            model.fit(embeddings=embeddings, raw_documents=corpus)

        
        #save models and data 
        model_dir = out_dir / f"{model_name}_model"
        model_dir.mkdir(exist_ok = True)

        #prep topic data
        topic_data = model.prepare_topic_data(corpus = corpus, embeddings = embeddings)
        
        #sometimes fails on gpu, try anyway
        try:
            model.to_disk(model_dir)
        except Exception as e:
            print("Failed to load model on CPU too:", e)
        
        topic_data.to_disk(out_dir / f"{model_name}_{n_topics}_topic_data.joblib")
        
        #increment for keeping track
        counter = counter + 1

        #elapsed time
        elapsed_time = time.time() - start_time

        #store fit times
        fit_times.append({"model_name": model_name, "time": elapsed_time})

    #store fit times to csv
    pd.DataFrame(fit_times).to_csv(out_dir / f"fit_times_{corpus_size}.csv")


def main():

    #read in data. 0 = no downsampling
    corpus, full_corpus_flag = get_corpus(1000) 

    #fit and store model + topic_data
    fit_topic_models(corpus, full_corpus_flag, 10)

if __name__ == "__main__":
    main()
