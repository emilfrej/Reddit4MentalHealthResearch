# NOTE: evaluate_topic_quality and get_keywords functions are adapted from turftopic's
# benchmarking code. See: https://github.com/x-tabdeveloping/turftopic
# You will need to implement these yourself or copy from turftopic source.
from turftopic_benchmarking import evaluate_topic_quality, get_keywords

from glovpy import GloVe
import gensim.downloader as api
import numpy as np
from turftopic.vectorizers.snowball import StemmingCountVectorizer
from turftopic import load_model
import pandas as pd
from glob import glob
from pathlib import Path
from gensim.models import KeyedVectors
from joblib import Parallel, delayed



BASE_MODEL_PATH = "fitted_models/SemanticSignalSeparation_model"
EMBED_DIR = Path("embeddings")
MODEL_DIR = Path("fitted_models")
EVAL_DIR = Path("paper/evaluations")  # save to paper/ for make_figures_for_paper.Rmd

N_TOPICS = [5, 10, 15, 20, 30, 50, 100, 150, 200, 300, 382]
SEEDS = [0, 1, 2]  



def main():
    #redin in corpus
    corpus_path = glob("*corpus*.csv")[0]
    corpus = pd.read_csv(corpus_path)["selftext"]

    #check if embeddings dir exists, if not make
    EMBED_DIR.mkdir(exist_ok=True)

    # get word2vec corpus if doesn't exist
    ex_path = EMBED_DIR / "word2vec_google.kv"
    if ex_path.exists():
        ex_wv = KeyedVectors.load(str(ex_path), mmap="r")
    else:
        ex_wv = api.load("word2vec-google-news-300")
        ex_wv.save(str(ex_path))

    # setup tokenizer
    tokenizer = StemmingCountVectorizer().build_analyzer()

    # embed corpus with glove. This took approx hour on 64core cpu
    vector_size = 300

    in_path = EMBED_DIR / f"glove_in_domain_{vector_size}.kv"
    if in_path.exists():
        print(f"Loading in-domain embeddings: {in_path}")
        in_wv = KeyedVectors.load(str(in_path), mmap="r")
    else:
        print("Training GloVe embeddings")

        tokenized_corpus = Parallel(n_jobs=-1)(
            delayed(tokenizer)(doc) for doc in corpus
        )

        glove = GloVe(
            vector_size=vector_size,
            threads=32 #**-1
        )
        glove.train(tokenized_corpus)

        in_wv = glove.wv
        in_wv.save(str(in_path))
        print("GloVe training complete")

    MODEL_DIR.mkdir(exist_ok=True)
    EVAL_DIR.mkdir(exist_ok=True)

    #for results
    all_rows = []

    #looping over ntopics
    for n in N_TOPICS:

        #loop over seeds
        for seed in SEEDS:
            model_path = MODEL_DIR / f"S3_n{n}_seed{seed}"
            eval_path = EVAL_DIR / f"topic_quality_n{n}_seed{seed}.csv"
            done_flag = model_path / "_DONE"

            # load model if done. If not refit
            if model_path.exists() and done_flag.exists():
                print(f"Loading model (n={n}, seed={seed})")
                model = load_model(str(model_path))
            else:
                print(f"Refitting model (n={n}, seed={seed})")
                model = load_model(BASE_MODEL_PATH)
                model.refit(corpus, n_components=n, random_state=seed, max_iter = 400)

                model.to_disk(str(model_path))
                done_flag.touch()  #chat gpt said it was impornt
                print("Model saved")

            # check if eval exists. if not do it
            if eval_path.exists():
                print("Loading evaluation")
                df_eval = pd.read_csv(eval_path)
            else:
                print("Computing evaluation")

                #eval the model
                keywords = get_keywords(model)
                scores = evaluate_topic_quality(
                    keywords=keywords,
                    ex_wv=ex_wv,
                    in_wv=in_wv
                )

                scores.update({
                    "n_topics": n,
                    "seed": seed
                })

                df_eval = pd.DataFrame([scores])
                df_eval.to_csv(eval_path, index=False)
                print("Evaluation saved")

            all_rows.append(df_eval)



    # save all evals
    raw_df = pd.concat(all_rows, ignore_index=True)
    raw_df.to_csv("topic_quality_raw.csv", index=False)

    #summarize
    summary_df = (
        raw_df
        .groupby("n_topics")
        .agg(
            diversity_mean=("diversity", "mean"),
            diversity_std=("diversity", "std"),
            c_in_mean=("c_in", "mean"),
            c_in_std=("c_in", "std"),
            c_ex_mean=("c_ex", "mean"),
            c_ex_std=("c_ex", "std"),
        )
        .reset_index()
    )

    #save it
    summary_df.to_csv("topic_quality_summary.csv", index=False)
    print(summary_df)


if __name__ == "__main__":
    main()