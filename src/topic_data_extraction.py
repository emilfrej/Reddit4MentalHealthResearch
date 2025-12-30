from turftopic.data import TopicData
import joblib
import copy
from turftopic import load_model
import pandas as pd
from glob import glob
from turftopic.analyzers import LLMAnalyzer

#Def a function to read in S3 topicdata and remove corpus, make llm topic names and save version wiht and wihtout llm names
def main():

    #read in corpus
    corpus_file = glob('*corpus*.csv')
    corpus = pd.read_csv(corpus_file[0])["selftext"]

    print("Reading in TD")
    topic_data = TopicData.from_disk("fitted_models/SemanticSignalSeparation_100_topic_data.joblib")
    print("done reading in TD. running llm analyser")

    #remove the corpus and save
    topic_data.corpus = None
    topic_data.to_disk("fitted_models/S3_llm_topics_no_corpus.joblib")

    
    
if __name__ == "__main__":
    main()
