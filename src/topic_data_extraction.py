from turftopic.data import TopicData
import pandas as pd
from glob import glob

#Read in S3 topicdata, remove corpus, and save lightweight version
def main():
    # Find topic data file
    td_files = glob('fitted_models/*_topic_data.joblib')
    if not td_files:
        raise FileNotFoundError("No topic_data.joblib found in fitted_models/")

    td_path = td_files[0]
    print(f"Reading topic data from {td_path}")
    topic_data = TopicData.from_disk(td_path)

    # Remove corpus to reduce file size
    topic_data.corpus = None

    out_path = "fitted_models/topic_data_no_corpus.joblib"
    topic_data.to_disk(out_path)
    print(f"Saved to {out_path}")

    
    
if __name__ == "__main__":
    main()
