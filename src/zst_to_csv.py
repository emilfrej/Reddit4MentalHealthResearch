"""
Decompress reddit .zst dumps to csv files for corpus and metadata
"""
import os
import io
import pandas as pd
import json
from pathlib import Path
import zstandard as zstd

#only keep relevant columns and chars above 100
def clean_df(df, min_char=100):
    df = df[['id', 'subreddit', 'selftext', 'author', 'created_utc']]

    #removes deleted and very short posts
    df_filtered = df[df['selftext'].str.len() >= min_char]

    return df_filtered

def decompress_to_combined_csv():
    '''
    Turns each subreddit dump in /rawdata from compressed .zst format to one big .csv
    Note: just r/mental_healt and r/depression took a 1.56m on 16core machine
    '''

    #get list of files (go up from src/ to project root)
    root = Path(__file__).resolve().parent.parent
    data_path = root / "reddit_data" 

    all_items = os.listdir(data_path)  # lists all files and folders
    files = [f for f in all_items if f.endswith('.zst')]

    # read in each file and setup vars for print statement
    dfs = []
    file_counter = 1
    n_files = len(files) 
    unfiltered_num_posts = 0

    for file in files:

        #print for user
        print(f"loading file {file_counter} out of {n_files}")

        #construct filename for each
        filename = os.path.join(data_path, file)

        #decompress .zst to json
        with open(filename, "rb") as fh:
            dctx = zstd.ZstdDecompressor()

            #for reading in 
            rows = []
            
            with dctx.stream_reader(fh) as reader:
                text_stream = io.TextIOWrapper(reader, encoding="utf-8", errors="replace")
                for line in text_stream:
                    #skip if bad
                    try:
                        rows.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        #turn rows into df                
        df = pd.DataFrame(rows)

        #increment 
        unfiltered_num_posts = unfiltered_num_posts + len(df)

        #filter out short texts and only keep certain cols
        df = clean_df(df)

        #append dfs to list
        dfs.append(df)
        
        #increment to keep track
        file_counter = file_counter + 1
        
    #concat to one when loop is done
    all_df = pd.concat(dfs, ignore_index=True)
    
    print(f"Unfiltered posts: {unfiltered_num_posts}")

    #get total number posts
    n_posts = len(all_df)

    #count number of text
    all_df["text_length"] = all_df["selftext"].str.len().fillna(0).astype(int)

    #store metadata in separate file
    meta_data = all_df[['subreddit', 'author', 'created_utc', 'text_length']]

    #anonymize authors. NAs and "[deleted]" get send to -1
    #converts to category codes so no usernames in output
    meta_data.loc[meta_data["author"] == "[deleted]", "author"] = pd.NA
    meta_data["author"] = (
        meta_data["author"]
            .astype("category")
            .cat.codes
            .astype("Int64")
    )

    #save to root
    meta_data.to_csv(root / f"meta_data_{n_posts}.csv")

    #also save to paper/ for make_figures_for_paper.Rmd
    paper_dir = root / "paper"
    paper_dir.mkdir(exist_ok=True)
    meta_data.to_csv(paper_dir / f"meta_data_{n_posts}.csv")

    #save corpus to .csv
    all_df[['selftext']].to_csv(root / f"combined_corpus_{n_posts}.csv")

#main func
def main():
    decompress_to_combined_csv()
    
#run script
if __name__ == "__main__":
    main()
