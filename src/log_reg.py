"""
Logistic regression case study: predicting if user will post in r/SuicideWatch
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
from glob import glob
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score, precision_recall_curve, confusion_matrix, roc_curve, roc_auc_score


#helper class to load topic data from csvs instead of joblib
class TopicDataFromCSV:
    """Simple container for topic data loaded from CSVs."""
    def __init__(self, document_topic_matrix, topic_term_matrix, vocab):
        self.document_topic_matrix = document_topic_matrix
        self.topic_term_matrix = topic_term_matrix
        self.vocab = vocab

    @classmethod
    def from_csv_dir(cls, csv_dir):
        """Load topic data from a directory containing the CSV files."""
        csv_dir = Path(csv_dir)
        doc_topic_df = pd.read_csv(csv_dir / "document_topic_matrix.csv")
        topic_term_df = pd.read_csv(csv_dir / "topic_term_matrix.csv")

        document_topic_matrix = doc_topic_df.values
        topic_term_matrix = topic_term_df.values
        vocab = np.array(topic_term_df.columns)

        return cls(document_topic_matrix, topic_term_matrix, vocab)


                            ## helpers for sklearn and topicdata ## 

#save_path - save to paper/ folder for make_figures_for_paper.Rmd
save_path = Path("paper/")

#make a function that will return a binary target vec: whether a user is going to post in r/SuicideWatch (SW)
# and features: mean document topic values from all posts made before a post in SW if any. 
# if they haven't posted in SW, the features are the mean doc topic values for all their posts
def make_SW_data(meta_data, document_topic_matrix):

    #binary indicator for SW posts
    SW_indicator = meta_data['subreddit'] == 'SuicideWatch'

    #get only rows that are SW
    SW_posts = meta_data[SW_indicator]

    #for each author find when they first posted in SW
    first_SW_time = SW_posts.groupby('author')['created_utc'].min().reset_index()
    first_SW_time = first_SW_time.rename(columns={'created_utc': 'first_SW_utc'})

    #merge back to meta
    meta_data = meta_data.merge(first_SW_time, on='author', how='left')

    #then group_by author and aggregate
    doc_top_df = pd.DataFrame(document_topic_matrix)

    #merge the document topic matrix with meta data
    combined_df = pd.concat([meta_data, doc_top_df], axis = 1)

    #code column for if user ever posted in SW
    combined_df['posted_in_SW'] = combined_df['first_SW_utc'].notna().astype(int)

    #check if row is before cut-off or if user never posted in SW
    combined_df = combined_df[
        (combined_df['posted_in_SW'] == 0) | ((combined_df['created_utc'] < combined_df['first_SW_utc']))]
    
    #check -1 (deleted user)
    #print(combined_df[combined_df["author"] == -1])

    #find names of all columns doc_topic loadings
    topic_cols = [col for col in combined_df.columns if isinstance(col, int)]  

    #mean per author
    author_topic_means = combined_df.groupby('author')[topic_cols].mean().reset_index()

    #assign idx
    author_summary = combined_df[['author', 'posted_in_SW']].drop_duplicates().merge(
        author_topic_means, on='author', how='left'
    )

    #construct targets
    y = author_summary["posted_in_SW"].to_numpy()

    #construct features
    X = author_summary[topic_cols].to_numpy()

    return X, y



# def function to make a classifer, predictions and true values from test train split
def logistic_reg_pipeline(X,y):
    #do train test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=1
    )

    #setup pipeline for logistic regression with L2 reg
    classifier = Pipeline([
        ("scaler", StandardScaler()),
        ("logreg", LogisticRegression(
            penalty="l2",
            max_iter=1000,
            class_weight="balanced"
        ))
    ])

    #fit
    classifier.fit(X_train, y_train)
    
    #make predictions and probs
    y_pred  = classifier.predict(X_test)
    y_proba = classifier.predict_proba(X_test)[:, 1]

    #return all objects
    return classifier, y_pred, y_proba, y_test



#get top k most predictive topics by coef size
def top_predictors(classifier, k, save_path):

    #extract logreg from sklearn pipeline
    logreg = classifier.named_steps["logreg"]

    #loop over coefs and store
    coefs = []
    for i, coef in enumerate(logreg.coef_.ravel()):
        coefs.append({"topic": i, "estimate": coef, "abs_estimate": np.abs(coef)})
    df = pd.DataFrame(coefs)

    #sort by abs value so we get both positive and negative predictors
    df = df.sort_values("abs_estimate", ascending = False)
    
    #save the df as csv in figures
    file_path = save_path / "predictor_values.csv"
    df.to_csv(file_path)

    return  df.head(k)


#def help function to get top and bottom n words from a topic id using topic data objs
def top_bottom_words(topic_idxs: list[int], topic_data, n: int = 10, save_path = save_path):
    """
    Return top and bottom-n words for each topic index.

    Returns
    -------
    dict[int, dict[str, list[str]]]
        {
          topic_idx: {
            "top": [...],
            "bottom": [...]
          }
        }
    """
    vocab = topic_data.vocab
    comps = topic_data.topic_term_matrix

    out = {}

    for k in topic_idxs:
        weights = comps[k]

        top_idx = np.argsort(weights)[-n:][::-1]
        bottom_idx = np.argsort(weights)[:n]

        out[k] = {
            "top": vocab[top_idx].tolist(),
            "bottom": vocab[bottom_idx].tolist(),
        }

    df = pd.DataFrame(out)    

    #save df to .csv (filename matches make_figures_for_paper.Rmd expectation)
    file_path = save_path / "top_bottom_topic_words-2.csv"
    df.to_csv(file_path)

    return df


                                        ### Plotting functions ###
def roc_plotly(y_test, y_proba, save_path):


    fpr, tpr, _ = roc_curve(y_test, y_proba)
    auc = roc_auc_score(y_test, y_proba)
    print(f"ROC AUC: {auc}")

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=fpr, y=tpr, mode="lines", name=f"AUC = {auc:.2f}")
    )
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            line=dict(dash="dash"),
            name="Chance"
        )
    )

    fig.update_layout(
        title="ROC Curve",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        template="simple_white"
    )


    save_path.mkdir(parents=True, exist_ok=True)
    file_path = save_path / "roc_plot.html"
    print(file_path)
    fig.write_html(file_path)

    return fig

def pr_plotly(y_test, y_proba, save_path):
    precision, recall, _ = precision_recall_curve(y_test, y_proba)
    ap = average_precision_score(y_test, y_proba)
    print(f"AP: {ap}")

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=recall,
            y=precision,
            mode="lines",
            name=f"AP = {ap:.2f}"
        )
    )

    # baseline = prevalence
    baseline = y_test.mean()
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[baseline, baseline],
            mode="lines",
            line=dict(dash="dash"),
            name=f"Chance {baseline:.2f}"
        )
    )

    fig.update_layout(
        title="Precision–Recall Curve",
        xaxis_title="Recall",
        yaxis_title="Precision",
        yaxis_range=[0, 1],
    )
    fig.update_layout(template="simple_white")

    save_path.mkdir(parents=True, exist_ok=True)
    file_path = save_path / "pr_plot.html"
    fig.write_html(file_path)

    return fig



def confusion_matrix_plotly(
    y_true,
    y_pred,
    save_path,
    labels=("Negative", "Positive"),
    normalize=False,
):
    cm = confusion_matrix(y_true, y_pred)

    if normalize:
        cm = cm / cm.sum(axis=1, keepdims=True)

    fig = px.imshow(
        cm,
        text_auto=".2f" if normalize else True,
        x=labels,
        y=labels,
        color_continuous_scale="Blues",
        aspect="equal",
    )

    fig.update_layout(
        title="Confusion Matrix",
        xaxis_title="Predicted label",
        yaxis_title="True label",
        template="simple_white",
    )

    fig.update_xaxes(side="bottom")

    save_path.mkdir(parents=True, exist_ok=True)
    file_path = save_path / "confusion_matrix.html"
    fig.write_html(file_path)

    return fig

def make_classification_report(y_test, y_pred, save_path):
    report = classification_report(
        y_test,
        y_pred,
        output_dict=True
    )

    df_report = pd.DataFrame(report).T.round(3)
    df_report.to_csv(save_path / "classification_report.csv")
    return df_report

if __name__ == "__main__":
    save_path = Path("paper")

    # Load topic data from CSVs
    topic_data = TopicDataFromCSV.from_csv_dir("topic_csvs")

    # Load metadata
    meta_files = glob("meta_data_*.csv")
    meta_df = pd.read_csv(meta_files[0])

    # Get the document topic matrix from topic data
    doc_top_mat = topic_data.document_topic_matrix

    # Construct target and features
    X, y = make_SW_data(meta_df, doc_top_mat)

    print(f"Freq of true classes: {y.mean()} out of total {len(y)}")

    # Test train split data, classify by L2 Logistic regression train, and make predictions
    classifier, y_pred, y_proba, y_test = logistic_reg_pipeline(X, y)

    # Make plots and store them
    make_classification_report(y_test, y_pred, save_path)
    roc_plotly(y_test, y_proba, save_path)
    pr_plotly(y_test, y_proba, save_path)
    confusion_matrix_plotly(y_test, y_pred, save_path, normalize=False)

    # Get top topics
    top_topics = top_predictors(classifier, 10, save_path).topic.tolist()

    # Return them as dataframe
    top_bottom_words(top_topics, topic_data, 5, save_path)

    # Sanity check: can we predict subreddit from post topics?
    X = doc_top_mat
    y = (meta_df['subreddit'] == 'SuicideWatch').astype(int).to_numpy()

    classifier, y_pred, y_proba, y_test = logistic_reg_pipeline(X, y)
    roc_plotly(y_test, y_proba, Path("figures/sanity_check"))







    