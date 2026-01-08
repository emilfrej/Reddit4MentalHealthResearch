"""
Extract TopicData object to compressed csv/npz files for sharing
"""
from turftopic.data import TopicData
from pathlib import Path
import argparse
import numpy as np
import scipy.sparse as sp
import zipfile
import shutil
from glob import glob


#takes topicdata joblib and extracts matricies to compressed format
def save_topic_data(td_path, out_dir, meta_data_path=None, create_zip=True):
    TD_PATH = Path(td_path)

    if not TD_PATH.exists():
        raise FileNotFoundError(f"{TD_PATH} does not exist.")

    print(f"Loading TopicData from {TD_PATH}")
    topic_data = TopicData.from_disk(TD_PATH)

    #pull out the diffrent matricies we need
    dtm = topic_data.document_term_matrix
    doc_topic = topic_data.document_topic_matrix
    topic_term = topic_data.topic_term_matrix
    vocab = topic_data.vocab
    doc_repr = topic_data.document_representation

    #setup output dir
    SAVE_DIR = Path(out_dir)
    SAVE_DIR.mkdir(exist_ok=True)

    #save doc-term matrix. use sparse format if its sparse
    print(f"Saving document_term_matrix (shape: {dtm.shape}, sparse: {sp.issparse(dtm)})")
    if sp.issparse(dtm):
        sp.save_npz(SAVE_DIR / "document_term_matrix.npz", dtm.tocsr())
    else:
        np.savez_compressed(SAVE_DIR / "document_term_matrix.npz", data=dtm)

    #save doc-topic matrix (dense)
    print(f"Saving document_topic_matrix (shape: {doc_topic.shape})")
    np.savez_compressed(SAVE_DIR / "document_topic_matrix.npz", data=doc_topic)

    #topic-term matrix is small so just dense
    print(f"Saving topic_term_matrix (shape: {topic_term.shape})")
    np.savez_compressed(SAVE_DIR / "topic_term_matrix.npz", data=topic_term)

    #embeddings
    print(f"Saving document_representation (shape: {doc_repr.shape})")
    np.savez_compressed(SAVE_DIR / "document_representation.npz", data=doc_repr)

    #vocab as numpy array
    print(f"Saving vocab ({len(vocab)} terms)")
    np.save(SAVE_DIR / "vocab.npy", np.array(vocab, dtype=object))

    #copy meta_data if provided
    if meta_data_path:
        meta_path = Path(meta_data_path)
        if meta_path.exists():
            dest_path = SAVE_DIR / meta_path.name
            shutil.copy(meta_path, dest_path)
            print(f"Copied {meta_path.name} to {SAVE_DIR}")
        else:
            print(f"Warning: meta_data file not found at {meta_path}")

    print(f"Saved compressed arrays to {SAVE_DIR}")

    #zip it up for easy sharing
    if create_zip:
        zip_path = SAVE_DIR.with_suffix(".zip")
        print(f"Creating zip archive: {zip_path}")
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for file in SAVE_DIR.iterdir():
                zf.write(file, file.name)

        # Print size comparison
        original_size = sum(f.stat().st_size for f in SAVE_DIR.iterdir())
        zip_size = zip_path.stat().st_size
        print(f"Uncompressed size: {original_size / 1e6:.1f} MB")
        print(f"Zip size: {zip_size / 1e6:.1f} MB")
        print(f"Done! Download: {zip_path}")


if __name__ == "__main__":
    #cli args
    parser = argparse.ArgumentParser()
    parser.add_argument("--td-path", default="fitted_models/SemanticSignalSeparation_topic_data.joblib")
    parser.add_argument("--out-dir", default="topic_data_compressed")
    parser.add_argument("--meta-data", default=None, help="Path to meta_data CSV file to include")
    parser.add_argument("--no-zip", action="store_true", help="Skip creating zip archive")
    args = parser.parse_args()

    #try to find meta_data automaticly if not given
    meta_data_path = args.meta_data
    if meta_data_path is None:
        meta_files = glob("meta_data*.csv")
        if meta_files:
            meta_data_path = meta_files[0]
            print(f"Auto-detected meta_data file: {meta_data_path}")

    save_topic_data(args.td_path, args.out_dir, meta_data_path=meta_data_path, create_zip=not args.no_zip)