import numpy as np
from itertools import chain, combinations

## copy past functions for diversity, word_embedding_coherence, get_keywords, evaluate_topic_quality here

def diversity(keywords: list[list[str]]) -> float:
    all_words = list(chain.from_iterable(keywords))
    unique_words = set(all_words)
    total_words = len(all_words)
    return float(len(unique_words) / total_words)


def word_embedding_coherence(keywords, wv):
    arrays = []
    for index, topic in enumerate(keywords):
        if len(topic) > 0:
            local_simi = []
            for word1, word2 in combinations(topic, 2):
                if word1 in wv.index_to_key and word2 in wv.index_to_key:
                    local_simi.append(wv.similarity(word1, word2))

                    #print(f"word 1 {word1}, word {word2}. Similarity {wv.similarity(word1, word2)}")
            arrays.append(np.nanmean(local_simi))
    return float(np.nanmean(arrays))


def get_keywords(model) -> list[list[str]]:
    """Get top words and ignore outlier topic."""
    n_topics = model.components_.shape[0]
    try:
        classes = model.classes_
    except AttributeError:
        classes = list(range(n_topics))
    res = []
    for topic_id, words in zip(classes, model.get_top_words()):
        if topic_id != -1:
            res.append(words)
    return res


def evaluate_topic_quality(keywords, ex_wv, in_wv) -> dict[str, float]:
    res = {
        "diversity": diversity(keywords),
        "c_in": word_embedding_coherence(keywords, in_wv),
        "c_ex": word_embedding_coherence(keywords, ex_wv),
    }
    return res

