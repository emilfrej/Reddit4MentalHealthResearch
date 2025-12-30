# Benchmarking functions adapted from turftopic
# See: https://github.com/x-tabdeveloping/turftopic
#
# To use S3_hyperparamtuning.py, copy the following functions from turftopic's
# benchmarking code and paste them here:
# - diversity()
# - word_embedding_coherence()
# - get_keywords()
# - evaluate_topic_quality()


def diversity(keywords: list[list[str]]) -> float:
    """Calculate topic diversity as proportion of unique words across all topics."""
    raise NotImplementedError("Copy this function from turftopic benchmarking code")


def word_embedding_coherence(keywords, wv) -> float:
    """Calculate coherence using word embedding similarity."""
    raise NotImplementedError("Copy this function from turftopic benchmarking code")


def get_keywords(model) -> list[list[str]]:
    """Extract top keywords from topic model."""
    raise NotImplementedError("Copy this function from turftopic benchmarking code")


def evaluate_topic_quality(keywords, ex_wv, in_wv) -> dict[str, float]:
    """Evaluate topic quality using diversity and coherence metrics."""
    raise NotImplementedError("Copy this function from turftopic benchmarking code")
