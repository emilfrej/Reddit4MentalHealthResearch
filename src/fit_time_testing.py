from model_and_topic_data_fitting import (get_corpus, fit_topic_models)

# main function to test different corpsizes
def main():

    corpus_sizes = [500, 1000, 2000, 10000]

    #loop over sizes and store results in paper/fit_time_testing for make_figures_for_paper.Rmd
    for size in corpus_sizes:
        print(f"Testing corpus of size {size}")
        corpus = get_corpus(size)
        fit_topic_models(corpus=corpus, out_dir="paper/fit_time_testing")


if __name__ == "__main__":
    main()