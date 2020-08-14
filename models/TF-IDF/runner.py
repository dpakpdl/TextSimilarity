from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def plot_similarity(labels, corr, rotation):
    sns.set(font_scale=1.2)
    g = sns.heatmap(
        corr,
        xticklabels=labels,
        yticklabels=labels,
        vmin=0,
        vmax=1,
        cmap="YlOrRd")
    g.set_xticklabels(labels, rotation=rotation)
    g.set_title("Semantic Textual Similarity")
    plt.show()


def find_most_similar_document(doc):
    input_idx = corpus.index(doc)
    n, _ = pairwise_similarity.shape
    pairwise_similarity[np.arange(n), np.arange(n)] = -1.0
    # exclude the comparison between same text (diagonal)
    similar_doc_index = pairwise_similarity[input_idx].argmax()
    return corpus[similar_doc_index]


corpus = [
    "I'd like an apple",
    "An apple a day keeps the doctor away",
    "Never compare an apple to an orange",
    "I prefer scikit-learn to Orange",
    "The scikit-learn docs are Orange and Blue"
]

tfidf = TfidfVectorizer().fit_transform(corpus)
# no need to normalize, since Vectorizer will return normalized tf-idf
pairwise_similarity = tfidf * tfidf.T
arr = pairwise_similarity.toarray()
print(arr)
print(find_most_similar_document("I'd like an apple"))

plot_similarity(corpus, arr, 90)
