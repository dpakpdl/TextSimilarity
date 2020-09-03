from typing import List

import gensim
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize

stop_words = set(stopwords.words('english'))


def pre_process_text(text):
    sentence_tokens = sent_tokenize(text)
    sentence_words = list()
    for sentence in sentence_tokens:
        words = [w for w in word_tokenize(sentence.lower()) if w not in stop_words]
        words = [''.join(filter(str.isalnum, word)) for word in words]
        words = list(filter(None, words))
        sentence_words.extend(words)
    return sentence_words


def similarity(documents: List[str], query: str):
    word_tokenized_docs = list()

    for document in documents:
        word_tokenized_docs.append(pre_process_text(document))

    dictionary = gensim.corpora.Dictionary(word_tokenized_docs)

    corpus = [dictionary.doc2bow(gen_doc) for gen_doc in word_tokenized_docs]

    tf_idf = gensim.models.TfidfModel(corpus)

    sims = gensim.similarities.Similarity('./workdir/', tf_idf[corpus], num_features=len(dictionary))

    query_doc = pre_process_text(query)

    query_doc_bow = dictionary.doc2bow(query_doc)

    query_doc_tf_idf = tf_idf[query_doc_bow]

    similarities = sims[query_doc_tf_idf]

    max_similarities = max(similarities)

    print('Comparing Result:', similarities)

    return max_similarities, documents[list(similarities).index(max_similarities)]


if __name__ == '__main__':
    documents = [
        "Venus is the second planet from the sun, and close to the earth, "
        "which is why it’s often referred to as our sister planet. It’s similar size to the Earth, "
        "about 7300 miles (12,000 kilometers). It’s nicknamed ‘’the morning star,” and thought to be the "
        "most inhabitable planet. Surface temperatures of Venus approach 900 degrees Fahrenheit, hot enough "
        "to melt the surface of the earth. Venus has a characteristic thick atmosphere, composed mainly of "
        "sulphuric acid and carbon dioxide.",
        "Mercury is the smallest planet in the solar system, approximately 3000 miles (4850 km) in diameter, "
        "hardly larger than the moon. Despite being the smallest, it’s extremely dense. "
        "In fact, it’s the second densest planet after Earth. It’s also the closest planet to the sun, "
        "making it dangerous to explore. Mercury is 48 million miles from the earth."
    ]
    text_to_search = "Zipping around the sun in only 88 days, Mercury is the closest planet" \
                     " to the sun, and it's also the smallest, only a little bit larger than Earth's moon." \
                     " Because its so close to the sun (about two-fifths the distance between Earth and the sun), " \
                     "Mercury experiences dramatic changes in its day and night temperatures: Day " \
                     "temperatures can reach a scorching 840  F (450 C), which is hot enough to melt lead. " \
                     "Meanwhile on the night side, temperatures drop to minus 290 F (minus 180 C). "

    print(similarity(documents, text_to_search))
