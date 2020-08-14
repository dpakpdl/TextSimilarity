import tensorflow_hub as hub
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

module_url = "https://tfhub.dev/google/universal-sentence-encoder/4"

model = hub.load(module_url)

print(f"module {module_url} loaded")

# sample text
messages = [
    # Smartphones
    "My phone is not good.",
    "Your cellphone looks great.",

    # Weather
    "Will it snow tomorrow?",
    "Recently a lot of hurricanes have hit the US",

    # Food and health
    "An apple a day, keeps the doctors away",
    "Eating strawberries is healthy",
]


def embed(input_text):
    return model(input_text)


def run_and_plot(messages_):
    message_embeddings_ = embed(messages_)
    plot_similarity(messages_, message_embeddings_, 90)


def plot_similarity(labels, features, rotation):
    corr = np.inner(features, features)
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


run_and_plot(messages)
