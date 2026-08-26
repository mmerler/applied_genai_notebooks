import marimo

__generated_with = "0.21.1"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # Module 1: Practice 2 - Word Sampling

    In the slides, we explored how the **likelihood of a particular phrase** in a text can be represented using **probabilities**. This concept allows us to **generate new text** that is stylistically and contextually similar to the original.

    To achieve the most realistic generation, the **probability of the next word** should ideally depend on **all of the preceding text.** However, calculating this is **computationally infeasible** for large documents.

    To simplify this, we make an assumption: the probability of the next word **depends only on a fixed number of preceding words**, known as the **context.** Recall that, more generally, text is usually broken up into elements called **tokens** (in modern language models these contain only parts of the word), but to keep things relatively simple we will initially work with one-word tokens.

    In the simplest case, we assume the next word depends only on the **immediately preceding word.** This is an example of a **Markov Assumption** and uses **bigram probabilities** to **generate random text** starting from a given word by sequentially sampling the next word based on its bigram probability.

    We will also **visualize the bigram probabilities** using a **matrix format**, where each row represents the current word and each column shows the probability of the next word.

    It will form a **baseline NLP model** that we will **deploy** as part of this module's class activity.
    As we learn more advanced techniques, we'll **update the model** to improve its performance and capabilities.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Bigram Probabilities

    How should the bigram probabilities be modeled? Intuitively, we can just derive the probabilities from the frequency of the bigram occurance in our data simply as:

    $$p(w_2 | w_1) = \frac{C(w_1, w_2)}{C(w_1)}$$

    where $C(w_1, w_2)$ is the count of the bigram $(w_1, w_2)$ and $C(w_1)$ is the count of the unigram $w_1$.

    This formula can actually be derived mathematically from the general principle of maximum likelihood estimation (MLE). MLE is a method of estimating the parameters of a statistical model by maximizing the likelihood function, which measures how well the model explains the observed data.

    Probability of a word sequence $w_1, w_2, \ldots, w_n$ can be expressed as:

    $$
    p(w_1, w_2, \ldots, w_n) = p(w_1) p(w_2 | w_1) p(w_3 | w_1, w_2) \cdots p(w_n | w_1, w_2, \ldots, w_{n-1})
    $$

    Using the Markov bigram assumption, this simplifies to:

    $$
    p(w_1, w_2, \ldots, w_n) = p(w_1) p(w_2 | w_1) p(w_3 | w_2) \cdots p(w_n | w_{n-1})
    $$

    An MLE estimator for the bigram probabilities corresponds to **modeling** the probability distribution $p$ in such a way that if $w_1, w_2, \ldots, w_n$ are the observed words in the training data, the probability above is maximized.

    Indeed, it can be shown that the MLE estimator for the bigram probabilities is given by the bigram count formula above, which is repeated below for convenience:

    $$p(w_2 | w_1) = \frac{C(w_1, w_2)}{C(w_1)}.$$

    This formula is used in the `analyze_bigrams` function below. The mathematics behind the derivation is beyond the scope of this module, but for those interested here is a nice writeup: https://leimao.github.io/blog/Maximum-Likelihood-Estimation-Ngram/.
    """)
    return


@app.cell
def _():
    from collections import defaultdict, Counter
    import numpy as np
    import matplotlib.pyplot as plt
    import random
    import re

    def simple_tokenizer(text, frequency_threshold=5):
        """Simple tokenizer that splits text into words."""
        # Convert to lowercase and extract words using regex
        tokens = re.findall(r"\b\w+\b", text.lower())
        if not frequency_threshold:
            return tokens
        # Count word frequencies
        word_counts = Counter(tokens)
        # Define a threshold for less frequent words (e.g., words appearing fewer than 5 times)
        filtered_tokens = [
            token for token in tokens if word_counts[token] >= frequency_threshold
        ]
        return filtered_tokens

    def analyze_bigrams(text, frequency_threshold=None):
        """Analyze text to compute bigram probabilities."""
        words = simple_tokenizer(text, frequency_threshold)
        bigrams = list(zip(words[:-1], words[1:]))  # Create bigrams

        # Count bigram and unigram frequencies
        bigram_counts = Counter(bigrams)
        unigram_counts = Counter(words)

        # Compute bigram probabilities
        bigram_probs = defaultdict(dict)
        for (word1, word2), count in bigram_counts.items():
            bigram_probs[word1][word2] = count / unigram_counts[word1]

        return list(unigram_counts.keys()), bigram_probs

    def generate_text(bigram_probs, start_word, num_words=20):
        """Generate text based on bigram probabilities."""
        current_word = start_word.lower()
        generated_words = [current_word]

        for _ in range(num_words - 1):
            next_words = bigram_probs.get(current_word)
            if not next_words:  # If no bigrams for the current word, stop generating
                break

            # Choose the next word based on probabilities
            next_word = random.choices(
                list(next_words.keys()), weights=next_words.values()
            )[0]
            generated_words.append(next_word)
            current_word = next_word  # Move to the next word

        return " ".join(generated_words)

    def print_bigram_probs_matrix_python(vocab, bigram_probs):
        """
        Print bigram probabilities in a matrix format for Python console output.

        Args:
        - bigram_probs (dict): A dictionary of bigram probabilities.
        """
        # Print the header row
        print(f"{'':<15}", end="")
        for word in vocab:
            print(f"{word:<15}", end="")
        print("\n" + "-" * (15 * (len(vocab) + 1)))

        # Print each row with probabilities
        for word1 in vocab:
            print(f"{word1:<15}", end="")
            for word2 in vocab:
                prob = bigram_probs.get(word1, {}).get(word2, 0)
                print(f"{prob:<15.2f}", end="")
            print()

    # Example input text
    input_text = """
    Darkness cannot drive out darkness, only light can do that
    """
    # Analyze the bigrams
    vocab, bigram_probabilities = analyze_bigrams(input_text)
    print_bigram_probs_matrix_python(vocab, bigram_probabilities)
    return analyze_bigrams, bigram_probabilities, generate_text


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Here we train the model on a short quote by Martin Luther King Jr.

    Since the training text is very short and each word appears only once or twice, the model has **limited continuation options**.

    Most words in the bigrams we constructed have only **one possible continuation**, so the model assigns **full probability** to that continuation.

    For example, since there are only **two bigrams** starting with the word *'darkness'*, the model gives **equal probability** to both possible continuations.

    Let's generate text starting from this word:
    """)
    return


@app.cell
def _(bigram_probabilities, generate_text):
    start_word = "darkness"
    generated_text = generate_text(bigram_probabilities, start_word, num_words=20)

    print("Generated Text:\n", generated_text)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    As we can see, **most of the generated text** mirrors the training text because **only one continuation** is possible for most words.

    The only variation occurs after the word **'darkness'**, where the model has **two possible continuations**. Since the selection is **random**, this will get different generated text if we run the generation enough times.

    To get more interesting and varied text, let's **calculate the bigram probabilities** from a **larger piece of text**.
    """)
    return


@app.cell
def _():
    import requests

    # Get 'Count of Monte Cristo' text from Project Gutenberg
    book_url = "https://www.gutenberg.org/cache/epub/1184/pg1184.txt"

    # Download the book
    response = requests.get(book_url)
    book_text = response.text

    # Remove Gutenberg header and footer
    start_marker = "*** START OF THIS PROJECT GUTENBERG EBOOK"
    end_marker = "*** END OF THIS PROJECT GUTENBERG EBOOK"

    start_idx = book_text.find(start_marker)
    end_idx = book_text.find(end_marker)

    if start_idx != -1 and end_idx != -1:
        book_text = book_text[start_idx + len(start_marker) : end_idx]
    return (book_text,)


@app.cell
def _(analyze_bigrams, book_text):
    # Example usage
    book_vocab, book_bigram_probabilities = analyze_bigrams(book_text)
    return (book_bigram_probabilities,)


@app.cell
def _(book_bigram_probabilities, generate_text):
    # Generate random text starting from a word
    print(
        "Generated Text:\n",
        generate_text(book_bigram_probabilities, "The", num_words=20),
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    While the generated text is somewhat nonsensical, it is more **unique** and isn't too far from sounding like English. It is, however, limited by the **quality of the bigram probabilities.**

    We can improve the text by using more advanced techniques, such as:
    - **Trigram probabilities**: Taking into account the previous two words instead of just one.
    - **Smoothing techniques**: Handling unseen word combinations more effectively.

    You will learn about these techniques in a **Natural Language Processing** course. However, they require **more complex models** and **larger datasets** to train on.

    A major limitation of these models is that they treat **different words as completely separate** entities, even if the words have **similar or identical meanings.**

    To overcome this, we need a **lower-dimensional latent space** that captures the **meaning of words** and their **relationships.**

    This is where **word embeddings** come into play:

    - **Word embeddings** are **dense vector representations** of words that capture **semantic relationships** between words.
    - These embeddings can be **learned from large text corpora** using **neural networks**, enabling models to understand similarities between words in a **contextual and meaningful** way.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    This activity provided an **intuition** for **sampling from a probability distribution** of possible word continuations. By understanding how probabilities influence text generation, we gained insight into the **basic mechanics** behind language models.

    Starting in the **next module**, we will build **more powerful models** using **neural networks** to generate these probabilities. These advanced methods will allow us to capture **richer contextual relationships**, improve **coherence**, and enhance the **quality of generated text**.
    """)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
