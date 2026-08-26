import marimo

__generated_with = "0.21.1"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Setup

    First, we import the *spacy* library and load the large English model.
    """)
    return


@app.cell
def _():
    import spacy

    nlp = spacy.load("en_core_web_lg")
    return (nlp,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    Next, let's define a function to calculate word embeddings based on an input word:
    """)
    return


@app.cell
def _(nlp):
    def calculate_embedding(input_word):
        word = nlp(input_word)
        return word.vector

    return (calculate_embedding,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Let's try with the word 'apple'.  For brevity, only the first elements of the embedding vector are displayed:
    """)
    return


@app.cell
def _(calculate_embedding):
    calculate_embedding("apple")[:10]
    return


@app.cell(hide_code=True)
def _(mo):
    word_input_ui = mo.ui.text(value="orange")
    mo.md(
        f"""
    ## More Practice with Word Embeddings

    Type in a word to generate an embedding vector: {word_input_ui}
    """
    )
    return (word_input_ui,)


@app.cell(hide_code=True)
def _(calculate_embedding, word_input_ui):
    word_embedding = calculate_embedding(word_input_ui.value)
    word_embedding[:10]
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Similarity
    Let's add a function to calculate the similarity between two words based on their embeddings:
    """)
    return


@app.cell
def _(nlp):
    def calculate_similarity(word1, word2):
        return nlp(word1).similarity(nlp(word2))

    return (calculate_similarity,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    Compare embeddings of words: 'apple' and 'car'
    """)
    return


@app.cell
def _(calculate_similarity):
    calculate_similarity("apple", "car")
    return


@app.cell(hide_code=True)
def _(mo):
    word1_input_ui = mo.ui.text(value="apple")
    word2_input_ui = mo.ui.text(value="orange")
    mo.md(
        f"""
    ## More Practice with Similarity

    Enter any two words to generate the similarity measure between them: 

    Word 1: {word1_input_ui} &nbsp; Word 2: {word2_input_ui}
    """
    )
    return word1_input_ui, word2_input_ui


@app.cell(hide_code=True)
def _(calculate_similarity, word1_input_ui, word2_input_ui):
    calculate_similarity(word1_input_ui.value, word2_input_ui.value)
    return


@app.cell(hide_code=True)
def _(mo):
    la_word1_input_ui = mo.ui.text(value="spain")
    la_word2_input_ui = mo.ui.text(value="paris")
    la_word3_input_ui = mo.ui.text(value="france")
    la_word4_input_ui = mo.ui.text(value="madrid")

    mo.md(
        f"""
    We can even do linear algebra with the underlying vector representations. Enter any three words and calculate similarity with a fourth one, e.g.:

    'woman' + 'king' - 'man' with 'queen'

    OR

    'spain' + 'paris' - 'france' with 'madrid': 

    Word 1: {la_word1_input_ui} + (Word 2: {la_word2_input_ui} - Word 3: {la_word3_input_ui})

    compared to

    Word 4: {la_word4_input_ui}.
    """
    )
    return (
        la_word1_input_ui,
        la_word2_input_ui,
        la_word3_input_ui,
        la_word4_input_ui,
    )


@app.cell(hide_code=True)
def _(
    la_word1_input_ui,
    la_word2_input_ui,
    la_word3_input_ui,
    la_word4_input_ui,
    nlp,
):
    la_word1_embedding = nlp(la_word1_input_ui.value).vector
    la_word2_embedding = nlp(la_word2_input_ui.value).vector
    la_word3_embedding = nlp(la_word3_input_ui.value).vector
    la_word = la_word1_embedding + (la_word2_embedding - la_word3_embedding)
    la_word4 = nlp(la_word4_input_ui.value).vector
    return la_word, la_word4


@app.cell
def _(la_word, la_word4):
    from sklearn.metrics.pairwise import cosine_similarity

    print("Cosine similarity: ", cosine_similarity([la_word], [la_word4])[0][0])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ----
    ## Sentence Embeddings

    Finally, to calculate an embedding for a sentence, we can just average the embeddings of all the words in that sentence.  We will again use `spacy` to calculate the sentence embeddings.

    ```python
    query = "What is the capital of France?"
    info_1 = "The capital of France is Paris"
    info_2 = "France is a beautiful country"
    info_3 = "Today is very warm in New York City"
    print("Response 1 Similarity: ", nlp(query).similarity(nlp(info_1)))
    print("Response 2 Similarity: ", nlp(query).similarity(nlp(info_2)))
    print("Response 3 Similarity: ", nlp(query).similarity(nlp(info_3)))
    ```
    """)
    return


@app.cell(hide_code=True)
def _(nlp):
    query = "What is the capital of France?"
    info_1 = "The capital of France is Paris"
    info_2 = "France is a beautiful country"
    info_3 = "Today is very warm in New York City"
    print("Response 1 Similarity: ", nlp(query).similarity(nlp(info_1)))
    print("Response 2 Similarity: ", nlp(query).similarity(nlp(info_2)))
    print("Response 3 Similarity: ", nlp(query).similarity(nlp(info_3)))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Being able to quickly calculate similarities between a query and target information text is very powerful for Information Retrieval, especially when combined with Large Language Models trained for chat/question answering capabilities.
    """)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
