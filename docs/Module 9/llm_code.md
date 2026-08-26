# LLM Fine Tuning Activity

!!! note "Related Marimo Notebook"
    This module builds upon concepts from the following interactive notebook:

    - [Module 9 Practical: GPT and Fine-Tuning](https://github.com/gurgentus/applied_genai_notebooks/blob/main/notebooks/Module_9_Practical_GPT.py)

As part of this activity you will update the text generation API developed in Modules 3 and 7 to use a fine-tuned LLM. In particular, you will fine-tune a small GPT2 model (openai-community/gpt2) to handle question-answering. Fine-tuning involves taking a model with pre-trained weights and continuing its training using a new dataset.

Since training an LLM model from scratch is a long and resource intensive process you will use the excellent Transformer library from HuggingFace (https://huggingface.co/docs/transformers/en/index) to load the pre-trained model.

You will incorporate appropriate code from Module9-GPT notebook into your FastAPI implementation. This involves updating several key steps. 

---

## 1. Prepare the Data

As we have seen in the previous modules, Transformer based LLMs are trained to predict the next token in a token sequence. These models are referred to as "base models."  In order to make them perform specific tasks, like summarization or question-answering, we can fine-tune them on the corresponding training datasets.  You will use a question-answer dataset from HuggingFace called the Stanford Question Answering Dataset (SQuAD) (https://huggingface.co/datasets/rajpurkar/squad). 

Create train and test data loaders using the 

```python
DataLoader(...)
```

classes.

## 2. Get the "base model"


```python
model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2").to(device)
```

## 3. Setup the training loop 

The training loop should be similar to the one in the Module 9-GPT notebook, however, you should remember to use the pretrained model loaded in the previous step instead of the one built from scratch.

## 4. Update the API

Add a new function/endpoint called "/generate_with_llm" to the existing "/generate_text" and "/generate_with_rnn" function/endpoints that were defined in the previous activities.

```python
class TextGenerationRequest(BaseModel):
    start_word: str
    length: int

@app.post("/generate_with_llm")
def generate_with_llm(request: TextGenerationRequest):
    generated_text = # TODO
    return {"generated_text": generated_text}
```

## 5. Test the text generation functionality

Test the application using `uv run fastapi dev app/main.py` to make sure that the /generate_with_llm api endpoint works correctly. Optionally, rebuild the Docker image if you're using containerization.