import marimo

__generated_with = "0.23.9"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Module 9: Practical 2 - Fine-Tuning an LLM

    In the first practical we trained a small GPT from scratch. Here we start from **pretrained** language models and fine-tune them to answer questions about a passage (the SQuAD dataset), in two ways:

    1. **Full fine-tuning** of GPT-2 small (124M parameters): every weight is updated.
    2. **LoRA fine-tuning** of a larger model, Qwen2.5-1.5B (1.5B parameters): the pretrained weights stay frozen and we train small low-rank adapters, about 1% of the parameters.

    Finally we compare all four models (two base models, two fine-tuned) on the same held-out questions: perplexity on the answers, exact match and F1, and the generated answers side by side.

    **Hardware.** One GPU with 16 GB of memory (for example a Colab T4) runs the whole notebook in roughly 20–30 minutes. On a laptop CPU, set `QUICK_RUN = True` for a fast smoke test; the LoRA part will still be slow.

    **Install:** `pip install torch transformers datasets peft accelerate pandas matplotlib tqdm`
    """)
    return


@app.cell
def _():
    import math
    import re
    import string
    from collections import Counter

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader
    from tqdm.auto import tqdm
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup
    from peft import LoraConfig, get_peft_model

    torch.manual_seed(42)
    np.random.seed(42)

    device = (
        torch.device("cuda")
        if torch.cuda.is_available()
        else torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    )

    # Half precision on GPU: bfloat16 on recent GPUs (Ampere or newer), float16 on older ones such as the T4.
    if device.type == "cuda":
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    else:
        dtype = torch.float32

    print(f"Using device: {device}, compute dtype: {dtype}")
    return (
        AutoModelForCausalLM,
        AutoTokenizer,
        Counter,
        DataLoader,
        F,
        LoraConfig,
        device,
        dtype,
        get_linear_schedule_with_warmup,
        get_peft_model,
        load_dataset,
        math,
        np,
        pd,
        plt,
        re,
        string,
        torch,
        tqdm,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Configuration

    `QUICK_RUN = True` uses tiny subsets so you can check that everything runs before committing GPU time. The models can be swapped: any causal language model on the Hugging Face Hub works, as long as it fits in memory.
    """)
    return


@app.cell
def _():
    QUICK_RUN = True  # True: tiny subsets for a fast end-to-end check

    FULL_FT_MODEL_ID = "openai-community/gpt2"  # 124M parameters, full fine-tuning
    LORA_MODEL_ID = "Qwen/Qwen2.5-1.5B"  # 1.5B-parameter base model, LoRA fine-tuning

    N_TRAIN = 200 if QUICK_RUN else 4000  # SQuAD training examples
    N_EVAL = 40 if QUICK_RUN else 200  # held-out questions used for every comparison
    MAX_LEN = 384  # max tokens per example (prompt + answer); longer examples are skipped
    MAX_NEW_TOKENS = 24  # SQuAD answers are short

    # Full fine-tuning (GPT-2)
    FULL_BATCH_SIZE = 8
    FULL_LR = 5e-5

    # LoRA fine-tuning (larger model)
    LORA_BATCH_SIZE = 4
    LORA_GRAD_ACCUM = 2  # effective batch size 8
    LORA_LR = 2e-4  # adapters are small and start at zero, so they use a higher learning rate
    LORA_RANK = 16
    LORA_ALPHA = 32

    EPOCHS = 1
    return (
        EPOCHS,
        FULL_BATCH_SIZE,
        FULL_FT_MODEL_ID,
        FULL_LR,
        LORA_ALPHA,
        LORA_BATCH_SIZE,
        LORA_GRAD_ACCUM,
        LORA_LR,
        LORA_MODEL_ID,
        LORA_RANK,
        MAX_LEN,
        MAX_NEW_TOKENS,
        N_EVAL,
        N_TRAIN,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Data: SQuAD question answering

    Each SQuAD example has a **context** paragraph from Wikipedia, a **question**, and an **answer** that is a short span of the context. A base language model only continues text, so we teach it the task by always writing examples in the same template:

    ```
    Context: <paragraph>
    Question: <question>
    Answer: <answer><end-of-text>
    ```

    At test time we give the model everything up to `Answer:` and let it write the rest. Ending every training answer with the end-of-text token teaches the model when to stop.
    """)
    return


@app.cell
def _(N_EVAL, N_TRAIN, load_dataset):
    squad = load_dataset("rajpurkar/squad")
    train_raw = squad["train"].shuffle(seed=42).select(range(min(N_TRAIN, len(squad["train"]))))
    # Take a few extra validation questions: some are skipped later for being too long.
    eval_raw = squad["validation"].shuffle(seed=42).select(range(min(2 * N_EVAL, len(squad["validation"]))))
    print(squad)
    return eval_raw, train_raw


@app.cell
def _(torch):
    def build_prompt(context, question):
        return f"Context: {context}\nQuestion: {question}\nAnswer:"

    def encode_example(tokenizer, context, question, answer, max_len):
        """Token ids for prompt + answer, with labels = -100 on the prompt so only the answer is learned."""
        prompt_ids = tokenizer(build_prompt(context, question), add_special_tokens=False)["input_ids"]
        answer_ids = tokenizer(" " + answer, add_special_tokens=False)["input_ids"] + [tokenizer.eos_token_id]
        if len(prompt_ids) + len(answer_ids) > max_len:
            return None  # too long: skip
        return {
            "input_ids": prompt_ids + answer_ids,
            "labels": [-100] * len(prompt_ids) + answer_ids,
        }

    def make_collate(pad_id):
        """Pad a batch on the right; padding gets label -100 and attention mask 0."""

        def collate(batch):
            length = max(len(ex["input_ids"]) for ex in batch)
            input_ids = torch.full((len(batch), length), pad_id, dtype=torch.long)
            labels = torch.full((len(batch), length), -100, dtype=torch.long)
            attention_mask = torch.zeros((len(batch), length), dtype=torch.long)
            for i, ex in enumerate(batch):
                n = len(ex["input_ids"])
                input_ids[i, :n] = torch.tensor(ex["input_ids"])
                labels[i, :n] = torch.tensor(ex["labels"])
                attention_mask[i, :n] = 1
            return {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}

        return collate

    return build_prompt, encode_example, make_collate


@app.cell
def _(train_raw):
    train_raw[0]
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Two models, two tokenizers

    Our notebook GPT used a word-level vocabulary. Pretrained models come with their own subword tokenizer, and we must use it: the token ids are what the embedding table was trained on. GPT-2 uses byte-level BPE with 50,257 tokens; Qwen2.5 uses byte-level BPE with about 151K tokens, so it usually needs fewer tokens for the same text.

    GPT-2 has no padding token, so we reuse its end-of-text token for padding (padding positions get label −100 and are ignored by the loss anyway).
    """)
    return


@app.cell
def _(AutoTokenizer, FULL_FT_MODEL_ID, LORA_MODEL_ID):
    def load_tokenizer(model_id):
        tok = AutoTokenizer.from_pretrained(model_id)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        tok.padding_side = "left"  # decoder-only models generate after the prompt, so pad on the left
        return tok

    full_tok = load_tokenizer(FULL_FT_MODEL_ID)
    lora_tok = load_tokenizer(LORA_MODEL_ID)

    _sentence = "Transformers empower users to fine-tune pretrained language models."
    for _name, _tok in [("GPT-2", full_tok), ("Qwen2.5", lora_tok)]:
        _pieces = _tok.convert_ids_to_tokens(_tok(_sentence, add_special_tokens=False)["input_ids"])
        print(f"{_name:8s} vocab {len(_tok):,}: {len(_pieces)} tokens -> {_pieces}")
    return full_tok, lora_tok


@app.cell
def _(
    MAX_LEN,
    N_EVAL,
    encode_example,
    eval_raw,
    full_tok,
    lora_tok,
    train_raw,
):
    def encode_split(tokenizer, rows):
        out = []
        for row in rows:
            ex = encode_example(tokenizer, row["context"], row["question"], row["answers"]["text"][0], MAX_LEN)
            if ex is not None:
                out.append(ex)
        return out

    full_train = encode_split(full_tok, train_raw)
    lora_train = encode_split(lora_tok, train_raw)

    # The evaluation questions must be identical for all models: keep only those that fit for both tokenizers.
    eval_examples = []
    for _row in eval_raw:
        _fits = all(
            encode_example(_tok, _row["context"], _row["question"], _row["answers"]["text"][0], MAX_LEN) is not None
            for _tok in (full_tok, lora_tok)
        )
        if _fits:
            eval_examples.append(
                {"context": _row["context"], "question": _row["question"], "answers": list(_row["answers"]["text"])}
            )
        if len(eval_examples) == N_EVAL:
            break

    full_eval = [encode_example(full_tok, e["context"], e["question"], e["answers"][0], MAX_LEN) for e in eval_examples]
    lora_eval = [encode_example(lora_tok, e["context"], e["question"], e["answers"][0], MAX_LEN) for e in eval_examples]

    print(f"Training examples: GPT-2 {len(full_train):,}, Qwen {len(lora_train):,} (of {len(train_raw):,}; the rest exceed {MAX_LEN} tokens)")
    print(f"Evaluation questions shared by all models: {len(eval_examples)}")
    return eval_examples, full_eval, full_train, lora_eval, lora_train


@app.cell
def _(full_tok, full_train, mo):
    _ex = full_train[0]
    _n_prompt = sum(1 for l in _ex["labels"] if l == -100)
    mo.md(
        f"""
    **One training example as GPT-2 sees it** ({len(_ex["input_ids"])} tokens). The prompt (first {_n_prompt} tokens) has label −100, so the loss only covers the answer:

    ```
    {full_tok.decode(_ex["input_ids"][:_n_prompt])}
    ```
    **Learned part:** `{full_tok.decode(_ex["input_ids"][_n_prompt:])!r}`
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Training and evaluation helpers

    The training loop is the one from the GPT practical, with three differences:

    - The Hugging Face model computes the loss itself when we pass `labels` (it shifts them by one position internally, and ignores −100).
    - Only parameters with `requires_grad=True` are optimized. For full fine-tuning that is every weight; with LoRA it is only the adapters.
    - On a GPU we run the forward pass in half precision with `torch.autocast`, which roughly halves memory and time.

    For evaluation we use the three measures from the slides:

    - **Perplexity on the answer tokens**: exp of the average negative log-probability of the reference answers. Compare a model only with itself before and after fine-tuning: GPT-2 and Qwen use different tokenizers, so their perplexities are not comparable.
    - **Exact match (EM) and F1**, the official SQuAD metrics, computed on the generated text after lowercasing and removing punctuation and articles. These *are* comparable across models.
    - **Greedy decoding** for generation: for factual answers we want the single most likely answer, not creative sampling.
    """)
    return


@app.cell
def _(
    DataLoader,
    device,
    dtype,
    get_linear_schedule_with_warmup,
    math,
    torch,
    tqdm,
):
    def autocast():
        return torch.autocast(device_type=device.type, dtype=dtype, enabled=(device.type == "cuda"))

    def train_causal_lm(model, train_data, collate, lr, epochs, batch_size, grad_accum=1, desc="training"):
        """Fine-tune any Hugging Face causal LM; returns the list of per-batch training losses."""
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
        loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=collate)
        total_steps = epochs * math.ceil(len(loader) / grad_accum)
        scheduler = get_linear_schedule_with_warmup(optimizer, int(0.05 * total_steps), total_steps)
        # float16 needs loss scaling to avoid underflowing gradients; bfloat16 and float32 do not
        scaler = torch.amp.GradScaler("cuda") if (device.type == "cuda" and dtype == torch.float16) else None

        model.train()
        losses = []
        for epoch in range(epochs):
            progress = tqdm(loader, desc=f"{desc} (epoch {epoch + 1}/{epochs})")
            for i, batch in enumerate(progress):
                batch = {k: v.to(device) for k, v in batch.items()}
                with autocast():
                    loss = model(**batch).loss / grad_accum
                if scaler is not None:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

                if (i + 1) % grad_accum == 0 or (i + 1) == len(loader):
                    if scaler is not None:
                        scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(params, 1.0)
                    if scaler is not None:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)

                losses.append(loss.item() * grad_accum)
                if i % 20 == 0:
                    progress.set_postfix(loss=f"{sum(losses[-20:]) / len(losses[-20:]):.3f}")
        model.eval()
        return losses

    return autocast, train_causal_lm


@app.cell
def _(
    Counter,
    DataLoader,
    F,
    autocast,
    build_prompt,
    device,
    math,
    np,
    re,
    string,
    torch,
):
    @torch.no_grad()
    def answer_perplexity(model, eval_data, collate, batch_size=8):
        """exp(mean negative log-likelihood) over the reference answer tokens only."""
        model.eval()
        total_nll, total_tokens = 0.0, 0
        for batch in DataLoader(eval_data, batch_size=batch_size, collate_fn=collate):
            batch = {k: v.to(device) for k, v in batch.items()}
            with autocast():
                logits = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits
            logits = logits[:, :-1].float()  # position t predicts token t+1
            targets = batch["labels"][:, 1:]
            total_nll += F.cross_entropy(
                logits.reshape(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-100, reduction="sum"
            ).item()
            total_tokens += (targets != -100).sum().item()
        return math.exp(total_nll / total_tokens)

    @torch.no_grad()
    def generate_answers(model, tokenizer, examples, max_new_tokens, batch_size=8):
        """Greedy answers for a list of {context, question} dicts."""
        model.eval()
        answers = []
        for start in range(0, len(examples), batch_size):
            prompts = [build_prompt(e["context"], e["question"]) for e in examples[start : start + batch_size]]
            enc = tokenizer(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(device)
            with autocast():
                out = model.generate(
                    **enc,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    use_cache=True,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            for row in out[:, enc["input_ids"].shape[1] :]:
                text = tokenizer.decode(row, skip_special_tokens=True)
                answers.append(text.strip().split("\n")[0].strip())  # keep the first line only
        return answers

    def _normalize(s):
        s = s.lower()
        s = "".join(ch for ch in s if ch not in set(string.punctuation))
        s = re.sub(r"\b(a|an|the)\b", " ", s)
        return " ".join(s.split())

    def _f1(prediction, reference):
        pred, ref = _normalize(prediction).split(), _normalize(reference).split()
        common = sum((Counter(pred) & Counter(ref)).values())
        if common == 0:
            return 0.0
        precision, recall = common / len(pred), common / len(ref)
        return 2 * precision * recall / (precision + recall)

    def squad_scores(predictions, examples):
        """Official SQuAD exact match and F1 (best over the reference answers), in %."""
        em = [max(float(_normalize(p) == _normalize(r)) for r in e["answers"]) for p, e in zip(predictions, examples)]
        f1 = [max(_f1(p, r) for r in e["answers"]) for p, e in zip(predictions, examples)]
        return 100 * np.mean(em), 100 * np.mean(f1)

    return answer_perplexity, generate_answers, squad_scores


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Part A: full fine-tuning of GPT-2

    First we measure the **base** GPT-2 on the evaluation questions. It has never seen this template, so expect it to ramble or repeat the context.
    """)
    return


@app.cell
def _(
    AutoModelForCausalLM,
    FULL_FT_MODEL_ID,
    MAX_NEW_TOKENS,
    answer_perplexity,
    device,
    eval_examples,
    full_eval,
    full_tok,
    generate_answers,
    make_collate,
    squad_scores,
):
    full_collate = make_collate(full_tok.pad_token_id)

    _base = AutoModelForCausalLM.from_pretrained(FULL_FT_MODEL_ID).to(device)
    full_base_ppl = answer_perplexity(_base, full_eval, full_collate)
    full_base_preds = generate_answers(_base, full_tok, eval_examples, MAX_NEW_TOKENS)
    full_base_em, full_base_f1 = squad_scores(full_base_preds, eval_examples)
    del _base

    print(f"Base GPT-2: answer perplexity {full_base_ppl:.1f}, EM {full_base_em:.1f}, F1 {full_base_f1:.1f}")
    for _e, _p in list(zip(eval_examples, full_base_preds))[:3]:
        print(f"\nQ: {_e['question']}\n   gold: {_e['answers'][0]!r}\n   base: {_p!r}")
    return (
        full_base_em,
        full_base_f1,
        full_base_ppl,
        full_base_preds,
        full_collate,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Now we load a fresh copy and update **all** of its weights on the training set. With 124M parameters, the model, its gradients and the AdamW optimizer state (two extra values per weight) need about 2 GB of memory in total.
    """)
    return


@app.cell
def _(
    AutoModelForCausalLM,
    EPOCHS,
    FULL_BATCH_SIZE,
    FULL_FT_MODEL_ID,
    FULL_LR,
    device,
    full_collate,
    full_train,
    train_causal_lm,
):
    full_model = AutoModelForCausalLM.from_pretrained(FULL_FT_MODEL_ID).to(device)
    full_trainable = sum(p.numel() for p in full_model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {full_trainable:,} (100%)")

    full_losses = train_causal_lm(
        full_model, full_train, full_collate, lr=FULL_LR, epochs=EPOCHS, batch_size=FULL_BATCH_SIZE, desc="GPT-2 full FT"
    )
    full_model.save_pretrained("./checkpoints/gpt2-squad")  # ~500 MB: a complete copy of the model
    return full_losses, full_model, full_trainable


@app.cell
def _(
    MAX_NEW_TOKENS,
    answer_perplexity,
    eval_examples,
    full_base_ppl,
    full_collate,
    full_eval,
    full_model,
    full_tok,
    generate_answers,
    squad_scores,
):
    full_ft_ppl = answer_perplexity(full_model, full_eval, full_collate)
    full_ft_preds = generate_answers(full_model, full_tok, eval_examples, MAX_NEW_TOKENS)
    full_ft_em, full_ft_f1 = squad_scores(full_ft_preds, eval_examples)
    print(f"Fine-tuned GPT-2: answer perplexity {full_base_ppl:.1f} -> {full_ft_ppl:.1f}, EM {full_ft_em:.1f}, F1 {full_ft_f1:.1f}")
    return full_ft_em, full_ft_f1, full_ft_ppl, full_ft_preds


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. Part B: LoRA fine-tuning of a larger model

    A 1.5B-parameter model is about 12× larger than GPT-2. Fully fine-tuning it would need roughly 1.5B × 16 bytes ≈ 24 GB just for weights, gradients and AdamW state: too much for one 16 GB GPU.

    **LoRA** (Hu et al., 2021) freezes every pretrained weight matrix $W$ and learns a low-rank update next to it:

    $$W' = W + \frac{\alpha}{r} BA, \qquad A \in \mathbb{R}^{r \times d_{in}},\; B \in \mathbb{R}^{d_{out} \times r},\; r \ll d$$

    - $B$ starts at zero, so at the beginning the model is exactly the base model.
    - Only $A$ and $B$ get gradients and optimizer state. With rank $r = 16$ on every linear layer that is about 1% of the parameters.
    - The frozen weights can stay in half precision, and the trained adapter is a file of a few tens of MB instead of a full model copy.
    - Because the adapter can be switched off (`disable_adapter()`), the same object gives us both the base and the fine-tuned model.
    """)
    return


@app.cell
def _(
    AutoModelForCausalLM,
    LORA_ALPHA,
    LORA_MODEL_ID,
    LORA_RANK,
    LoraConfig,
    device,
    dtype,
    get_peft_model,
):
    _base = AutoModelForCausalLM.from_pretrained(LORA_MODEL_ID, dtype=dtype).to(device)
    # Recompute activations in the backward pass instead of storing them: less memory, a bit slower
    _base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    _base.config.use_cache = False

    lora_config = LoraConfig(
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        lora_dropout=0.05,
        target_modules="all-linear",  # attention and MLP projections (the LM head is excluded)
        task_type="CAUSAL_LM",
    )
    lora_model = get_peft_model(_base, lora_config)
    lora_trainable, lora_total = lora_model.get_nb_trainable_parameters()
    print(f"Trainable parameters: {lora_trainable:,} of {lora_total:,} ({100 * lora_trainable / lora_total:.2f}%)")
    return lora_model, lora_total, lora_trainable


@app.cell
def _(
    EPOCHS,
    LORA_BATCH_SIZE,
    LORA_GRAD_ACCUM,
    LORA_LR,
    lora_model,
    lora_tok,
    lora_train,
    make_collate,
    train_causal_lm,
):
    lora_collate = make_collate(lora_tok.pad_token_id)
    lora_losses = train_causal_lm(
        lora_model,
        lora_train,
        lora_collate,
        lr=LORA_LR,
        epochs=EPOCHS,
        batch_size=LORA_BATCH_SIZE,
        grad_accum=LORA_GRAD_ACCUM,
        desc="LoRA",
    )
    lora_model.save_pretrained("./checkpoints/qwen-squad-lora")  # saves only the adapter weights
    return lora_collate, lora_losses


@app.cell
def _(
    MAX_NEW_TOKENS,
    answer_perplexity,
    eval_examples,
    generate_answers,
    lora_collate,
    lora_eval,
    lora_losses,
    lora_model,
    lora_tok,
    squad_scores,
):
    assert lora_losses, "train the LoRA model first"
    # With the adapter switched off, this is exactly the original base model
    with lora_model.disable_adapter():
        lora_base_ppl = answer_perplexity(lora_model, lora_eval, lora_collate)
        lora_base_preds = generate_answers(lora_model, lora_tok, eval_examples, MAX_NEW_TOKENS)
    lora_base_em, lora_base_f1 = squad_scores(lora_base_preds, eval_examples)

    lora_ft_ppl = answer_perplexity(lora_model, lora_eval, lora_collate)
    lora_ft_preds = generate_answers(lora_model, lora_tok, eval_examples, MAX_NEW_TOKENS)
    lora_ft_em, lora_ft_f1 = squad_scores(lora_ft_preds, eval_examples)

    print(f"Base Qwen2.5:  answer perplexity {lora_base_ppl:.2f}, EM {lora_base_em:.1f}, F1 {lora_base_f1:.1f}")
    print(f"LoRA Qwen2.5:  answer perplexity {lora_ft_ppl:.2f}, EM {lora_ft_em:.1f}, F1 {lora_ft_f1:.1f}")
    return (
        lora_base_em,
        lora_base_f1,
        lora_base_ppl,
        lora_base_preds,
        lora_ft_em,
        lora_ft_f1,
        lora_ft_ppl,
        lora_ft_preds,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6. Comparing the four models

    The table puts the numbers together. Read the columns carefully:

    - **Answer perplexity** shows how much each fine-tuning run helped *its own* model. Don't compare it between GPT-2 and Qwen: they split text into different tokens.
    - **EM and F1** are computed on the generated text, so they compare all four models fairly.
    - **Trained parameters** and **saved size** show what each approach costs.
    """)
    return


@app.cell
def _(
    FULL_FT_MODEL_ID,
    LORA_MODEL_ID,
    full_base_em,
    full_base_f1,
    full_base_ppl,
    full_ft_em,
    full_ft_f1,
    full_ft_ppl,
    full_trainable,
    lora_base_em,
    lora_base_f1,
    lora_base_ppl,
    lora_ft_em,
    lora_ft_f1,
    lora_ft_ppl,
    lora_total,
    lora_trainable,
    pd,
):
    _gpt2 = FULL_FT_MODEL_ID.split("/")[-1]
    _big = LORA_MODEL_ID.split("/")[-1]
    results = pd.DataFrame(
        [
            (f"{_gpt2} (base)", "-", 0, full_base_ppl, full_base_em, full_base_f1),
            (f"{_gpt2} (full fine-tune)", "all weights", full_trainable, full_ft_ppl, full_ft_em, full_ft_f1),
            (f"{_big} (base)", "-", 0, lora_base_ppl, lora_base_em, lora_base_f1),
            (f"{_big} (LoRA)", f"adapters ({100 * lora_trainable / lora_total:.1f}%)", lora_trainable, lora_ft_ppl, lora_ft_em, lora_ft_f1),
        ],
        columns=["model", "what is trained", "trained parameters", "answer perplexity", "EM (%)", "F1 (%)"],
    ).round(2)
    results
    return (results,)


@app.cell
def _(full_losses, lora_losses, np, plt, results):
    _fig, (_ax1, _ax2) = plt.subplots(1, 2, figsize=(12, 4))

    def _smooth(values, k=20):
        values = np.asarray(values, dtype=float)
        if len(values) < k:
            return values
        return np.convolve(values, np.ones(k) / k, mode="valid")

    _ax1.plot(_smooth(full_losses), label="GPT-2, full fine-tune")
    _ax1.plot(_smooth(lora_losses), label="Qwen2.5-1.5B, LoRA")
    _ax1.set_xlabel("training batch")
    _ax1.set_ylabel("loss on answer tokens (moving average)")
    _ax1.set_title("Training loss")
    _ax1.grid(alpha=0.3)
    _ax1.legend()

    _x = np.arange(len(results))
    _ax2.bar(_x - 0.2, results["EM (%)"], width=0.4, label="exact match")
    _ax2.bar(_x + 0.2, results["F1 (%)"], width=0.4, label="F1")
    _ax2.set_xticks(_x)
    _ax2.set_xticklabels(results["model"], rotation=15, ha="right")
    _ax2.set_ylabel("%")
    _ax2.set_title("Held-out SQuAD questions")
    _ax2.grid(alpha=0.3, axis="y")
    _ax2.legend()
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Generations side by side

    The same held-out questions, answered greedily by each model. Look for: base models that continue the text instead of answering, answers that are right but phrased differently from the reference (F1 gives partial credit), and questions where only the larger model gets it right.
    """)
    return


@app.cell
def _(
    eval_examples,
    full_base_preds,
    full_ft_preds,
    lora_base_preds,
    lora_ft_preds,
    pd,
):
    comparison = pd.DataFrame(
        {
            "question": [e["question"] for e in eval_examples],
            "reference": [e["answers"][0] for e in eval_examples],
            "GPT-2 base": full_base_preds,
            "GPT-2 full FT": full_ft_preds,
            "Qwen base": lora_base_preds,
            "Qwen LoRA": lora_ft_preds,
        }
    )
    comparison.head(20)
    return


@app.cell
def _(
    MAX_NEW_TOKENS,
    full_model,
    full_tok,
    generate_answers,
    lora_losses,
    lora_model,
    lora_tok,
    mo,
):
    # Try your own question: edit the context and question and re-run this cell.
    assert lora_losses, "train the LoRA model first"
    my_example = {
        "context": (
            "The Transformer architecture was introduced in 2017 in the paper 'Attention Is All You Need' "
            "by researchers at Google. It replaced recurrence with self-attention, which made training on "
            "GPUs much faster. GPT-2, released by OpenAI in 2019, is a decoder-only transformer."
        ),
        "question": "Which company released GPT-2?",
    }
    _a_full = generate_answers(full_model, full_tok, [my_example], max_new_tokens=MAX_NEW_TOKENS)[0]
    _a_lora = generate_answers(lora_model, lora_tok, [my_example], max_new_tokens=MAX_NEW_TOKENS)[0]
    mo.md(f"**Q:** {my_example['question']}  \n**GPT-2 full FT:** {_a_full}  \n**Qwen LoRA:** {_a_lora}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 7. Saving and serving

    - Full fine-tuning saved a complete model to `gpt2-squad/` (about 500 MB).
    - LoRA saved only the adapter to `qwen-squad-lora/` (tens of MB). To use it, load the base model and attach the adapter, or merge it into the weights once with `merge_and_unload()`.

    A sketch of the endpoint for the class activity, loading the model **once** at startup:

    ```python
    from fastapi import FastAPI
    from pydantic import BaseModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    app = FastAPI()
    tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained("gpt2-squad").eval()

    class QARequest(BaseModel):
        question: str
        context: str
        max_new_tokens: int = 24

    @app.post("/generate_with_llm")
    def generate_with_llm(req: QARequest):
        prompt = f"Context: {req.context}\nQuestion: {req.question}\nAnswer:"
        enc = tokenizer(prompt, return_tensors="pt")
        out = model.generate(**enc, max_new_tokens=req.max_new_tokens, do_sample=False,
                             pad_token_id=tokenizer.eos_token_id)
        answer = tokenizer.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        return {"answer": answer.strip().split("\n")[0]}
    ```

    For the LoRA model: `PeftModel.from_pretrained(AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-1.5B"), "qwen-squad-lora")`.

    ## Exercises

    1. Fine-tune GPT-2 with LoRA instead of full fine-tuning (`target_modules=["c_attn", "c_proj", "c_fc"]`). How close does it get, with how many trainable parameters?
    2. Change the LoRA rank to 4 and 64. How do trainable parameters, training time and F1 change?
    3. Remove the −100 masking on the prompt (train on the whole sequence). What happens to answer perplexity and to the generations?
    4. Increase `N_TRAIN`, or train for a second epoch. When does F1 stop improving?
    5. Swap `LORA_MODEL_ID` for another model of similar size (for example `HuggingFaceTB/SmolLM2-1.7B`). How do its tokenizer and results differ?
    """)
    return


if __name__ == "__main__":
    app.run()
