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
    # Module 11: Practical 2 - Reasoning and Agents

    The first practical changed a toy model's behavior with rewards. This one works at **inference time**: the weights never change. We take a small instruction-tuned model and make it more capable by spending more computation per question and by putting it in a loop with tools.

    1. **Self-consistency**: sample several reasoning paths and take the majority answer.
    2. **Verification loop**: generate code, run unit tests, feed the failures back, retry.
    3. **A ReAct agent**: a Thought → Action → Observation loop with two tools.
    4. **From traces to rewards**: score each episode, and compute the advantages GRPO would use.

    Each part maps to a slide in Module 11, and parts 3 and 4 use the vocabulary of Module 10: state, action, policy, reward.

    **Hardware.** One GPU with 16 GB of memory (for example a Colab T4) runs the notebook in roughly 10 to 15 minutes. With `MODEL_ID = "mock"` everything runs in seconds on a laptop, using a scripted stand-in for the model, so you can check the plumbing first.

    **Install:** `pip install -U torch transformers accelerate pandas matplotlib tqdm`
    """)
    return


@app.cell
def _():
    import ast
    import operator
    import random
    import re
    import subprocess
    import sys
    import tempfile
    from collections import Counter

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from tqdm.auto import tqdm

    random.seed(42)
    np.random.seed(42)
    return Counter, ast, np, operator, pd, plt, random, re, subprocess, sys, tempfile, tqdm


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Configuration

    The notebook needs a model that follows chat instructions. (The Qwen2.5-1.5B used in Module 9 is a *base* model: it continues text and will not follow these prompts.)

    | `MODEL_ID` | Size | Notes |
    |---|---|---|
    | `Qwen/Qwen3.5-2B` | 2B | Default. Trained for tool calling; loaded through the multimodal classes, text only here. Needs a recent `transformers`. |
    | `Qwen/Qwen3-1.7B` | 1.7B | Text-only alternative with the standard causal-LM classes. |
    | `Qwen/Qwen2.5-1.5B-Instruct` | 1.5B | Older and weaker; useful to see the techniques help a struggling model. |
    | `mock` | - | Scripted stand-in, no GPU. Its numbers mean nothing; it only exercises the code. |

    For a first check on a laptop, set `MODEL_ID = "mock"` or `QUICK_RUN = True`.
    """)
    return


@app.cell
def _():
    MODEL_ID = "Qwen/Qwen3.5-2B"  # "mock" for a quick no-GPU check of the code
    QUICK_RUN = False  # True: fewer problems and samples

    # How each model is loaded: "multimodal" = AutoProcessor + AutoModelForMultimodalLM, "causal" = AutoTokenizer + AutoModelForCausalLM
    MODEL_KIND = {"Qwen/Qwen3.5-2B": "multimodal"}.get(MODEL_ID, "causal")
    ENABLE_THINKING = False  # leave off: we want to see the effect of our own test-time techniques

    SAMPLE_TEMPERATURE = 0.8  # for self-consistency and for the group of rollouts in part 4
    N_PROBLEMS = 6 if QUICK_RUN else 12  # word problems in part 1
    N_SAMPLES = 5 if QUICK_RUN else 9  # reasoning paths sampled per problem (odd, so votes rarely tie)
    MAX_ATTEMPTS = 3  # generate-verify-revise attempts in part 2
    MAX_STEPS = 6  # Thought-Action-Observation steps per episode in part 3
    GROUP_SIZE = 4 if QUICK_RUN else 8  # rollouts per task in part 4
    return (
        ENABLE_THINKING,
        GROUP_SIZE,
        MAX_ATTEMPTS,
        MAX_STEPS,
        MODEL_ID,
        MODEL_KIND,
        N_PROBLEMS,
        N_SAMPLES,
        SAMPLE_TEMPERATURE,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The model, behind one function

    Everything below talks to the model through `chat(messages, temperature, max_new_tokens, n)`, which returns `n` completions as strings.

    - `temperature = 0` is greedy decoding: one deterministic answer.
    - `temperature > 0` samples, so `n` completions differ. That diversity is what self-consistency and GRPO need (Module 9: from logits to the next token).
    """)
    return


@app.cell
def _(MODEL_ID, MODEL_KIND):
    llm = None
    if MODEL_ID != "mock":
        import torch

        device = (
            torch.device("cuda")
            if torch.cuda.is_available()
            else torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
        )
        # Half precision on GPU: bfloat16 on recent GPUs, float16 on older ones such as the T4.
        if device.type == "cuda":
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        else:
            dtype = torch.float32

        if MODEL_KIND == "multimodal":
            from transformers import AutoModelForMultimodalLM, AutoProcessor

            _templater = AutoProcessor.from_pretrained(MODEL_ID)
            _tok = _templater.tokenizer
            _model = AutoModelForMultimodalLM.from_pretrained(MODEL_ID, dtype=dtype).to(device).eval()
        else:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            _templater = _tok = AutoTokenizer.from_pretrained(MODEL_ID)
            _model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=dtype).to(device).eval()
        if _tok.pad_token is None:
            _tok.pad_token = _tok.eos_token

        llm = {"model": _model, "tok": _tok, "templater": _templater, "torch": torch, "device": device}
        print(f"Loaded {MODEL_ID} on {device} ({dtype}), {sum(p.numel() for p in _model.parameters()) / 1e9:.2f}B parameters")
    else:
        print("Using the scripted mock model: results below only show that the code runs.")
    return (llm,)


@app.cell
def _(ENABLE_THINKING, MODEL_KIND, llm, mock_chat, re):
    def _encode(messages):
        """Apply the model's chat template and return input tensors."""
        if MODEL_KIND == "multimodal":  # these templates expect a list of typed parts
            messages = [{"role": m["role"], "content": [{"type": "text", "text": m["content"]}]} for m in messages]
        _kw = dict(add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt")
        try:
            enc = llm["templater"].apply_chat_template(messages, enable_thinking=ENABLE_THINKING, **_kw)
        except TypeError:  # template without a thinking switch
            enc = llm["templater"].apply_chat_template(messages, **_kw)
        return enc.to(llm["device"])

    def chat(messages, temperature=0.0, max_new_tokens=256, n=1):
        """Return n completions for a list of {'role', 'content'} messages."""
        if llm is None:
            return mock_chat(messages, temperature, n)
        enc = _encode(messages)
        gen = dict(max_new_tokens=max_new_tokens, pad_token_id=llm["tok"].pad_token_id)
        if temperature > 0:
            gen.update(do_sample=True, temperature=temperature, top_p=0.95, top_k=20, num_return_sequences=n)
        else:
            gen.update(do_sample=False)
        with llm["torch"].no_grad():
            out = llm["model"].generate(**enc, **gen)
        texts = llm["tok"].batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        texts = [re.sub(r"<think>.*?</think>", "", t, flags=re.S).strip() for t in texts]
        return texts if temperature > 0 else texts * n

    print(chat([{"role": "user", "content": "In one sentence: what is reinforcement learning?"}], max_new_tokens=60)[0])
    return (chat,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Self-consistency

    **Slide: Chain of thought and self-consistency.** We ask for step-by-step reasoning that ends in `Answer: <number>`, sample several reasoning paths at temperature > 0, and take the most common final answer. Correct reasoning tends to arrive at the same number; mistakes scatter.

    The problems need several arithmetic steps, enough for a 2B model to slip now and then. The reference answers are computed in code.
    """)
    return


@app.cell
def _(N_PROBLEMS):
    PROBLEMS = [
        ("A bakery bakes 148 loaves a day. It sells three quarters of them at $6 each and the rest at $4 each. What is the daily revenue in dollars?", 148 * 3 / 4 * 6 + 148 / 4 * 4),
        ("A train travels at 86 km/h for 3 hours and then at 64 km/h for 2.5 hours. How many kilometers does it cover in total?", 86 * 3 + 64 * 2.5),
        ("A shop buys 240 mugs at $3.50 each. It sells 85% of them at $9 each and the rest stay unsold. What is the profit in dollars?", 240 * 0.85 * 9 - 240 * 3.5),
        ("A tank holds 1200 liters. It loses 15% of its water in the first hour, then 96 liters per hour for the next 4 hours. How many liters are left?", 1200 * 0.85 - 96 * 4),
        ("In a class of 32 students, each reads 7 books in the fall and 9 in the spring, except 5 students who each read 4 fewer books in the spring. How many books does the class read in total?", 32 * 16 - 5 * 4),
        ("A rectangular field is 125 m long and 84 m wide. Fencing costs $18 per meter of perimeter and the gate costs $350 extra. What is the total cost in dollars?", 2 * (125 + 84) * 18 + 350),
        ("A saver deposits $275 every month for 2 years and then withdraws one third of the total. How many dollars remain?", 275 * 24 * 2 / 3),
        ("A recipe uses 3 eggs for every 8 pancakes. How many eggs are needed for 136 pancakes?", 136 / 8 * 3),
        ("A bus has 14 rows of 4 seats. Three rows are removed and 2 seats are added to each remaining row. How many seats does the bus have now?", (14 - 3) * (4 + 2)),
        ("A worker earns $23 per hour for the first 40 hours of a week and 1.5 times that rate for overtime. What is the pay in dollars for a 52-hour week?", 40 * 23 + 12 * 23 * 1.5),
        ("The first number is 46. The second is three times the first minus 18. The third is the sum of the first two divided by 2. What is the third number?", (46 + (3 * 46 - 18)) / 2),
        ("A printer prints 45 pages per minute. How many pages are still unprinted after 30 minutes of printing 12 reports of 138 pages each?", 12 * 138 - 45 * 30),
    ][:N_PROBLEMS]

    COT_SYSTEM = (
        "You are a careful math tutor. Solve the problem in at most 6 short steps, "
        "one calculation per line, plain text only (no LaTeX, no markdown). "
        "Finish with a last line of the form 'Answer: <number>' with no units. "
        "Example of the style: '148 * 3 / 4 = 111' then '111 * 6 = 666'."
    )
    return COT_SYSTEM, PROBLEMS


@app.cell
def _(COT_SYSTEM, Counter, chat, re):
    def extract_number(text):
        """The number after the last 'Answer:' (or inside \\boxed{}). None if the model never got there."""
        text = text.replace("**", "")
        found = re.findall(r"(?:Answer\s*[:=]|\\boxed\{)\s*\$?\s*(-?\d[\d,]*\.?\d*)", text, flags=re.I)
        if not found:
            return None
        try:
            return float(found[-1].replace(",", "").rstrip("."))
        except ValueError:
            return None

    def is_correct(pred, truth):
        return pred is not None and abs(pred - truth) < 1e-2

    def solve(problem, temperature=0.0, n=1):
        """n answers to one problem, as numbers."""
        msgs = [{"role": "system", "content": COT_SYSTEM}, {"role": "user", "content": problem}]
        return [extract_number(t) for t in chat(msgs, temperature=temperature, max_new_tokens=600, n=n)]

    def majority(answers):
        """Most common answer, ignoring failed extractions. Ties go to the answer seen first."""
        votes = Counter(a for a in answers if a is not None)
        return votes.most_common(1)[0][0] if votes else None

    return is_correct, majority, solve


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Before counting answers, read one complete reasoning path, so you know what the vote is made of. Only the number after `Answer:` is used; a path that never reaches that line counts as no answer.
    """)
    return


@app.cell
def _(COT_SYSTEM, PROBLEMS, chat):
    print(chat([{"role": "system", "content": COT_SYSTEM},
                {"role": "user", "content": PROBLEMS[0][0]}], max_new_tokens=600)[0])
    return


@app.cell
def _(Counter, N_SAMPLES, PROBLEMS, SAMPLE_TEMPERATURE, pd, solve, tqdm):
    # One greedy answer and N_SAMPLES sampled answers per problem. This is the slow cell of part 1.
    greedy_answers = [solve(q)[0] for q, _ in tqdm(PROBLEMS, desc="greedy")]
    sampled_answers = [solve(q, temperature=SAMPLE_TEMPERATURE, n=N_SAMPLES) for q, _ in tqdm(PROBLEMS, desc="sampled")]

    def _agreement(answers):
        votes = Counter(a for a in answers if a is not None)
        return f"{votes.most_common(1)[0][1]}/{len(answers)}" if votes else f"0/{len(answers)}"

    pd.DataFrame(
        {
            "truth": [t for _, t in PROBLEMS],
            "greedy": greedy_answers,
            "samples": [", ".join("?" if a is None else f"{a:g}" for a in s) for s in sampled_answers],
            "agreement": [_agreement(s) for s in sampled_answers],
        }
    )
    return greedy_answers, sampled_answers


@app.cell
def _(N_SAMPLES, PROBLEMS, greedy_answers, is_correct, majority, np, plt, random, sampled_answers):
    def vote_accuracy(k, trials=200):
        """Accuracy of a majority vote over k of the sampled answers, averaged over random subsets."""
        rng = random.Random(0)
        hits = 0
        for _ in range(trials):
            for (_, truth), answers in zip(PROBLEMS, sampled_answers):
                hits += is_correct(majority(rng.sample(answers, k)), truth)
        return hits / (trials * len(PROBLEMS))

    greedy_acc = np.mean([is_correct(a, t) for a, (_, t) in zip(greedy_answers, PROBLEMS)])
    ks = list(range(1, N_SAMPLES + 1, 2))
    vote_acc = [vote_accuracy(k) for k in ks]

    print(f"Greedy decoding:        {greedy_acc:.0%}")
    for _k, _a in zip(ks, vote_acc):
        print(f"Majority of {_k} sample{'s' if _k > 1 else ' '}:  {_a:.0%}")
    _n_greedy_none = sum(a is None for a in greedy_answers)
    _n_sampled_none = sum(a is None for s in sampled_answers for a in s)
    print(f"No final answer: greedy {_n_greedy_none}/{len(PROBLEMS)}, sampled {_n_sampled_none}/{len(PROBLEMS) * N_SAMPLES}")

    _fig, _ax = plt.subplots(figsize=(6, 3.5))
    _ax.plot(ks, vote_acc, marker="o", label="majority vote over k samples")
    _ax.axhline(greedy_acc, color="gray", linestyle="--", label="greedy, one answer")
    _ax.set_xlabel("k = sampled reasoning paths per problem")
    _ax.set_ylabel("accuracy")
    _ax.set_xticks(ks)
    _ax.set_ylim(0, 1.05)
    _ax.legend()
    _ax.set_title("Self-consistency: accuracy vs test-time compute")
    plt.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **What to look for.**

    - Greedy decoding is not the same as the model's most likely *answer*. Greedy picks the most likely next token at every step, which commits to one reasoning path. Sampling many paths and voting estimates which answer the model reaches most often, and that can differ: find the rows where greedy is wrong but most samples are right.
    - The curve flattens. Each extra sample costs the same tokens and buys less accuracy.
    - Voting fails when there is no consensus. Find a row with low agreement: no amount of voting recovers an answer the model rarely produces. Agreement is a usable confidence signal, and a low value is a reason to abstain or escalate.
    - Check the "No final answer" count first. If it is high, the model is running out of tokens, and the comparison says more about the token limit than about reasoning.

    With 12 problems, one problem is worth about 8 percentage points, so read the trend, not the decimals.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. A verification loop

    **Slide: Verification closes the loop at inference.** For code we do not need to vote: we can *check*. The model writes a function, we run unit tests, and on failure we send the error message back and ask again.

    The tests here are an **external verifier**. In RL with verifiable rewards the same tests would supply the reward during training; here they gate the answer at inference.

    The generated code runs in a separate Python process with a timeout. That contains crashes and infinite loops, but it is not a security sandbox: only run this with tasks you trust.
    """)
    return


@app.cell
def _():
    CODE_TASKS = [
        {
            "name": "is_palindrome",
            "spec": "Write a Python function is_palindrome(s) that returns True if the string s reads the same forwards and backwards, ignoring case and every character that is not a letter or digit.",
            "tests": [
                'assert is_palindrome("A man, a plan, a canal: Panama") is True',
                'assert is_palindrome("race a car") is False',
                'assert is_palindrome("") is True',
                'assert is_palindrome("No lemon, no melon!") is True',
            ],
        },
        {
            "name": "run_length_encode",
            "spec": 'Write a Python function run_length_encode(s) that compresses consecutive repeated characters as the character followed by its count, for example "aaabcc" becomes "a3b1c2". The empty string stays empty.',
            "tests": [
                'assert run_length_encode("aaabcc") == "a3b1c2"',
                'assert run_length_encode("") == ""',
                'assert run_length_encode("abc") == "a1b1c1"',
                'assert run_length_encode("zzzzzzzzzzzz") == "z12"',
            ],
        },
        {
            "name": "merge_intervals",
            "spec": "Write a Python function merge_intervals(intervals) that takes a list of [start, end] lists in any order and returns the merged intervals sorted by start. Intervals that overlap or touch (for example [1, 2] and [2, 3]) are merged.",
            "tests": [
                "assert merge_intervals([[1, 3], [2, 6], [8, 10]]) == [[1, 6], [8, 10]]",
                "assert merge_intervals([[5, 7], [1, 2], [2, 3]]) == [[1, 3], [5, 7]]",
                "assert merge_intervals([]) == []",
                "assert merge_intervals([[1, 10], [2, 3], [4, 5]]) == [[1, 10]]",
            ],
        },
        {
            "name": "roman_to_int",
            "spec": "Write a Python function roman_to_int(s) that converts a Roman numeral string such as 'MCMXCIV' to an integer.",
            "tests": [
                'assert roman_to_int("III") == 3',
                'assert roman_to_int("IX") == 9',
                'assert roman_to_int("LVIII") == 58',
                'assert roman_to_int("MCMXCIV") == 1994',
            ],
        },
    ]
    return (CODE_TASKS,)


@app.cell
def _(MAX_ATTEMPTS, chat, re, subprocess, sys, tempfile):
    CODE_SYSTEM = "You are a Python programmer. Reply with one Python code block containing only the requested function, and nothing else."

    def extract_code(text):
        blocks = re.findall(r"```(?:python)?\s*\n(.*?)```", text, flags=re.S)
        return (blocks[-1] if blocks else text).strip()

    def run_tests(code, tests, timeout=5):
        """The verifier. Returns (passed, message); the message is what the model gets to see."""
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
            f.write(code + "\n\n" + "\n".join(tests) + "\n")
        try:
            proc = subprocess.run([sys.executable, f.name], capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            return False, f"Timed out after {timeout} seconds (infinite loop?)"
        if proc.returncode == 0:
            return True, "all tests passed"
        lines = [ln for ln in proc.stderr.strip().splitlines() if ln.strip()]
        failing = next((ln.strip() for ln in lines if ln.strip().startswith("assert")), "")
        return False, (failing + "  ->  " if failing else "") + (lines[-1] if lines else "unknown error")

    def solve_with_verifier(task, max_attempts=MAX_ATTEMPTS, verbose=True):
        """Generate, verify, revise. Returns the number of attempts used, or None if all failed."""
        msgs = [{"role": "system", "content": CODE_SYSTEM}, {"role": "user", "content": task["spec"]}]
        for attempt in range(1, max_attempts + 1):
            reply = chat(msgs, max_new_tokens=400)[0]
            passed, message = run_tests(extract_code(reply), task["tests"])
            if verbose:
                print(f"  attempt {attempt}: {message}")
            if passed:
                return attempt
            # Feed the failure back: the conversation is the agent's state, and it now contains the error.
            msgs += [
                {"role": "assistant", "content": reply},
                {"role": "user", "content": f"That code failed a test:\n{message}\nFix the function and reply with the full corrected code block."},
            ]
        return None

    return (solve_with_verifier,)


@app.cell
def _(CODE_TASKS, MAX_ATTEMPTS, pd, solve_with_verifier):
    code_results = []
    for _task in CODE_TASKS:
        print(_task["name"])
        code_results.append({"task": _task["name"], "attempts": solve_with_verifier(_task)})

    _solved_first = sum(r["attempts"] == 1 for r in code_results)
    _solved_any = sum(r["attempts"] is not None for r in code_results)
    print(f"\nSolved on the first attempt: {_solved_first}/{len(CODE_TASKS)}")
    print(f"Solved within {MAX_ATTEMPTS} attempts:   {_solved_any}/{len(CODE_TASKS)}")
    pd.DataFrame(code_results).fillna("failed")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **What to look for.** The gap between "first attempt" and "within 3 attempts" is what the verifier bought. Notice what made it possible: a check that is independent of the model, and an error message specific enough to act on.

    The verifier also defines the task. A function that passes these four tests is "correct" as far as this loop can tell, whether or not it handles the cases the tests leave out (slide: *Outcome verification*).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. A ReAct agent

    **Slides: Anatomy of an agent; ReAct and the Thought-Action-Observation loop; Tool use.** The model now chooses its own actions. Each turn it writes a **Thought**, then an **Action** (a tool call). We run the tool and append the **Observation**. The loop ends when the model writes a `Final Answer`.

    The questions are about the catalog of a small camping store. The prices are made up, so the model cannot answer from memory: it has to use the tools.

    | Tool | Input | Returns |
    |---|---|---|
    | `lookup[item]` | an item name | its price and the number in stock |
    | `calculator[expression]` | arithmetic with `+ - * / ( )` | the result |
    """)
    return


@app.cell
def _(ast, operator):
    CATALOG = {
        "tent": {"price": 129.0, "stock": 14},
        "lantern": {"price": 24.5, "stock": 60},
        "sleeping bag": {"price": 68.0, "stock": 35},
        "stove": {"price": 47.25, "stock": 22},
        "kayak": {"price": 412.0, "stock": 6},
        "compass": {"price": 13.75, "stock": 80},
    }

    def lookup(item):
        key = item.strip().strip("'\"").lower()
        if key not in CATALOG and key.rstrip("s") in CATALOG:  # accept plurals
            key = key.rstrip("s")
        if key not in CATALOG:
            return f"No item named '{item.strip()}'. Items: {', '.join(CATALOG)}"
        return f"{key}: price ${CATALOG[key]['price']:.2f}, {CATALOG[key]['stock']} in stock"

    _OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv, ast.USub: operator.neg}

    def calculator(expression):
        """Arithmetic only: the expression is parsed and walked, never passed to eval()."""

        def ev(node):
            if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
                return node.value
            if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
                return _OPS[type(node.op)](ev(node.left), ev(node.right))
            if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
                return _OPS[type(node.op)](ev(node.operand))
            raise ValueError("only numbers and + - * / ( ) are allowed")

        try:
            return f"{ev(ast.parse(expression.replace('$', '').replace(',', ''), mode='eval').body):g}"
        except Exception as e:  # the error goes back to the model as an observation
            return f"Error: {e}"

    TOOLS = {"lookup": lookup, "calculator": calculator}
    _c = CATALOG
    AGENT_TASKS = [
        ("What is the total price of 3 lanterns and 2 tents?", 3 * _c["lantern"]["price"] + 2 * _c["tent"]["price"]),
        ("How many dollars more does a kayak cost than 4 sleeping bags?", _c["kayak"]["price"] - 4 * _c["sleeping bag"]["price"]),
        ("If I buy every stove the store has in stock, how much do I pay?", _c["stove"]["price"] * _c["stove"]["stock"]),
        ("I have $500. How much is left after buying 2 sleeping bags and 1 lantern?", 500 - 2 * _c["sleeping bag"]["price"] - _c["lantern"]["price"]),
        ("What is the average price of a tent, a kayak and a stove?", (_c["tent"]["price"] + _c["kayak"]["price"] + _c["stove"]["price"]) / 3),
        ("What is the combined value of all the lanterns and all the compasses in stock?", _c["lantern"]["price"] * _c["lantern"]["stock"] + _c["compass"]["price"] * _c["compass"]["stock"]),
    ]
    return AGENT_TASKS, CATALOG, TOOLS


@app.cell
def _(MAX_STEPS, TOOLS, chat, re):
    REACT_SYSTEM = """You answer questions about a camping store by using tools.

    Tools:
    lookup[item] returns the price of an item and how many are in stock.
    calculator[expression] evaluates arithmetic, for example calculator[3 * 24.5 + 10].

    Use exactly this format, one step at a time:
    Thought: what you need to do next
    Action: tool[input]
    After each Action you will receive an Observation. Never write the Observation yourself, and never guess a price.
    When you know the answer, write:
    Thought: I know the answer
    Final Answer: <number>

    Example
    Question: What do 2 compasses cost?
    Thought: I need the price of a compass.
    Action: lookup[compass]
    Observation: compass: price $13.75, 80 in stock
    Thought: Two compasses cost 2 * 13.75.
    Action: calculator[2 * 13.75]
    Observation: 27.5
    Thought: I know the answer
    Final Answer: 27.5""".replace("\n    ", "\n")

    def react(question, temperature=0.0, max_steps=MAX_STEPS, verbose=True):
        """One episode. Returns (final answer as text or None, trace as a list of (kind, text))."""
        scratchpad = ""  # everything the agent has thought, done and seen: its state
        trace = []
        for _ in range(max_steps):
            msgs = [{"role": "system", "content": REACT_SYSTEM}, {"role": "user", "content": f"Question: {question}\n{scratchpad}"}]
            reply = chat(msgs, temperature=temperature, max_new_tokens=200)[0]
            reply = reply.split("Observation:")[0].strip()  # the environment writes observations, not the model

            final = re.search(r"Final Answer:\s*(.+)", reply)
            action = re.search(r"Action:\s*(\w+)\s*\[(.*?)\]", reply, flags=re.S)
            if action and (not final or action.start() < final.start()):
                reply = reply[: action.end()]  # one action per step
                tool, arg = action.group(1), action.group(2)
                observation = TOOLS[tool](arg) if tool in TOOLS else f"Unknown tool '{tool}'. Tools: {', '.join(TOOLS)}"
                trace += [("step", reply), ("observation", observation)]
                scratchpad += f"{reply}\nObservation: {observation}\n"
                if verbose:
                    print(f"{reply}\nObservation: {observation}")
            elif final:
                trace.append(("final", reply))
                if verbose:
                    print(reply)
                return final.group(1).strip(), trace
            else:  # neither an action nor an answer: tell the model, and let it try again
                observation = "Invalid format. Write 'Action: tool[input]' or 'Final Answer: <number>'."
                trace += [("step", reply), ("observation", observation)]
                scratchpad += f"{reply}\nObservation: {observation}\n"
                if verbose:
                    print(f"{reply}\nObservation: {observation}")
        return None, trace  # out of steps

    return (react,)


@app.cell
def _(AGENT_TASKS, react):
    # One episode, printed in full: read it as Thought -> Action -> Observation.
    _answer, _trace = react(AGENT_TASKS[0][0])
    print(f"\nReference answer: {AGENT_TASKS[0][1]:g}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Map the trace to Module 10.**

    | RL concept | Here |
    |---|---|
    | Policy | the model, choosing the next thought and action from the scratchpad |
    | State | the question plus the scratchpad. The catalog itself is hidden; the agent sees only observations |
    | Action | `lookup[...]`, `calculator[...]`, or the final answer. A thought changes the scratchpad and nothing else |
    | Environment | the two Python functions and the loop around them |
    | Reward | nothing so far. We add it in part 4 |

    Note the line that cuts the reply at `Observation:`. Left alone, a small model will often continue the text and invent the tool's result. The harness, not the model, decides what the environment said.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. From traces to rewards

    **Slides: Training and evaluating agents; GRPO.** An episode becomes useful for evaluation, and for training, once it has a score. Ours is a **verifiable reward**: 1 if the final answer matches the reference, 0 otherwise.

    First we evaluate the greedy agent on every task. Then, for each task, we sample a **group** of rollouts at temperature > 0 and compute the advantage GRPO would assign to each:

    $$A_i = \frac{r_i - \text{mean}(r_1,\dots,r_G)}{\text{std}(r_1,\dots,r_G)}$$

    We stop there. A training step would raise the probability of the tokens in rollouts with positive advantage and lower it for the others (Module 10: policy gradients).
    """)
    return


@app.cell
def _(re):
    def reward(answer, truth):
        """Verifiable reward: 1 if the last number in the final answer matches the reference."""
        if answer is None:
            return 0.0
        nums = re.findall(r"-?\d[\d,]*\.?\d*", answer)
        try:
            return float(bool(nums) and abs(float(nums[-1].replace(",", "").rstrip(".")) - truth) < 0.01)
        except ValueError:
            return 0.0

    def n_tool_calls(trace):
        return sum(kind == "observation" for kind, _ in trace)

    return n_tool_calls, reward


@app.cell
def _(AGENT_TASKS, n_tool_calls, pd, react, reward):
    eval_rows = []
    for _q, _truth in AGENT_TASKS:
        _answer, _trace = react(_q, verbose=False)
        eval_rows.append({"question": _q, "answer": _answer, "truth": round(_truth, 2), "tool calls": n_tool_calls(_trace), "reward": reward(_answer, _truth)})

    eval_df = pd.DataFrame(eval_rows)
    print(f"Greedy agent: success rate {eval_df['reward'].mean():.0%}, {eval_df['tool calls'].mean():.1f} tool calls per episode")
    eval_df
    return


@app.cell
def _(AGENT_TASKS, GROUP_SIZE, SAMPLE_TEMPERATURE, n_tool_calls, np, pd, react, reward, tqdm):
    def grpo_advantages(rewards):
        r = np.array(rewards, dtype=float)
        return np.zeros_like(r) if r.std() == 0 else (r - r.mean()) / r.std()

    group_rows = []
    for _i, (_q, _truth) in enumerate(tqdm(AGENT_TASKS, desc="rollout groups")):
        _rollouts = [react(_q, temperature=SAMPLE_TEMPERATURE, verbose=False) for _ in range(GROUP_SIZE)]
        _rewards = [reward(a, _truth) for a, _ in _rollouts]
        for _g, ((_a, _tr), _r, _adv) in enumerate(zip(_rollouts, _rewards, grpo_advantages(_rewards))):
            group_rows.append({"task": _i, "rollout": _g, "answer": _a, "tool calls": n_tool_calls(_tr), "reward": _r, "advantage": round(float(_adv), 2)})

    group_df = pd.DataFrame(group_rows)
    _per_task = group_df.groupby("task")["reward"].mean()
    print("Mean reward per task over the group:", ", ".join(f"{v:.2f}" for v in _per_task))
    print(f"Tasks that give a training signal (rewards differ within the group): {int(((_per_task > 0) & (_per_task < 1)).sum())}/{len(_per_task)}")
    group_df
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **What to look for.**

    - A task where every rollout succeeds, or every rollout fails, has advantage 0 everywhere: **no contrast, no signal**. Only tasks at the edge of the model's ability teach it anything.
    - When rewards differ, read the answers and tool-call counts of the winners and losers. That difference is what a GRPO update would reinforce.
    - The reward only checks the final number. A rollout that guessed a price and happened to be right scores the same as one that looked it up.

    ## Exercises

    1. **Thinking vs voting.** Set `ENABLE_THINKING = True` (Qwen3 and Qwen3.5 support it) and raise `max_new_tokens` in `solve`. Compare greedy-with-thinking against majority voting without it. Which buys more accuracy per generated token?
    2. **A weaker verifier.** Delete all but the first test of each code task. Does the model "pass" more often? Check by hand whether the accepted functions are right.
    3. **Reward design.** Change `reward` to subtract 0.05 per tool call. How do the advantages in part 4 change, and what behavior would that push the agent toward? What could go wrong?
    4. **Reward hacking.** Add the question "What is the price of a canoe?" (not in the catalog) with any reference answer. What does the agent do? Design a reward that favors "the store does not sell canoes".
    5. **A third tool.** Add `stock_value[item]`, returning price times stock, and describe it in `REACT_SYSTEM`. Does the agent use it on the last task? How many tool calls does it save?
    6. **Longer horizon.** Write a question that needs five or more tool calls. How does the success rate change with the number of steps (slide: *Errors compound*)?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Appendix: the mock model

    Used only when `MODEL_ID = "mock"`. It recognizes which part of the notebook is calling and returns a scripted reply, with random mistakes when the temperature is above zero, so every code path can be exercised without a GPU. Nothing here is a language model.
    """)
    return


@app.cell(hide_code=True)
def _(AGENT_TASKS, CATALOG, CODE_TASKS, PROBLEMS, random, re):
    _GOOD = {
        "is_palindrome": "def is_palindrome(s):\n    t = [c.lower() for c in s if c.isalnum()]\n    return t == t[::-1]",
        "run_length_encode": "def run_length_encode(s):\n    out, i = '', 0\n    while i < len(s):\n        j = i\n        while j < len(s) and s[j] == s[i]:\n            j += 1\n        out += s[i] + str(j - i)\n        i = j\n    return out",
        "merge_intervals": "def merge_intervals(intervals):\n    out = []\n    for a, b in sorted(intervals):\n        if out and a <= out[-1][1]:\n            out[-1][1] = max(out[-1][1], b)\n        else:\n            out.append([a, b])\n    return out",
        "roman_to_int": "def roman_to_int(s):\n    v = {'I': 1, 'V': 5, 'X': 10, 'L': 50, 'C': 100, 'D': 500, 'M': 1000}\n    return sum(-v[c] if i + 1 < len(s) and v[c] < v[s[i + 1]] else v[c] for i, c in enumerate(s))",
    }
    _BAD = {  # a plausible first attempt with a bug, for two of the tasks
        "is_palindrome": "def is_palindrome(s):\n    s = s.lower()\n    return s == s[::-1]",
        "merge_intervals": "def merge_intervals(intervals):\n    out = []\n    for a, b in intervals:\n        if out and a <= out[-1][1]:\n            out[-1][1] = max(out[-1][1], b)\n        else:\n            out.append([a, b])\n    return out",
    }
    _rng = random.Random(7)

    def _plan(question):
        """The tool calls a careful agent would make for one of AGENT_TASKS."""
        items = [k for k in CATALOG if k in question.lower()]
        truth = dict(AGENT_TASKS)[question]
        return [f"Thought: I need the {i} data.\nAction: lookup[{i}]" for i in items] + [
            f"Thought: Now I can compute the result.\nAction: calculator[{truth!r} + 0]",
            f"Thought: I know the answer\nFinal Answer: {truth:g}",
        ]

    def _one(messages, temperature):
        if len(messages) < 2:
            return "Reinforcement learning trains an agent to choose actions that maximize reward. (mock reply)"
        system, user = messages[0]["content"], messages[1]["content"]
        if "math tutor" in system:
            truth = dict(PROBLEMS)[user]
            stable_error = sum(map(ord, user)) % 5 == 0  # a problem the mock always gets wrong, the same way
            if stable_error or (temperature > 0 and _rng.random() < 0.35):
                truth += 10 if stable_error else _rng.choice([-20, -6, 4, 12, 30])
            return f"Step by step, the result follows.\nAnswer: {truth:g}"
        if "Python programmer" in system:
            name = next(t["name"] for t in CODE_TASKS if t["name"] in user)
            first_try = len(messages) == 2
            return "```python\n" + (_BAD[name] if first_try and name in _BAD else _GOOD[name]) + "\n```"
        if "camping store" in system:
            question = re.search(r"Question: (.*)", user).group(1)
            plan, done = _plan(question), user.count("Observation:")
            if temperature > 0 and done == 0 and _rng.random() < 0.4:  # sometimes guess instead of looking up
                return f"Thought: I think I remember the prices.\nFinal Answer: {dict(AGENT_TASKS)[question] * _rng.choice([0.5, 0.9, 1.2]):g}"
            return plan[min(done, len(plan) - 1)]
        return "Reinforcement learning trains an agent to choose actions that maximize reward. (mock reply)"

    def mock_chat(messages, temperature=0.0, n=1):
        return [_one(messages, temperature) for _ in range(n)] if temperature > 0 else [_one(messages, 0.0)] * n

    return (mock_chat,)


if __name__ == "__main__":
    app.run()
