# Practical 2: Fine-Tuning an LLM

This interactive marimo notebook fine-tunes pretrained language models for question answering on SQuAD in two ways: full fine-tuning of GPT-2 (124M parameters) and LoRA fine-tuning of Qwen2.5-1.5B. It then compares the base and fine-tuned models on the same held-out questions using answer perplexity, exact match, F1 and side-by-side generations.

!!! note "Related Resources"
    - [View on GitHub](https://github.com/mmerler/applied_genai_notebooks/blob/main/notebooks/Module_9_Practical_2_finetuning.py) - See the source code
    - [Run Locally](../running_locally.md) - Instructions for running with GPU acceleration support
    - A single GPU with 16 GB of memory (for example a Colab T4) runs the full notebook; set `QUICK_RUN = True` for a fast check on a laptop.

!!! warning "Static preview — not interactive"
    The notebook below is a static export, so its sliders, dropdowns, and
    text inputs are display-only and will not recompute outputs. To run it
    interactively, click **Run** or **Edit** in the notebook, then choose one of:

    - **Open in molab** — run it in marimo's cloud-hosted notebook platform
    - **Run locally** — `uv add marimo`, then `uv run marimo edit <notebook>.py`
    - **Copy code** or **Download** the notebook

<iframe src="../../exports/Module_9_Practical_2_finetuning.html"
        style="width: 100%; height: 1000px; border: 1px solid #ddd; border-radius: 4px;"
        sandbox="allow-scripts allow-same-origin">
</iframe>
