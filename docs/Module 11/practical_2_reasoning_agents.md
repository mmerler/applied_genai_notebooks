# Practical 2: Reasoning and Agents

This interactive marimo notebook improves a small instruction-tuned model (Qwen3.5-2B) at inference time, without changing its weights. It compares greedy decoding with self-consistency (majority voting over sampled reasoning paths), wraps code generation in a verify-and-retry loop with unit tests, builds a ReAct agent with two tools, and scores the agent's episodes with a verifiable reward to compute the advantages GRPO would use.

!!! note "Related Resources"
    - [View on GitHub](https://github.com/mmerler/applied_genai_notebooks/blob/main/notebooks/Module_11_Practical_2_ReasoningAndAgents.py) - See the source code
    - [Jupyter version](https://github.com/mmerler/applied_genai_notebooks/blob/main/jupyter_notebooks/Module_11_Practical_2_ReasoningAndAgents.ipynb)
    - [Run Locally](../running_locally.md) - Instructions for running with GPU acceleration support
    - A single GPU with 16 GB of memory (for example a Colab T4) runs the full notebook; set `MODEL_ID = "mock"` to check the code on a laptop without a GPU.

!!! warning "Static preview — not interactive"
    The notebook below is a static export, so its sliders, dropdowns, and
    text inputs are display-only and will not recompute outputs. To run it
    interactively, click **Run** or **Edit** in the notebook, then choose one of:

    - **Open in molab** — run it in marimo's cloud-hosted notebook platform
    - **Run locally** — `uv add marimo`, then `uv run marimo edit <notebook>.py`
    - **Copy code** or **Download** the notebook

<iframe src="../../exports/Module_11_Practical_2_ReasoningAndAgents.html"
        style="width: 100%; height: 1000px; border: 1px solid #ddd; border-radius: 4px;"
        sandbox="allow-scripts allow-same-origin">
</iframe>
