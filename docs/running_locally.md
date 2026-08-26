# Running Notebooks Locally

To run the notebooks with the practicals locally with GPU acceleration support, follow these steps:

## Prerequisites

- Python 3.11 or higher
- Git
- (Optional) NVIDIA GPU with CUDA support, or Apple Silicon Mac (M1/M2/M3) for GPU acceleration

## Installation Steps

### 1. Clone the Repository

```bash
git clone https://github.com/gurgentus/applied_genai_notebooks.git
cd applied_genai_notebooks
```

### 2. Install UV Package Manager

UV is a fast Python package manager that we use for dependency management:

```bash
pip install uv
```

### 3. Install Dependencies

Install all required dependencies including development tools:

```bash
uv sync
```

This will install all packages needed to run the notebooks, including PyTorch with GPU support if available.

## Running Notebooks

### Starting Marimo

To run a specific notebook:

```bash
marimo edit notebooks/Module_2_Practical_1_Probability.py
```

Or to run all Module 2 notebooks together:

```bash
marimo edit notebooks/Module_2_Practical_1_Probability.py \
            notebooks/Module_2_Practical_2_Word_Sampling.py \
            notebooks/Module_2_Practical_3_Word_Embeddings.py
```

### GPU Acceleration

PyTorch automatically detects and uses available GPU acceleration:

- **NVIDIA GPUs**: CUDA support for Windows/Linux systems with NVIDIA GPUs
- **Apple Silicon**: MPS (Metal Performance Shaders) support for M1/M2/M3 Macs
- **CPU Fallback**: Runs on CPU if no GPU is detected

You can verify GPU availability in the notebooks. For later modules with deep learning models (Modules 4+), GPU acceleration significantly improves training and inference speed.

#### Verifying GPU Support

In a notebook, you can check which device PyTorch is using:

```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"MPS available: {torch.backends.mps.is_available()}")
print(f"Device: {torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')}")
```

## Troubleshooting

### UV Installation Issues

If `uv sync` fails, try:

```bash
uv sync --extra dev
```

### GPU Not Detected

**For NVIDIA GPUs:**
- Ensure NVIDIA GPU drivers are installed
- CUDA toolkit installed (compatible with PyTorch version)
- Run `nvidia-smi` to verify GPU is accessible

**For Apple Silicon Macs:**
- Requires macOS 12.3 or later
- MPS support is automatically included with PyTorch 1.12+

### Module-Specific Dependencies

Some notebooks may require additional models or data:

- **Module 2, Practical 3**: Requires spacy language model
  ```bash
  uv run python -m spacy download en_core_web_lg
  ```

## Additional Resources

- [Marimo Documentation](https://docs.marimo.io/)
- [PyTorch GPU Support](https://pytorch.org/get-started/locally/)
- [Apple MPS Documentation](https://pytorch.org/docs/stable/notes/mps.html)
- [Project Repository](https://github.com/gurgentus/applied_genai_notebooks)
