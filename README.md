# Applied Generative AI Notebooks

These notebooks and activities cover fundamental concepts and practical implementations of modern generative AI techniques including neural networks, transformers, diffusion models, and reinforcement learning.

## 📖 Acknowledgments

Module 4-8 examples are adapted from:

**Foster, D. (2024).** *Generative Deep Learning: Teaching Machines to Paint, Write, Compose, and Play* (2nd ed.). O'Reilly Media.

## 📚 Course Modules

### Core Neural Network Foundations
- **Module 3: Practical 1** - Fully Connected Neural Networks (with checkpoint system)
- **Module 3: Practical 2** - Convolution Intuition  
- **Module 3: Practical 3** - Convolutional Neural Networks
- **Module 4: Practical** - Variational Autoencoders
- **Module 5: Practical** - GAN (Generative Adversarial Networks)
- **Module 6: Practical** - RNN (Recurrent Neural Networks)
- **Module 6: Practical** - Normalizing Flows

### Advanced Generative Methods  
- **Module 7: Practical 1** - Energy Based Methods
- **Module 7: Practical 2** - Diffusion Methods 
- **Module 8: Practical** - Transformer Architecture

### Reinforcement Learning
- **Module 9: Practical** - Basics of Reinforcement Learning
- **Module 10: Practical** - Reinforcement Learning and LLM

## ✨ Key Features

### Interactive Learning
All notebooks are built with [marimo](https://marimo.io) for:
- **Real-time execution** with immediate feedback
- **Interactive widgets** for parameter exploration  
- **Reproducible results** with proper random seed management

## 🚀 Running the Notebooks

### Marimo Notebooks (Interactive)
1. Install dependencies: `uv sync` or `pip install -r requirements.txt`
2. Run individual notebooks: `marimo edit notebooks/Module_X_Practical_Y.py`
3. Or run all notebooks: `marimo tutorial intro`

### Jupyter Notebooks (Traditional)
Jupyter versions are available in the `jupyter_notebooks/` directory:
```bash
# Launch Jupyter Lab
jupyter lab jupyter_notebooks/

# Or run a specific notebook
jupyter notebook jupyter_notebooks/Module_3_Practical_1_FCNN.ipynb
```

### HTML Notebooks (View-Only)
Static HTML versions are available in the `html_notebooks/` directory for easy viewing without installing Jupyter:
```bash
# Serve HTML notebooks locally
python -m http.server -d html_notebooks 8080

# Then open http://localhost:8080 in your browser
```

Or simply open the HTML files directly in any web browser.

### Checkpoint Usage
For models with checkpointing (FCNN, Diffusion):
```python
# Save checkpoints during training (automatic)
# Load specific epoch
load_checkpoint(model, optimizer, 'checkpoints/model_epoch_010.pth', device)
# Resume training from checkpoint
```

## 📖 Documentation

### Running the Documentation Site
The course documentation is built with MkDocs and includes class activities, setup guides, and project instructions.

```bash
# Install documentation dependencies
uv add mkdocs-material mkdocs-pdf-export-plugin

# Serve documentation locally
mkdocs serve
```

Access the documentation at `http://127.0.0.1:8000`

### Building Documentation
```bash
# Build static documentation
mkdocs build

# Serve built documentation
python -m http.server -d site
```

## 🧪 Testing

To test the export process, run `scripts/build.py` from the root directory.

```bash
python scripts/build.py
```

This will export all notebooks in a folder called `_site/` in the root directory. Then to serve the site, run:

```bash
python -m http.server -d _site
```

This will serve the site at `http://localhost:8000`.
