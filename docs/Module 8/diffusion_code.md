# Diffusion Model Class Activity

!!! note "Related Marimo Notebooks"
    This module builds upon concepts from the following interactive notebooks:

    - [Module 8 Practical 1: Energy-Based Methods](https://github.com/gurgentus/applied_genai_notebooks/blob/main/notebooks/Module_8_Practical_1_EnergyBasedMethods.py)
    - [Module 8 Practical 2: Diffusion Methods](https://github.com/gurgentus/applied_genai_notebooks/blob/main/notebooks/Module_8_Practical_2_DiffusionMethods.py)

As part of this activity we will add the diffusion model to the helper library we developed as part of Module 4.

You will encorporate appropriate code from Module8-DiffusionMethods notebook into your helper_lib module. 

---

## 1. Model Module

In `model.py`, define your neural network model:

```python
import torch.nn as nn

def get_model(model_name):
    # TODO: define and return the appropriate model_name - one of: FCNN, CNN, EnhancedCNN, VAE, GAN, Diffusion
    return model
```

---

## 2. Trainer Module

In `trainer.py`, add train_diffusion function:

```python
import torch

def train_diffusion(model, data_loader, criterion, optimizer, device='cpu', epochs=10):
    # TODO: run several iterations of the training loop (based on epochs parameter) and return the model
    return model
```

## 3. Image Generator

In `generator.py`, add generate_samples function:

```python
import torch

def generate_samples(model, device, num_samples=10, diffusion_steps=100):
    # TODO: generate num_samples points from a standard normal distribution, run the reverse diffusion to construct the image, and plot the samples on a grid
    plt.show()
```

---

```


