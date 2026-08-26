# GAN Class Activity

!!! note "Related Marimo Notebook"
    This module builds upon concepts from the following interactive notebook:

    - [Module 6 Practical 1: Generative Adversarial Networks (GAN)](https://github.com/gurgentus/applied_genai_notebooks/blob/main/notebooks/Module_6_Practical_1_GAN.py)

As part of this activity we will add the gan model to the helper library we developed as part of Module 4.

You will encorporate appropriate code from Module6-GAN notebook into your helper_lib module. 

---

## 1. Model Module

In `model.py`, define your neural network model:

```python
import torch.nn as nn

def get_model(model_name):
    # TODO: define and return the appropriate model_name - one of: FCNN, CNN, EnhancedCNN, VAE, GAN
    return model
```

---

## 2. Trainer Module

In `trainer.py`, add train_gan function:

```python
import torch

def train_gan(model, data_loader, criterion, optimizer, device='cpu', epochs=10):
    # TODO: run several iterations of the training loop (based on epochs parameter) and return the model
    return model
```

## 3. Image Generator

In `generator.py`, add train_gan function:

```python
import torch

def generate_samples(model, device, num_samples=10):
    # TODO: generate num_samples points in the latent space, run the generator to construct the image, and plot the samples on a grid
    plt.show()
```

---

```


