# Helper Library Setup Guide for Neural Network Projects

!!! note "Related Marimo Notebooks"
    This module builds upon concepts from the following interactive notebooks:

    - [Module 4 Practical 1: Fully Connected Neural Networks (FCNN)](https://github.com/gurgentus/applied_genai_notebooks/blob/main/notebooks/Module_4_Practical_1_FCNN.py)
    - [Module 4 Practical 2: Convolutions](https://github.com/gurgentus/applied_genai_notebooks/blob/main/notebooks/Module_4_Practical_2_Convolutions.py)
    - [Module 4 Practical 3: Convolutional Neural Networks (CNN)](https://github.com/gurgentus/applied_genai_notebooks/blob/main/notebooks/Module_4_Practical_3_CNN.py)

As part of this activity we will create a helper library that encapsulates common functionalities for data loading, model training, and evaluation, reducing duplication across various neural network projects we will work on this semester. 

You will move appropriate code from Module4-FCNN and Module4-CNN notebooks to the newly created module.

---

## 1. Directory Structure

Set up your project structure as follows:

```
helper_lib/
├── __init__.py
├── data_loader.py
├── trainer.py
├── evaluator.py
├── model.py
├── checkpoints.py
└── utils.py
```

---

## 2. Data Loader Module

In `data_loader.py`, encapsulate data loading logic:

```python
import torch
from torchvision import datasets, transforms

def get_data_loader(data_dir, batch_size=32, train=True):
    # TODO: create the data loader
    return loader
```

---

## 3. Model Module

In `model.py`, define your neural network model:

```python
import torch.nn as nn

def get_model(model_name):
    # TODO: define and return the appropriate model_name - one of: FCNN, CNN, EnhancedCNN
    return model
```

---

## 4. Trainer Module

In `trainer.py`, abstract the training loop with checkpoint support:

```python
import torch
from tqdm import tqdm
from .checkpoints import save_checkpoint

def train_model(model, train_loader, val_loader, criterion, optimizer, device='cpu', epochs=10, checkpoint_dir='checkpoints'):
    """
    Enhanced training loop with checkpoint functionality
    
    TODO: Implement training loop that:
    1. Trains the model for specified epochs
    2. Tracks training and validation metrics
    3. Automatically saves checkpoints each epoch
    4. Saves the best performing model
    5. Returns the trained model
    
    Hint: Look at the FCNN notebook for checkpoint implementation examples
    """
    # TODO: Implement training loop with checkpoint saving
    return model
```

---

## 5. Checkpoint Module

In `checkpoints.py`, implement checkpoint saving and loading:

```python
import torch
import os

def save_checkpoint(model, optimizer, epoch, loss, accuracy, checkpoint_dir='checkpoints'):
    """
    Save model checkpoint
    
    TODO: Implement checkpoint saving that includes:
    1. Model state dict
    2. Optimizer state dict  
    3. Epoch number
    4. Loss and accuracy metrics
    5. Create checkpoint directory if needed
    
    Hint: Reference the FCNN notebook checkpoint implementation
    """
    # TODO: Implement checkpoint saving
    pass

def load_checkpoint(model, optimizer, checkpoint_path, device):
    """
    Load model checkpoint and restore training state
    
    TODO: Implement checkpoint loading that:
    1. Loads the checkpoint file
    2. Restores model and optimizer states
    3. Returns epoch, loss, and accuracy information
    
    Why save optimizer state? See FCNN notebook documentation!
    """
    # TODO: Implement checkpoint loading  
    pass
```

---

## 6. Evaluator Module

In `evaluator.py`, encapsulate evaluation metrics:

```python
import torch

def evaluate_model(model, data_loader, criterion, device='cpu'):
    # TODO: calculate average loss and accuracy on the test dataset
    return avg_loss, accuracy
```

---

## 7. Usage Example

Here's how to use your enhanced helper library with checkpoint functionality:

```python
from helper_lib.data_loader import get_data_loader
from helper_lib.trainer import train_model
from helper_lib.evaluator import evaluate_model
from helper_lib.model import get_model
from helper_lib.checkpoints import load_checkpoint
import torch.nn as nn
import torch.optim as optim

# Load data
train_loader = get_data_loader('data/train', batch_size=64)
val_loader = get_data_loader('data/val', batch_size=64, train=False)
test_loader = get_data_loader('data/test', batch_size=64, train=False)

# Initialize model and training components
model = get_model("CNN")
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Option 1: Train from scratch with checkpoint saving
trained_model = train_model(
    model, train_loader, val_loader, criterion, optimizer, 
    epochs=10, checkpoint_dir='checkpoints'
)

# Option 2: Resume training from a checkpoint
# load_checkpoint(model, optimizer, 'checkpoints/model_epoch_005.pth', device='cpu')
# trained_model = train_model(model, train_loader, val_loader, criterion, optimizer, epochs=10)

# Evaluate final model
avg_loss, accuracy = evaluate_model(trained_model, test_loader, criterion)
print(f"Test Accuracy: {accuracy:.2f}%")
```

---

## 8. Key Learning Objectives

By completing this activity, students will:

1. **Understand modular code organization** for ML projects
2. **Implement checkpoint systems** for training resilience  
3. **Learn the importance of optimizer state** in checkpoint saving
4. **Practice abstracting common ML patterns** into reusable components
5. **Experience professional ML development workflows**

### Checkpoint Benefits Students Will Discover:

- **Resume interrupted training** without starting over
- **Compare model performance** across different epochs  
- **Experiment with hyperparameters** using saved states
- **Implement early stopping** and best model selection
- **Debug training issues** by examining saved states


