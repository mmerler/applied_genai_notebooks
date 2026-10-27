import marimo

__generated_with = "0.23.9"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    import copy
    import math
    import os
    import random
    import time

    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torchvision
    import torchvision.transforms as transforms
    from torch.utils.data import DataLoader
    from torchvision.utils import make_grid

    return (
        DataLoader,
        F,
        copy,
        make_grid,
        math,
        nn,
        np,
        os,
        plt,
        random,
        time,
        torch,
        torchvision,
        transforms,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Module 8: Practical 3 — Diffusion Methods, Part 2

    ## DDPM versus DDIM sampling and classifier-free guidance

    This practical extends **Practical 2**. There we trained an *unconditional*
    diffusion model on flowers. Here we train a **class-conditional** noise
    predictor, learn to steer it with **classifier-free guidance**, and compare
    two different ways to run the learned reverse process.

    By the end of the practical, you will be able to:

    1. explain why DDPM sampling is stochastic and why DDIM can use fewer steps;
    2. implement both samplers against the same trained network;
    3. train classifier-free guidance (CFG) with condition dropout;
    4. measure the speed–quality–control trade-off using matched initial noise.

    > **Compute note:** the defaults train a small model for 5 epochs the first
    > time the notebook runs, then save a checkpoint so later runs start
    > instantly. This takes a few minutes on a GPU or Apple Silicon, and longer
    > on a CPU. You can train more epochs with the button in Section 4.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Data

    Practical 2 used Flowers102. That dataset has only about 10 training
    images per class, which is far too few to learn 102 separate conditional
    distributions. For a class-conditional model we want **many images per
    class** and **classes that look clearly different**, so that we can see
    whether guidance actually steers the model.

    We therefore use **Fashion-MNIST**: 60,000 training images in 10 clothing
    classes (6,000 per class), padded to $32 \times 32$. A small model learns
    it in a few epochs, and every class is easy to recognize by eye, which
    makes the sampler and guidance comparisons easy to judge.

    To try color images, set `DATASET = "CIFAR10"` below (50,000 images, 10
    classes). Expect blurrier samples: CIFAR-10 needs a larger model and many
    more epochs to look good.

    The images are mapped from $[0,1]$ to $[-1,1]$, so the clean data and the
    Gaussian noise live on comparable scales.
    """)
    return


@app.cell
def _():
    # Experiment configuration. These are the main knobs to change in class.
    DATASET = "FashionMNIST"  # or "CIFAR10"
    IMAGE_SIZE = 32
    BATCH_SIZE = 128
    DIFFUSION_STEPS = 1000
    BASE_CHANNELS = 32
    TIME_DIM = 128
    CONDITION_DROP_PROBABILITY = 0.10
    EMA_DECAY = 0.999
    LEARNING_RATE = 1e-3
    AUTO_TRAIN_EPOCHS = 5  # trained automatically on the first run
    CHECKPOINT_PATH = f"checkpoints/module8_practical3_cfg_{DATASET.lower()}.pt"
    return (
        AUTO_TRAIN_EPOCHS,
        BASE_CHANNELS,
        BATCH_SIZE,
        CHECKPOINT_PATH,
        CONDITION_DROP_PROBABILITY,
        DATASET,
        DIFFUSION_STEPS,
        EMA_DECAY,
        IMAGE_SIZE,
        LEARNING_RATE,
        TIME_DIM,
    )


@app.cell
def _(np, random, torch):
    def _seed_everything(seed=7):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    _seed_everything()
    device = torch.device(
        "mps"
        if torch.backends.mps.is_available()
        else "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )
    print(f"Using device: {device}")
    return (device,)


@app.cell
def _(DATASET, IMAGE_SIZE, torchvision, transforms):
    if DATASET == "FashionMNIST":
        _padding = (IMAGE_SIZE - 28) // 2
        # No horizontal flips: shoes and bags all face the same way in Fashion-MNIST
        _train_transform = transforms.Compose(
            [
                transforms.Pad(_padding),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),
            ]
        )
        _test_transform = _train_transform
        _dataset_class = torchvision.datasets.FashionMNIST
        NUM_CHANNELS = 1
    elif DATASET == "CIFAR10":
        _train_transform = transforms.Compose(
            [
                transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
        _test_transform = transforms.Compose(
            [
                transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
        _dataset_class = torchvision.datasets.CIFAR10
        NUM_CHANNELS = 3
    else:
        raise ValueError("DATASET must be 'FashionMNIST' or 'CIFAR10'")

    train_dataset = _dataset_class(
        root="./data", train=True, download=True, transform=_train_transform
    )
    validation_dataset = _dataset_class(
        root="./data", train=False, download=True, transform=_test_transform
    )
    CLASS_NAMES = list(train_dataset.classes)
    NUM_CLASSES = len(CLASS_NAMES)
    print(f"{DATASET}: {len(train_dataset):,} training and {len(validation_dataset):,} validation images")
    print("Classes:", CLASS_NAMES)
    return (
        CLASS_NAMES,
        NUM_CHANNELS,
        NUM_CLASSES,
        train_dataset,
        validation_dataset,
    )


@app.cell
def _(BATCH_SIZE, DataLoader, device, train_dataset, validation_dataset):
    _pin_memory = device.type == "cuda"
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=_pin_memory,
        drop_last=True,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=_pin_memory,
    )
    return train_loader, validation_loader


@app.cell
def _(CLASS_NAMES, make_grid, plt, train_dataset):
    # One example of each class
    _examples = []
    for _class_index in range(len(CLASS_NAMES)):
        _image_index = list(train_dataset.targets).index(_class_index)
        _examples.append(train_dataset[_image_index][0])
    _preview = make_grid([(_x + 1) / 2 for _x in _examples], nrow=len(CLASS_NAMES), padding=2, pad_value=1)
    _figure, _axis = plt.subplots(figsize=(12, 2))
    _axis.imshow(_preview.permute(1, 2, 0).clamp(0, 1))
    _axis.set_xticks([(_i + 0.5) * _preview.shape[2] / len(CLASS_NAMES) for _i in range(len(CLASS_NAMES))])
    _axis.set_xticklabels(CLASS_NAMES, fontsize=8)
    _axis.set_yticks([])
    _axis.set_title("One training example per class")
    _figure.tight_layout()
    _figure
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. The shared forward process

    DDPM and DDIM use the **same training objective** and the same forward
    noising process. Let

    $$\alpha_t = 1-\beta_t, \qquad \bar\alpha_t = \prod_{s=1}^{t}\alpha_s.$$

    A clean image can be moved directly to any noise level:

    $$x_t = \sqrt{\bar\alpha_t}\,x_0 + \sqrt{1-\bar\alpha_t}\,\epsilon, \qquad \epsilon\sim\mathcal N(0,I).$$

    The network learns to predict the sampled noise with the standard
    $\ell_2$ objective:

    $$\mathcal L = \mathbb E\left[\|\epsilon-\epsilon_\theta(x_t,t,y)\|_2^2\right].$$

    **Connection to Practical 2.** Practical 2 wrote the same forward process
    with a continuous time $t \in [0, 1]$ and two functions, the *signal rate*
    and the *noise rate*. They are exactly $\sqrt{\bar\alpha_t}$ and
    $\sqrt{1-\bar\alpha_t}$ here. Here we use the original DDPM formulation:
    1,000 discrete steps with a linear $\beta_t$ schedule. You have also
    already used one of the two samplers below: the `reverse_diffusion`
    method in Practical 2 is **DDIM with $\eta = 0$**.
    """)
    return


@app.cell
def _(nn, torch):
    def extract(values, timesteps, reference):
        """Gather one schedule value per item and reshape it for broadcasting."""
        gathered = values.gather(0, timesteps)
        return gathered.reshape(timesteps.shape[0], *([1] * (reference.ndim - 1)))


    class DiffusionSchedule(nn.Module):
        def __init__(self, steps=1000, beta_start=1e-4, beta_end=0.02):
            super().__init__()
            betas = torch.linspace(beta_start, beta_end, steps, dtype=torch.float32)
            alphas = 1.0 - betas
            alpha_bars = torch.cumprod(alphas, dim=0)
            alpha_bars_previous = torch.cat(
                [torch.ones(1, dtype=torch.float32), alpha_bars[:-1]]
            )

            # Mean and variance of q(x_{t-1} | x_t, x_0), used by the DDPM sampler
            posterior_variance = (
                betas * (1.0 - alpha_bars_previous) / (1.0 - alpha_bars)
            )
            posterior_mean_coef_clean = (
                betas * torch.sqrt(alpha_bars_previous) / (1.0 - alpha_bars)
            )
            posterior_mean_coef_noisy = (
                (1.0 - alpha_bars_previous) * torch.sqrt(alphas) / (1.0 - alpha_bars)
            )

            self.steps = steps
            self.register_buffer("betas", betas)
            self.register_buffer("alphas", alphas)
            self.register_buffer("alpha_bars", alpha_bars)
            self.register_buffer("alpha_bars_previous", alpha_bars_previous)
            self.register_buffer("sqrt_alpha_bars", torch.sqrt(alpha_bars))
            self.register_buffer(
                "sqrt_one_minus_alpha_bars", torch.sqrt(1.0 - alpha_bars)
            )
            self.register_buffer(
                "posterior_variance", posterior_variance.clamp(min=1e-20)
            )
            self.register_buffer("posterior_mean_coef_clean", posterior_mean_coef_clean)
            self.register_buffer("posterior_mean_coef_noisy", posterior_mean_coef_noisy)


    def q_sample(clean_images, timesteps, schedule, noise=None):
        if noise is None:
            noise = torch.randn_like(clean_images)
        signal = extract(schedule.sqrt_alpha_bars, timesteps, clean_images)
        noise_rate = extract(
            schedule.sqrt_one_minus_alpha_bars, timesteps, clean_images
        )
        return signal * clean_images + noise_rate * noise, noise

    return DiffusionSchedule, q_sample


@app.cell
def _(DIFFUSION_STEPS, DiffusionSchedule, device):
    schedule = DiffusionSchedule(steps=DIFFUSION_STEPS).to(device)
    return (schedule,)


@app.cell
def _(plt, schedule):
    _figure, _axis = plt.subplots(figsize=(9, 4))
    _axis.plot(schedule.sqrt_alpha_bars.cpu(), label=r"signal: $\sqrt{\bar\alpha_t}$")
    _axis.plot(
        schedule.sqrt_one_minus_alpha_bars.cpu(),
        label=r"noise: $\sqrt{1-\bar\alpha_t}$",
    )
    _axis.set(xlabel="diffusion step t", ylabel="rate", title="Linear beta schedule")
    _axis.grid(alpha=0.25)
    _axis.legend()
    _figure.tight_layout()
    _figure
    return


@app.cell
def _(device, make_grid, plt, q_sample, schedule, torch, train_dataset):
    _clean = torch.stack([train_dataset[_i][0] for _i in range(3)]).to(device)
    _noise = torch.randn_like(_clean)
    _times = [0, 99, 299, 599, 999]
    _corrupted = []
    for _image_index in range(_clean.shape[0]):
        for _t in _times:
            _time_tensor = torch.tensor([_t], device=device, dtype=torch.long)
            _x_t, _ = q_sample(_clean[_image_index:_image_index + 1], _time_tensor, schedule, noise=_noise[_image_index:_image_index + 1])
            _corrupted.append(_x_t.cpu())
    _corruption_grid = make_grid(
        torch.cat(_corrupted), nrow=len(_times), normalize=True, value_range=(-1, 1), padding=2, pad_value=1
    )
    _figure, _axis = plt.subplots(figsize=(8, 5))
    _axis.imshow(_corruption_grid.permute(1, 2, 0))
    _axis.set_title(f"Early, middle and late timesteps: t = {_times}")
    _axis.axis("off")
    _figure.tight_layout()
    _figure
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Classifier-free guidance

    A conditional model predicts $\epsilon_\theta(x_t,t,y)$. To obtain an
    unconditional prediction from the **same network**, we randomly replace
    the class label with a learned null label during training.

    At generation time, classifier-free guidance combines two predictions:

    $$\hat\epsilon = \epsilon_\theta(x_t,t,\varnothing) + w\left(\epsilon_\theta(x_t,t,y) - \epsilon_\theta(x_t,t,\varnothing)\right).$$

    - $w=0$: unconditional generation
    - $w=1$: ordinary conditional generation
    - $w>1$: stronger class adherence, often with less diversity or artifacts

    The U-Net below injects the **time and class embeddings into every residual
    block**, where they scale and shift the normalized features. Practical 2's
    U-Net took a simpler route: it resized the time embedding into an
    image-sized feature map and concatenated it once at the input. Both work;
    injecting the condition in every block is what modern diffusion
    architectures do, because every layer can then adapt to the noise level
    and the class.
    """)
    return


@app.cell
def _(F, math, nn, torch):
    class SinusoidalTimeEmbedding(nn.Module):
        def __init__(self, dimension):
            super().__init__()
            self.dimension = dimension

        def forward(self, timesteps):
            half = self.dimension // 2
            frequency_scale = math.log(10000) / max(half - 1, 1)
            frequencies = torch.exp(
                -frequency_scale
                * torch.arange(half, device=timesteps.device, dtype=torch.float32)
            )
            angles = timesteps.float()[:, None] * frequencies[None, :]
            embedding = torch.cat([angles.sin(), angles.cos()], dim=1)
            if self.dimension % 2:
                embedding = F.pad(embedding, (0, 1))
            return embedding


    class ConditionalResidualBlock(nn.Module):
        def __init__(self, in_channels, out_channels, condition_dim):
            super().__init__()
            self.norm1 = nn.GroupNorm(8, in_channels)
            self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
            self.condition_projection = nn.Sequential(
                nn.SiLU(), nn.Linear(condition_dim, 2 * out_channels)
            )
            self.norm2 = nn.GroupNorm(8, out_channels)
            self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
            self.skip = (
                nn.Conv2d(in_channels, out_channels, 1)
                if in_channels != out_channels
                else nn.Identity()
            )

        def forward(self, features, condition):
            hidden = self.conv1(F.silu(self.norm1(features)))
            # The condition (time + class) scales and shifts every feature map
            scale, shift = self.condition_projection(condition).chunk(2, dim=1)
            hidden = self.norm2(hidden)
            hidden = hidden * (1 + scale[:, :, None, None]) + shift[:, :, None, None]
            hidden = self.conv2(F.silu(hidden))
            return hidden + self.skip(features)


    class ConditionalUNet(nn.Module):
        def __init__(
            self,
            image_channels=1,
            num_classes=10,
            base_channels=32,
            time_dim=128,
        ):
            super().__init__()
            self.image_channels = image_channels
            self.num_classes = num_classes
            self.null_label = num_classes  # the extra "no class" label used by CFG

            self.time_embedding = nn.Sequential(
                SinusoidalTimeEmbedding(time_dim),
                nn.Linear(time_dim, time_dim),
                nn.SiLU(),
                nn.Linear(time_dim, time_dim),
            )
            self.class_embedding = nn.Embedding(num_classes + 1, time_dim)

            self.input_conv = nn.Conv2d(image_channels, base_channels, 3, padding=1)
            self.down1 = ConditionalResidualBlock(
                base_channels, base_channels, time_dim
            )
            self.downsample1 = nn.Conv2d(
                base_channels, base_channels, 4, stride=2, padding=1
            )
            self.down2 = ConditionalResidualBlock(
                base_channels, 2 * base_channels, time_dim
            )
            self.downsample2 = nn.Conv2d(
                2 * base_channels, 2 * base_channels, 4, stride=2, padding=1
            )

            self.middle1 = ConditionalResidualBlock(
                2 * base_channels, 4 * base_channels, time_dim
            )
            self.middle2 = ConditionalResidualBlock(
                4 * base_channels, 2 * base_channels, time_dim
            )

            self.up2 = ConditionalResidualBlock(
                4 * base_channels, 2 * base_channels, time_dim
            )
            self.up1 = ConditionalResidualBlock(
                3 * base_channels, base_channels, time_dim
            )
            self.output_norm = nn.GroupNorm(8, base_channels)
            self.output_conv = nn.Conv2d(base_channels, image_channels, 3, padding=1)
            nn.init.zeros_(self.output_conv.weight)
            nn.init.zeros_(self.output_conv.bias)

        def forward(self, noisy_images, timesteps, labels):
            condition = self.time_embedding(timesteps) + self.class_embedding(labels)

            first = self.down1(self.input_conv(noisy_images), condition)
            second = self.down2(self.downsample1(first), condition)
            hidden = self.middle1(self.downsample2(second), condition)
            hidden = self.middle2(hidden, condition)

            hidden = F.interpolate(hidden, size=second.shape[-2:], mode="nearest")
            hidden = self.up2(torch.cat([hidden, second], dim=1), condition)
            hidden = F.interpolate(hidden, size=first.shape[-2:], mode="nearest")
            hidden = self.up1(torch.cat([hidden, first], dim=1), condition)
            return self.output_conv(F.silu(self.output_norm(hidden)))

    return (ConditionalUNet,)


@app.cell
def _(
    BASE_CHANNELS,
    ConditionalUNet,
    LEARNING_RATE,
    NUM_CHANNELS,
    NUM_CLASSES,
    TIME_DIM,
    copy,
    device,
    torch,
):
    model = ConditionalUNet(
        image_channels=NUM_CHANNELS,
        num_classes=NUM_CLASSES,
        base_channels=BASE_CHANNELS,
        time_dim=TIME_DIM,
    ).to(device)
    ema_model = copy.deepcopy(model).to(device).eval()
    for _parameter in ema_model.parameters():
        _parameter.requires_grad_(False)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4
    )
    model_parameter_count = sum(_parameter.numel() for _parameter in model.parameters())
    print(f"Conditional U-Net parameters: {model_parameter_count:,}")
    return ema_model, model, optimizer


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Training with condition dropout

    The only CFG-specific change to the training loop is the label dropout.
    With probability `CONDITION_DROP_PROBABILITY`, a real class is replaced by
    `model.null_label`. The network therefore learns conditional and
    unconditional denoising in one set of weights.

    An **exponential moving average** (EMA) copy of the weights is used for
    validation and sampling. EMA usually produces more stable samples than the
    rapidly changing training weights. Note the *warm-up* in `update_ema`: with
    a fixed decay of 0.999, the average would still be 86% random
    initialization after 150 steps, and the EMA model would produce pure noise
    early in training. Ramping the decay up from about 0.1 avoids this.

    The validation loss uses a fixed random seed, so the same noise and
    timesteps are drawn every epoch and the losses are comparable.
    """)
    return


@app.cell
def _(F, nn, os, q_sample, torch):
    @torch.no_grad()
    def update_ema(ema_network, training_network, decay, step):
        # Warm-up: the effective decay grows from 0.1 towards `decay`
        effective_decay = min(decay, (1 + step) / (10 + step))
        for ema_parameter, parameter in zip(
            ema_network.parameters(), training_network.parameters()
        ):
            ema_parameter.mul_(effective_decay).add_(parameter, alpha=1.0 - effective_decay)


    def train_epoch(
        training_network,
        ema_network,
        loader,
        optimizer,
        schedule,
        device,
        condition_drop_probability,
        ema_decay,
        start_step,
    ):
        training_network.train()
        losses = []
        step = start_step
        # Asynchronous copies are only safe from pinned memory, which only CUDA uses
        non_blocking = device.type == "cuda"
        for images, labels in loader:
            images = images.to(device, non_blocking=non_blocking)
            labels = labels.to(device, non_blocking=non_blocking)
            timesteps = torch.randint(
                0, schedule.steps, (images.shape[0],), device=device
            )
            noisy_images, target_noise = q_sample(images, timesteps, schedule)

            dropped = torch.rand(labels.shape[0], device=device) < condition_drop_probability
            training_labels = labels.clone()
            training_labels[dropped] = training_network.null_label

            predicted_noise = training_network(noisy_images, timesteps, training_labels)
            loss = F.mse_loss(predicted_noise, target_noise)
            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"The training loss became {loss.item()} at step {step + 1}. "
                    "Nothing was saved. Try a lower LEARNING_RATE (e.g. 2e-4)."
                )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(training_network.parameters(), max_norm=1.0)
            optimizer.step()
            step += 1
            update_ema(ema_network, training_network, ema_decay, step)
            losses.append(loss.detach().item())
        return sum(losses) / len(losses), step


    @torch.no_grad()
    def validate_epoch(network, loader, schedule, device, seed=0):
        network.eval()
        # Draw timesteps and noise on the CPU with a fixed seed, so every
        # epoch is evaluated on exactly the same noisy images
        generator = torch.Generator().manual_seed(seed)
        losses = []
        for images, labels in loader:
            timesteps = torch.randint(0, schedule.steps, (images.shape[0],), generator=generator)
            noise = torch.randn(images.shape, generator=generator)
            images, labels = images.to(device), labels.to(device)
            timesteps, noise = timesteps.to(device), noise.to(device)
            noisy_images, _ = q_sample(images, timesteps, schedule, noise=noise)
            predicted_noise = network(noisy_images, timesteps, labels)
            losses.append(F.mse_loss(predicted_noise, noise).item())
        return sum(losses) / len(losses)


    def save_checkpoint(path, training_network, ema_network, optimizer, training_state, config):
        _checkpoint_directory = os.path.dirname(path)
        if _checkpoint_directory:
            os.makedirs(_checkpoint_directory, exist_ok=True)
        torch.save(
            {
                "epoch": training_state["epoch"],
                "step": training_state["step"],
                "history": training_state["history"],
                "model_state_dict": training_network.state_dict(),
                "ema_model_state_dict": ema_network.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": config,
            },
            path,
        )


    def load_checkpoint(path, training_network, ema_network, optimizer, device):
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        training_network.load_state_dict(checkpoint["model_state_dict"])
        ema_network.load_state_dict(checkpoint["ema_model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        return checkpoint

    return load_checkpoint, save_checkpoint, train_epoch, validate_epoch


@app.cell
def _(
    CHECKPOINT_PATH,
    device,
    ema_model,
    load_checkpoint,
    model,
    optimizer,
    os,
    torch,
):
    # training_state is a mutable dictionary: the training cell below updates it
    # in place, so pressing "Train more" several times keeps counting epochs
    # and keeps the full loss history.
    _checkpoint = None
    if os.path.exists(CHECKPOINT_PATH):
        _checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
        _weights = list(_checkpoint["ema_model_state_dict"].values())
        if not all(torch.isfinite(_w).all() for _w in _weights if _w.is_floating_point()):
            print(f"Ignoring {CHECKPOINT_PATH}: it contains NaN or infinite weights.")
            _checkpoint = None
    if _checkpoint is not None:
        _checkpoint = load_checkpoint(
            CHECKPOINT_PATH, model, ema_model, optimizer, device
        )
        training_state = {
            "epoch": int(_checkpoint["epoch"]),
            "step": int(_checkpoint.get("step", 0)),
            "history": list(_checkpoint.get("history", [])),
        }
        print(f"Loaded checkpoint after epoch {training_state['epoch']}: {CHECKPOINT_PATH}")
    else:
        training_state = {"epoch": 0, "step": 0, "history": []}
        print("No checkpoint found. The model will be trained below.")
    return (training_state,)


@app.cell
def _(mo):
    epoch_control = mo.ui.slider(
        start=1, stop=50, step=1, value=5, label="Additional epochs"
    )
    train_button = mo.ui.run_button(label="Train more")
    mo.hstack([epoch_control, train_button], justify="start", gap=2)
    return epoch_control, train_button


@app.cell
def _(
    AUTO_TRAIN_EPOCHS,
    CHECKPOINT_PATH,
    CONDITION_DROP_PROBABILITY,
    DATASET,
    EMA_DECAY,
    IMAGE_SIZE,
    NUM_CLASSES,
    device,
    ema_model,
    epoch_control,
    model,
    optimizer,
    save_checkpoint,
    schedule,
    time,
    train_button,
    train_epoch,
    train_loader,
    training_state,
    validate_epoch,
    validation_loader,
):
    # Train automatically on the first run (no checkpoint yet), and again
    # whenever "Train more" is pressed.
    if training_state["epoch"] == 0:
        _epochs_to_run = AUTO_TRAIN_EPOCHS
    elif train_button.value:
        _epochs_to_run = epoch_control.value
    else:
        _epochs_to_run = 0

    for _ in range(_epochs_to_run):
        _start = time.perf_counter()
        _train_loss, training_state["step"] = train_epoch(
            model,
            ema_model,
            train_loader,
            optimizer,
            schedule,
            device,
            CONDITION_DROP_PROBABILITY,
            EMA_DECAY,
            training_state["step"],
        )
        _validation_loss = validate_epoch(ema_model, validation_loader, schedule, device)
        training_state["epoch"] += 1
        training_state["history"].append(
            {
                "epoch": training_state["epoch"],
                "train_loss": _train_loss,
                "validation_loss": _validation_loss,
            }
        )
        print(
            f"Epoch {training_state['epoch']:03d} | train {_train_loss:.4f} "
            f"| validation (EMA) {_validation_loss:.4f} | {time.perf_counter() - _start:.0f}s"
        )
        save_checkpoint(
            CHECKPOINT_PATH,
            model,
            ema_model,
            optimizer,
            training_state,
            {
                "dataset": DATASET,
                "image_size": IMAGE_SIZE,
                "num_classes": NUM_CLASSES,
                "condition_drop_probability": CONDITION_DROP_PROBABILITY,
                "diffusion_steps": schedule.steps,
            },
        )
    if _epochs_to_run:
        print(f"Saved checkpoint: {CHECKPOINT_PATH}")

    # A snapshot of the history: cells that use it re-run after every training run
    training_history = list(training_state["history"])
    return (training_history,)


@app.cell
def _(mo, plt, training_history):
    if training_history:
        _figure, _axis = plt.subplots(figsize=(8, 4))
        _axis.plot([_row["epoch"] for _row in training_history], [_row["train_loss"] for _row in training_history], marker="o", label="train")
        _axis.plot(
            [_row["epoch"] for _row in training_history],
            [_row["validation_loss"] for _row in training_history],
            marker="o",
            label="validation (EMA model)",
        )
        _axis.set(xlabel="epoch", ylabel="MSE", title="Noise-prediction loss")
        _axis.grid(alpha=0.25)
        _axis.legend()
        _figure.tight_layout()
        _loss_display = _figure
    else:
        _loss_display = mo.md("*Train the model to display loss curves.*")
    _loss_display
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. DDPM versus DDIM

    Both samplers call the same EMA noise predictor, and both start by
    converting the predicted noise into a prediction of the clean image,

    $$\hat x_0 = \frac{x_t - \sqrt{1-\bar\alpha_t}\,\hat\epsilon}{\sqrt{\bar\alpha_t}},$$

    which is clipped to the data range $[-1, 1]$. They differ in how they use
    $\hat x_0$ to take the next step.

    One subtle detail: DDIM uses *both* $\hat x_0$ and $\hat\epsilon$ in its
    update, so after clipping $\hat x_0$ we recompute
    $\hat\epsilon = (x_t - \sqrt{\bar\alpha_t}\,\hat x_0)/\sqrt{1-\bar\alpha_t}$
    to keep the two consistent. Guidance amplifies $\hat\epsilon$, so with
    $w > 1$ the clipped and unclipped predictions can disagree a lot; skipping
    this step produces blocky, saturated images that get *worse* with more
    DDIM steps (try it: comment out those lines and rerun the benchmark).

    | Sampler | Reverse update | Typical implication |
    |---|---|---|
    | **DDPM** | Samples $x_{t-1}$ from the Gaussian posterior $q(x_{t-1} \mid x_t, \hat x_0)$, adding fresh noise at every step | Stochastic; normally uses all 1,000 training steps |
    | **DDIM** | Jumps from $t$ to any earlier time $s$: $x_s = \sqrt{\bar\alpha_s}\,\hat x_0 + \sqrt{1-\bar\alpha_s - \sigma^2}\,\hat\epsilon + \sigma z$, with $\sigma = 0$ when $\eta = 0$ | Can skip many steps; with $\eta=0$ the same initial noise always gives the same image |

    With $\eta = 1$ and all 1,000 steps, DDIM adds exactly the same amount of
    noise as DDPM.

    The comparisons below are controlled: they use the same network, class
    labels, guidance scale, and initial noise. Only the reverse solver
    changes.
    """)
    return


@app.cell
def _(torch):
    @torch.no_grad()
    def predict_noise_cfg(network, noisy_images, timesteps, labels, guidance_scale):
        null_labels = torch.full_like(labels, network.null_label)
        # Run the unconditional and conditional predictions in one batch
        doubled_images = torch.cat([noisy_images, noisy_images], dim=0)
        doubled_timesteps = torch.cat([timesteps, timesteps], dim=0)
        doubled_labels = torch.cat([null_labels, labels], dim=0)
        unconditional, conditional = network(
            doubled_images, doubled_timesteps, doubled_labels
        ).chunk(2)
        return unconditional + guidance_scale * (conditional - unconditional)


    def _initial_images(network, labels, image_size, initial_noise):
        if initial_noise is not None:
            return initial_noise.clone()
        return torch.randn(
            labels.shape[0],
            network.image_channels,
            image_size,
            image_size,
            device=labels.device,
        )


    @torch.no_grad()
    def ddpm_sample(
        network,
        schedule,
        labels,
        image_size,
        guidance_scale=3.0,
        initial_noise=None,
    ):
        """Ancestral DDPM sampling over every training timestep."""
        network.eval()
        batch_size = labels.shape[0]
        images = _initial_images(network, labels, image_size, initial_noise)

        for step in reversed(range(schedule.steps)):
            timesteps = torch.full(
                (batch_size,), step, device=labels.device, dtype=torch.long
            )
            predicted_noise = predict_noise_cfg(
                network, images, timesteps, labels, guidance_scale
            )
            alpha_bar = schedule.alpha_bars[step]
            predicted_clean = (
                images - torch.sqrt(1.0 - alpha_bar) * predicted_noise
            ) / torch.sqrt(alpha_bar)
            predicted_clean = predicted_clean.clamp(-1, 1)

            # Mean of the posterior q(x_{t-1} | x_t, x_0) with x_0 = predicted_clean
            posterior_mean = (
                schedule.posterior_mean_coef_clean[step] * predicted_clean
                + schedule.posterior_mean_coef_noisy[step] * images
            )
            if step > 0:
                images = posterior_mean + torch.sqrt(
                    schedule.posterior_variance[step]
                ) * torch.randn_like(images)
            else:
                images = posterior_mean
        return images.clamp(-1, 1)


    @torch.no_grad()
    def ddim_sample(
        network,
        schedule,
        labels,
        image_size,
        sampling_steps=50,
        guidance_scale=3.0,
        eta=0.0,
        initial_noise=None,
    ):
        """DDIM sampling; eta=0 is deterministic and eta>0 adds stochasticity."""
        if not 1 <= sampling_steps <= schedule.steps:
            raise ValueError("sampling_steps must be between 1 and schedule.steps")

        network.eval()
        batch_size = labels.shape[0]
        device = labels.device
        images = _initial_images(network, labels, image_size, initial_noise)
        selected = torch.linspace(
            schedule.steps - 1, 0, sampling_steps, device=device
        ).round().long()

        for index, step_tensor in enumerate(selected):
            step = int(step_tensor.item())
            previous_step = (
                int(selected[index + 1].item())
                if index + 1 < selected.numel()
                else -1
            )
            timesteps = torch.full(
                (batch_size,), step, device=device, dtype=torch.long
            )
            predicted_noise = predict_noise_cfg(
                network, images, timesteps, labels, guidance_scale
            )

            alpha_bar = schedule.alpha_bars[step]
            previous_alpha_bar = (
                schedule.alpha_bars[previous_step]
                if previous_step >= 0
                else torch.ones((), device=device)
            )
            predicted_clean = (
                images - torch.sqrt(1.0 - alpha_bar) * predicted_noise
            ) / torch.sqrt(alpha_bar)
            predicted_clean = predicted_clean.clamp(-1, 1)
            # Recompute the noise that is consistent with the clipped x0. Without
            # this, the "direction" term below keeps pointing at the unclipped x0,
            # and with strong guidance the error builds up over many steps.
            predicted_noise = (
                images - torch.sqrt(alpha_bar) * predicted_clean
            ) / torch.sqrt(1.0 - alpha_bar)

            sigma = eta * torch.sqrt(
                ((1.0 - previous_alpha_bar) / (1.0 - alpha_bar))
                * (1.0 - alpha_bar / previous_alpha_bar)
            )
            direction = torch.sqrt(
                torch.clamp(1.0 - previous_alpha_bar - sigma**2, min=0.0)
            ) * predicted_noise
            random_term = sigma * torch.randn_like(images) if eta > 0 else 0.0
            images = (
                torch.sqrt(previous_alpha_bar) * predicted_clean
                + direction
                + random_term
            )
        return images.clamp(-1, 1)

    return ddim_sample, ddpm_sample


@app.cell
def _(make_grid, plt):
    def show_sample_grid(images, nrow, title, row_labels=None, column_labels=None, figsize=None):
        """Plot a batch of images in [-1, 1] as a grid with optional row/column labels."""
        _grid = make_grid((images.cpu() + 1) / 2, nrow=nrow, padding=2, pad_value=1)
        _rows = (images.shape[0] + nrow - 1) // nrow
        _figure, _axis = plt.subplots(figsize=figsize or (1.2 * nrow + 1.5, 1.2 * _rows + 0.8))
        _axis.imshow(_grid.permute(1, 2, 0).clamp(0, 1))
        _cell_height = _grid.shape[1] / _rows
        _cell_width = _grid.shape[2] / nrow
        _axis.set_yticks([(_i + 0.5) * _cell_height for _i in range(_rows)] if row_labels else [])
        _axis.set_yticklabels(row_labels or [], fontsize=9)
        _axis.set_xticks([(_i + 0.5) * _cell_width for _i in range(nrow)] if column_labels else [])
        _axis.set_xticklabels(column_labels or [], fontsize=9)
        _axis.tick_params(length=0)
        for _spine in _axis.spines.values():
            _spine.set_visible(False)
        _axis.set_title(title)
        _figure.tight_layout()
        return _figure

    return (show_sample_grid,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Class-conditional samples

    One row per class, 8 samples each, generated with DDIM (50 steps,
    $\eta = 0$) and guidance $w = 3$. This grid is regenerated every time the
    model is trained further.
    """)
    return


@app.cell
def _(
    CLASS_NAMES,
    IMAGE_SIZE,
    NUM_CLASSES,
    ddim_sample,
    device,
    ema_model,
    schedule,
    show_sample_grid,
    torch,
    training_history,
):
    _per_class = 8
    _labels = torch.arange(NUM_CLASSES, device=device).repeat_interleave(_per_class)
    _generator = torch.Generator().manual_seed(0)
    _noise = torch.randn(
        len(_labels), ema_model.image_channels, IMAGE_SIZE, IMAGE_SIZE, generator=_generator
    ).to(device)
    _samples = ddim_sample(
        ema_model,
        schedule,
        _labels,
        IMAGE_SIZE,
        sampling_steps=50,
        guidance_scale=3.0,
        eta=0.0,
        initial_noise=_noise,
    )
    _epochs = training_history[-1]["epoch"] if training_history else 0
    show_sample_grid(
        _samples,
        nrow=_per_class,
        title=f"DDIM-50, guidance w = 3, after {_epochs} epochs",
        row_labels=CLASS_NAMES,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Try it yourself

    Pick a sampler, a class, a guidance scale and (for DDIM) a number of steps,
    then click **Generate**. The 8 images always start from the same initial
    noise, so you can compare settings directly. DDPM always uses all 1,000
    steps.
    """)
    return


@app.cell
def _(CLASS_NAMES, mo):
    sampler_control = mo.ui.dropdown(
        options=["DDIM", "DDPM"], value="DDIM", label="Sampler"
    )
    class_control = mo.ui.dropdown(
        options=CLASS_NAMES, value=CLASS_NAMES[0], label="Class"
    )
    guidance_control = mo.ui.slider(
        start=0.0, stop=10.0, step=0.5, value=3.0, label="CFG scale"
    )
    step_control = mo.ui.slider(
        start=5, stop=200, step=5, value=50, label="DDIM steps"
    )
    sample_button = mo.ui.run_button(label="Generate 8 images")
    mo.vstack(
        [
            mo.hstack([sampler_control, class_control], justify="start", gap=2),
            mo.hstack([guidance_control, step_control], justify="start", gap=2),
            sample_button,
        ]
    )
    return (
        class_control,
        guidance_control,
        sample_button,
        sampler_control,
        step_control,
    )


@app.cell
def _(
    CLASS_NAMES,
    IMAGE_SIZE,
    class_control,
    ddim_sample,
    ddpm_sample,
    device,
    ema_model,
    guidance_control,
    mo,
    sample_button,
    sampler_control,
    schedule,
    show_sample_grid,
    step_control,
    torch,
):
    if sample_button.value:
        _class_index = CLASS_NAMES.index(class_control.value)
        _labels = torch.full((8,), _class_index, device=device, dtype=torch.long)
        _generator = torch.Generator().manual_seed(42)
        _initial_noise = torch.randn(
            8,
            ema_model.image_channels,
            IMAGE_SIZE,
            IMAGE_SIZE,
            generator=_generator,
        ).to(device)
        if sampler_control.value == "DDPM":
            _samples = ddpm_sample(
                ema_model,
                schedule,
                _labels,
                IMAGE_SIZE,
                guidance_scale=guidance_control.value,
                initial_noise=_initial_noise,
            )
            _sample_title = "DDPM, 1,000 stochastic steps"
        else:
            _samples = ddim_sample(
                ema_model,
                schedule,
                _labels,
                IMAGE_SIZE,
                sampling_steps=step_control.value,
                guidance_scale=guidance_control.value,
                eta=0.0,
                initial_noise=_initial_noise,
            )
            _sample_title = f"DDIM, {step_control.value} deterministic steps"
        _sample_display = show_sample_grid(
            _samples,
            nrow=8,
            title=f"{_sample_title} | {class_control.value} | guidance {guidance_control.value:.1f}",
        )
    else:
        _sample_display = mo.md("*Choose settings and click **Generate 8 images**.*")
    _sample_display
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6. Controlled sampler benchmark

    Every row below starts from **the same initial noise** with the same
    class labels and guidance scale $w = 3$: DDPM with all 1,000 ancestral
    steps, then DDIM with 100, 50, 20 and 10 steps. The table reports the
    wall-clock time; the image grid lets you judge the quality.

    With classifier-free guidance, each sampling step evaluates the network
    **twice** (conditional and unconditional), so the number of network
    evaluations is twice the number of steps.

    Notice that the DDIM rows look alike, because DDIM with $\eta = 0$ is a
    deterministic map from the initial noise to an image, while DDPM injects
    fresh noise at every step and produces different images from the same
    starting point.

    **Do not treat pixel standard deviation as an image-quality score.** It is
    a quick collapse/saturation diagnostic. For a report-quality experiment,
    add FID or KID over many generated samples and repeat across random seeds.
    """)
    return


@app.cell
def _(
    CLASS_NAMES,
    IMAGE_SIZE,
    NUM_CLASSES,
    ddim_sample,
    ddpm_sample,
    device,
    ema_model,
    mo,
    schedule,
    show_sample_grid,
    time,
    torch,
    training_history,
):
    _benchmark_labels = torch.arange(NUM_CLASSES, device=device)
    _benchmark_generator = torch.Generator().manual_seed(123)
    _benchmark_noise = torch.randn(
        NUM_CLASSES,
        ema_model.image_channels,
        IMAGE_SIZE,
        IMAGE_SIZE,
        generator=_benchmark_generator,
    ).to(device)

    def _synchronize():
        if device.type == "cuda":
            torch.cuda.synchronize()
        elif device.type == "mps":
            torch.mps.synchronize()

    _benchmark_rows = []
    _benchmark_images = []
    _row_names = []
    for _sampler_name, _sampling_steps in [("DDPM", 1000), ("DDIM", 100), ("DDIM", 50), ("DDIM", 20), ("DDIM", 10)]:
        _synchronize()
        _start = time.perf_counter()
        if _sampler_name == "DDPM":
            _generated = ddpm_sample(
                ema_model,
                schedule,
                _benchmark_labels,
                IMAGE_SIZE,
                guidance_scale=3.0,
                initial_noise=_benchmark_noise,
            )
        else:
            _generated = ddim_sample(
                ema_model,
                schedule,
                _benchmark_labels,
                IMAGE_SIZE,
                sampling_steps=_sampling_steps,
                guidance_scale=3.0,
                eta=0.0,
                initial_noise=_benchmark_noise,
            )
        _synchronize()
        _elapsed = time.perf_counter() - _start
        _benchmark_rows.append(
            {
                "sampler": f"{_sampler_name}-{_sampling_steps}",
                "steps": _sampling_steps,
                "network evaluations (with CFG)": 2 * _sampling_steps,
                "seconds": round(_elapsed, 2),
                "images / second": round(NUM_CLASSES / _elapsed, 2),
                "pixel std (diagnostic)": round(_generated.std().item(), 3),
            }
        )
        _benchmark_images.append(_generated.cpu())
        _row_names.append(f"{_sampler_name}-{_sampling_steps}")

    benchmark_results = _benchmark_rows
    _figure = show_sample_grid(
        torch.cat(_benchmark_images),
        nrow=NUM_CLASSES,
        title="Same initial noise, guidance w = 3",
        row_labels=_row_names,
        column_labels=CLASS_NAMES,
        figsize=(14, 7.5),
    )
    mo.vstack([_figure, mo.ui.table(benchmark_results, selection=None)])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 7. Guidance-scale sweep

    Hold the initial noise, sampler (DDIM-50) and step count fixed, and vary
    only the guidance scale $w$, for three different classes. This isolates
    the effect of CFG on class adherence, diversity, contrast, and artifacts.
    Each class gets two different starting noises, so you can also see how
    diversity changes with $w$.
    """)
    return


@app.cell
def _(
    CLASS_NAMES,
    IMAGE_SIZE,
    ddim_sample,
    device,
    ema_model,
    schedule,
    show_sample_grid,
    torch,
    training_history,
):
    _sweep_scales = [0.0, 1.0, 3.0, 5.0, 7.5]
    _sweep_classes = [0, len(CLASS_NAMES) // 2, len(CLASS_NAMES) - 1]
    _seeds_per_class = 2
    _sweep_generator = torch.Generator().manual_seed(999)
    _rows = []
    _row_labels = []
    for _class_index in _sweep_classes:
        for _seed_index in range(_seeds_per_class):
            _noise = torch.randn(
                1, ema_model.image_channels, IMAGE_SIZE, IMAGE_SIZE, generator=_sweep_generator
            ).to(device)
            _label = torch.tensor([_class_index], device=device)
            for _scale in _sweep_scales:
                _rows.append(
                    ddim_sample(
                        ema_model,
                        schedule,
                        _label,
                        IMAGE_SIZE,
                        sampling_steps=50,
                        guidance_scale=_scale,
                        eta=0.0,
                        initial_noise=_noise,
                    ).cpu()
                )
            _row_labels.append(f"{CLASS_NAMES[_class_index]} #{_seed_index + 1}")

    show_sample_grid(
        torch.cat(_rows),
        nrow=len(_sweep_scales),
        title="DDIM-50, same noise in each row; only the guidance scale changes",
        row_labels=_row_labels,
        column_labels=[f"w = {_s}" for _s in _sweep_scales],
        figsize=(9, 10),
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 8. Class activity and deliverables

    Work in pairs and submit the following:

    1. **Train or load** the class-conditional model. Submit the noise schedule
       plot, the corrupted examples at early, middle and late timesteps, and
       the training and validation loss curves.
    2. **Compare samplers** with matched initial noise: DDPM-1000 and DDIM at
       100, 50, 20 and 10 steps. Submit the image grid and runtime table. At
       how few DDIM steps does quality visibly drop?
    3. **Sweep CFG** over $w\in\{0,1,3,5,7.5\}$ for at least three classes.
       Identify where stronger conditioning begins to reduce diversity or
       introduce artifacts. Is there a class that only looks right with
       $w > 1$?
    4. **Explain the trade-off** in 150–250 words. Which configuration would
       you deploy if latency mattered? Which would you use for exploration?

    ### Optional extensions

    - Set DDIM's $\eta>0$ and explain how stochasticity changes repeatability.
    - Replace the linear beta schedule with a cosine schedule (as in
      Practical 2) and compare the few-step DDIM results.
    - Change what the network predicts: train it to predict the clean image
      $x_0$, or the velocity $v = \sqrt{\bar\alpha_t}\,\epsilon - \sqrt{1-\bar\alpha_t}\,x_0$,
      instead of the noise. What changes in the two samplers?
    - Switch to `DATASET = "CIFAR10"`. How many epochs does it take before the
      classes become recognizable?
    - Add KID or FID over at least 1,000 samples and report uncertainty across
      three random seeds.

    ### Primary references

    - [Denoising Diffusion Probabilistic Models (Ho et al., 2020)](https://arxiv.org/abs/2006.11239)
    - [Denoising Diffusion Implicit Models (Song et al., 2020)](https://arxiv.org/abs/2010.02502)
    - [Classifier-Free Diffusion Guidance (Ho & Salimans, 2022)](https://arxiv.org/abs/2207.12598)
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Takeaways

    - **Training stays the same:** DDPM and DDIM share the forward process and
      noise-prediction objective.
    - **The sampler changes the path:** DDPM is ancestral and stochastic; DDIM
      can be deterministic and much faster with a reduced step schedule.
    - **CFG is a train-and-sample technique:** condition dropout teaches the
      null prediction; guidance combines null and conditional predictions.
    - **More guidance is not always better:** control, diversity, and visual
      fidelity must be evaluated together.
    """)
    return


if __name__ == "__main__":
    app.run()
