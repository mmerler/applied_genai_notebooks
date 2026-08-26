import marimo

__generated_with = "0.23.9"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    import numpy as np
    import matplotlib.pyplot as plt
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torchvision
    import torchvision.transforms as transforms
    from torch.utils.data import DataLoader
    import torch.nn.functional as F

    return DataLoader, F, nn, np, plt, torch, torchvision, transforms


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # Module 7: Practical 2 - Diffusion Methods
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    Recall that the idea behind Diffusion Models is to map training data to a simple normal distribution by adding noise through a series of steps and then have a neural network **learn** the inverse process.

    In contrast to autoencoders, the first step is done through simple well defined functions and only the decoder is learned.
    """)
    return


@app.cell
def _(torch):
    # Check which GPU is available
    device = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f'Using device: {device}')
    return (device,)


@app.cell
def _(torchvision, transforms):
    # Prepare the Data
    transform = transforms.Compose([
        transforms.CenterCrop(340),
        transforms.Resize((64, 64)),
        transforms.ToTensor()
        # transforms.Normalize((0.5,), (0.5,))
    ])
    train_dataset = torchvision.datasets.Flowers102(root='./data', split='train', download=True, transform=transform)
    test_dataset = torchvision.datasets.Flowers102(root='./data', split='val', download=True, transform=transform)
    return test_dataset, train_dataset


@app.cell
def _(DataLoader, test_dataset, train_dataset):
    BATCH_SIZE = 64
    IMAGE_SIZE = 64
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    return BATCH_SIZE, IMAGE_SIZE, test_loader, train_loader


@app.cell
def _(train_dataset):
    first_image, first_label = train_dataset[0]
    print(first_image.shape)
    return first_image, first_label


@app.cell
def _(first_image, first_label, plt):
    def show_image(image, label=None):
        print("Label: ", label)
        plt.figure(figsize=(4, 3)) 
        plt.imshow(image.permute(1,2,0).squeeze(), cmap='gray')
        plt.show()
    show_image(first_image, first_label)
    return (show_image,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Let's take $x_0$ to be a random variable representing images in our data distribution. We will assume that it has mean $0$ and variance $1$. This is ok since we have preprocessed the training images to have mean $0$ and variance $1$.

    Next, we will corrupt the images by adding standard gaussian noise $\epsilon$ (mean 0, variance 1). How much noise should be added?

    To keep the transformed distribution having the same mean and variance, we can control the noise amount using a parameter $\beta$:

    $x_1 = \sqrt{1-\beta}x_0 + \sqrt{\beta} \epsilon$

    Then if $x_0$ has mean $0$ and variance $1$, $x_1$ will have mean:

    $\sqrt{1-\beta}*0+\sqrt{\beta}*0=0$

    and variance:

    $(\sqrt{1-\beta})^2*1 + (\sqrt{\beta})^2*1=1-\beta+\beta=1$.
    """)
    return


@app.cell
def _(np):
    def corrupt_image(image, beta = 0.1):
        image_with_noise = np.sqrt(1-beta)*image + np.sqrt(beta)*np.random.normal(0, 1, image.shape)
        return image_with_noise
    # show_image(corrupt_image(first_image))
    return (corrupt_image,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    How does changing $\beta$ influence the amount of noise added to the image?  Try to find a value that makes the image look like a random noise image.  You can use the slider below to change the value of $\beta$.
    """)
    return


@app.cell
def _(mo):
    slider = mo.ui.slider(start=0, stop=1, step=0.05)
    slider
    return (slider,)


@app.cell
def _(corrupt_image, first_image, show_image, slider):
    show_image(corrupt_image(first_image, beta=slider.value))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We can repeat this process using the new $x_1$ as the input to the next step, and so on, until we reach $x_T$. Moreover, we can also use a different $\beta$ for each step. The setup of how this parameter should vary is referred to as the **diffusion schedule**.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We'll want to repeat this process through several iterations, at each step corrupting the image by a small amount. To make things simpler, we can perform some mathematical magic using the fact that multiplying a Gaussian distribution by a constant results in another Gaussian distribution. Similarly, adding two Gaussians results in a Gaussian.  So, one can show mathematically that for two standard Gaussian random variables:

    $A \epsilon_0 + B \epsilon_1 = \left(\sqrt{A^2 + B^2}\right)\epsilon$, where $\epsilon$ is also a Gaussian random variable.

    Hence,

    $x_2 = \sqrt{1-\beta_1}x_1 + \sqrt{\beta_1} \epsilon_1 = \sqrt{1-\beta_1}(\sqrt{1-\beta_0}x_0 + \sqrt{\beta_0} \epsilon_0) + \sqrt{\beta_1} \epsilon_1 = \sqrt{(1-\beta_0)(1-\beta_1)}x_0 + \sqrt{\beta_0(1-\beta_1)}\epsilon_0 + \sqrt{\beta_1}\epsilon_1 = \sqrt{(1-\beta_0)(1-\beta_1)}x_0 + \sqrt{1 - (1-\beta_0)(1-\beta_1)}\epsilon$,

    and so on. This will be used in defining the multipliers for the image (signal rates) and the noise (noise rates).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Let's look at examples of several diffusion schedules:
    """)
    return


@app.cell
def _(math, torch):
    def linear_diffusion_schedule(diffusion_times, min_rate=1e-4, max_rate=0.02):
        """
        diffusion_times: Tensor of shape (T,) with values in [0, 1)
        Returns:
            noise_rates: Tensor of shape (T,)
            signal_rates: Tensor of shape (T,)
        """
        diffusion_times = diffusion_times.to(dtype=torch.float32)
        betas = min_rate + diffusion_times * (max_rate - min_rate)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)

        signal_rates = torch.sqrt(alpha_bars)
        noise_rates = torch.sqrt(1.0 - alpha_bars)
        return noise_rates, signal_rates


    def cosine_diffusion_schedule(diffusion_times):
        # diffusion_times: Tensor of shape [T] or [B] with values in [0, 1]
        signal_rates = torch.cos(diffusion_times * math.pi / 2)
        noise_rates = torch.sin(diffusion_times * math.pi / 2)
        return noise_rates, signal_rates

    def offset_cosine_diffusion_schedule(diffusion_times, min_signal_rate=0.02, max_signal_rate=0.95):
        # Flatten diffusion_times to handle any shape
        original_shape = diffusion_times.shape
        diffusion_times_flat = diffusion_times.flatten()

        # Compute start and end angles from signal rate bounds
        start_angle = torch.acos(torch.tensor(max_signal_rate, dtype=torch.float32, device=diffusion_times.device))
        end_angle = torch.acos(torch.tensor(min_signal_rate, dtype=torch.float32, device=diffusion_times.device))

        # Linearly interpolate angles
        diffusion_angles = start_angle + diffusion_times_flat * (end_angle - start_angle)

        # Compute signal and noise rates
        signal_rates = torch.cos(diffusion_angles).reshape(original_shape)
        noise_rates = torch.sin(diffusion_angles).reshape(original_shape)

        return noise_rates, signal_rates

    return linear_diffusion_schedule, offset_cosine_diffusion_schedule


@app.cell
def _(BATCH_SIZE, IMAGE_SIZE, linear_diffusion_schedule, torch, train_loader):
    images, _ = next(iter(train_loader))
    noises = torch.randn(size=(BATCH_SIZE, 1, IMAGE_SIZE, IMAGE_SIZE))

    diffusion_times = torch.rand((BATCH_SIZE,), device=images.device)  # scalar per sample
    diffusion_times = diffusion_times.view(BATCH_SIZE, 1, 1, 1)         # for broadcasting

    noise_rates, signal_rates = linear_diffusion_schedule(diffusion_times)
    noisy_images = signal_rates * images + noise_rates * noises
    return (noisy_images,)


@app.cell
def _(noisy_images, plt):
    from torchvision.utils import make_grid
    grid = make_grid(noisy_images, normalize=True)
    plt.imshow(grid.permute(1, 2, 0))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The idea is to train a neural network to model the noise that was added. The input to this network will be the noisy image and the variance of the noise that was added (this comes from the diffusion schedule we select). The noise variance is typically encoded using some sort of an embedding, in this case a Sinusoidal Embedding.
    """)
    return


@app.cell
def _(nn, torch):
    import math

    class SinusoidalEmbedding(nn.Module):
        def __init__(self, num_frequencies=16):
            super().__init__()
            self.num_frequencies = num_frequencies
            frequencies = torch.exp(torch.linspace(math.log(1.0), math.log(1000.0), num_frequencies))
            self.register_buffer("angular_speeds", 2.0 * math.pi * frequencies.view(1, 1, 1, -1))

        def forward(self, x):
            """
            x: Tensor of shape (B, 1, 1, 1)
            returns: Tensor of shape (B, 1, 1, 2 * num_frequencies)
            """
            x = x.expand(-1, 1, 1, self.num_frequencies)
            sin_part = torch.sin(self.angular_speeds * x)
            cos_part = torch.cos(self.angular_speeds * x)
            return torch.cat([sin_part, cos_part], dim=-1)         

    return SinusoidalEmbedding, math


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Next we build the three new layers used in a UNet neural network, a popular architecture used in Diffusion Models.
    """)
    return


@app.cell
def _(F, nn, torch):
    class ResidualBlock(nn.Module):
        def __init__(self, in_channels, out_channels):
            super().__init__()
            self.needs_projection = in_channels != out_channels
            if self.needs_projection:
                self.proj = nn.Conv2d(in_channels, out_channels, kernel_size=1)
            else:
                self.proj = nn.Identity()

            self.norm = nn.BatchNorm2d(in_channels, affine=False)
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
            self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        def swish(self, x):
            return x * torch.sigmoid(x)

        def forward(self, x):
            residual = self.proj(x)
            x = self.swish(self.conv1(x))
            x = self.conv2(x)
            return x + residual

    class DownBlock(nn.Module):
        def __init__(self, width, block_depth, in_channels):
            super().__init__()
            self.blocks = nn.ModuleList()
            for i in range(block_depth):
                self.blocks.append(ResidualBlock(in_channels, width))
                in_channels = width
            self.pool = nn.AvgPool2d(kernel_size=2)

        def forward(self, x, skips):
            for block in self.blocks:
                x = block(x)
                skips.append(x)
            x = self.pool(x)
            return x

    class UpBlock(nn.Module):
        def __init__(self, width, block_depth, in_channels):
            super().__init__()
            self.blocks = nn.ModuleList()
            for _ in range(block_depth):
                self.blocks.append(ResidualBlock(in_channels + width, width))
                in_channels = width

        def forward(self, x, skips):
            x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
            for block in self.blocks:
                skip = skips.pop()
                x = torch.cat([x, skip], dim=1)
                x = block(x)
            return x

    return DownBlock, ResidualBlock, UpBlock


@app.cell
def _(DownBlock, F, ResidualBlock, SinusoidalEmbedding, UpBlock, nn, torch):
    class UNet(nn.Module):
        def __init__(self, image_size, num_channels, embedding_dim=32):
            super().__init__()
            self.initial = nn.Conv2d(num_channels, 32, kernel_size=1)
            self.num_channels = num_channels
            self.image_size = image_size
            self.embedding_dim = embedding_dim
            self.embedding = SinusoidalEmbedding(num_frequencies=16)
            self.embedding_proj = nn.Conv2d(embedding_dim, 32, kernel_size=1)

            self.down1 = DownBlock(32, in_channels=64, block_depth=2)
            self.down2 = DownBlock(64, in_channels=32, block_depth=2)
            self.down3 = DownBlock(96, in_channels=64, block_depth=2) 

            self.mid1 = ResidualBlock(in_channels=96, out_channels=128)
            self.mid2 = ResidualBlock(in_channels=128, out_channels=128)

            self.up1 = UpBlock(96, in_channels=128, block_depth=2) 
            self.up2 = UpBlock(64, block_depth=2, in_channels=96)
            self.up3 = UpBlock(32, block_depth=2, in_channels=64)

            self.final = nn.Conv2d(32, num_channels, kernel_size=1)
            nn.init.zeros_(self.final.weight)  # Keep zero init like TF reference

        def forward(self, noisy_images, noise_variances):
            skips = []
            x = self.initial(noisy_images)
            noise_emb = self.embedding(noise_variances)  # shape: (B, 1, 1, 32)
            # Upsample to match image size 
            noise_emb = F.interpolate(noise_emb.permute(0, 3, 1, 2), size=(self.image_size, self.image_size), mode='nearest')
            x = torch.cat([x, noise_emb], dim=1)

            x = self.down1(x, skips)
            x = self.down2(x, skips) 
            x = self.down3(x, skips)    

            x = self.mid1(x)     
            x = self.mid2(x)   

            x = self.up1(x, skips)
            x = self.up2(x, skips)
            x = self.up3(x, skips)

            return self.final(x)

    return (UNet,)


@app.cell
def _(IMAGE_SIZE, nn, show_image, torch):
    import copy

    class DiffusionModel(nn.Module):
        def __init__(self, model, schedule_fn):
            super().__init__()
            self.network = model
            self.ema_network = copy.deepcopy(model)
            self.ema_network.eval()
            self.ema_decay = 0.8
            self.schedule_fn = schedule_fn
            self.normalizer_mean = 0.0
            self.normalizer_std = 1.0

        def to(self, device):
            # Override to() to ensure both networks move to the same device
            super().to(device)
            self.ema_network.to(device)
            return self

        def set_normalizer(self, mean, std):
            self.normalizer_mean = mean
            self.normalizer_std = std

        def denormalize(self, x):
            return torch.clamp(x * self.normalizer_std + self.normalizer_mean, 0.0, 1.0)

        def denoise(self, noisy_images, noise_rates, signal_rates, training):
            # Use EMA network for inference, main network for training
            if training:
                network = self.network
                network.train()
            else:
                network = self.ema_network
                network.eval()

            pred_noises = network(noisy_images, noise_rates ** 2)
            pred_images = (noisy_images - noise_rates * pred_noises) / signal_rates
            return pred_noises, pred_images

        def reverse_diffusion(self, initial_noise, diffusion_steps):
            step_size = 1.0 / diffusion_steps
            current_images = initial_noise
            for step in range(diffusion_steps):
                t = torch.ones((initial_noise.shape[0], 1, 1, 1), device=initial_noise.device) * (1 - step * step_size)
                noise_rates, signal_rates = self.schedule_fn(t)
                pred_noises, pred_images = self.denoise(current_images, noise_rates, signal_rates, training=False)

                next_diffusion_times = t - step_size
                next_noise_rates, next_signal_rates = self.schedule_fn(next_diffusion_times)
                current_images = next_signal_rates * pred_images + next_noise_rates * pred_noises
            return pred_images

        def generate(self, num_images, diffusion_steps, image_size=64, initial_noise=None):
            if initial_noise is None:
                initial_noise = torch.randn((num_images, self.network.num_channels, image_size, image_size), device=next(self.parameters()).device)
            with torch.no_grad():
                return self.denormalize(self.reverse_diffusion(initial_noise, diffusion_steps))

        def train_step(self, images, optimizer, loss_fn):
            images = (images - self.normalizer_mean) / self.normalizer_std
            noises = torch.randn_like(images)

            diffusion_times = torch.rand((images.size(0), 1, 1, 1), device=images.device)
            noise_rates, signal_rates = self.schedule_fn(diffusion_times)
            noisy_images = signal_rates * images + noise_rates * noises

            pred_noises, _ = self.denoise(noisy_images, noise_rates, signal_rates, training=True)
            loss = loss_fn(pred_noises, noises)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            return loss.item()

        def test_step(self, images, loss_fn):
            images = (images - self.normalizer_mean) / self.normalizer_std
            noises = torch.randn_like(images)

            diffusion_times = torch.rand((images.size(0), 1, 1, 1), device=images.device)
            noise_rates, signal_rates = self.schedule_fn(diffusion_times)
            noisy_images = signal_rates * images + noise_rates * noises

            with torch.no_grad():
                pred_noises, _ = self.denoise(noisy_images, noise_rates, signal_rates, training=False)
                loss = loss_fn(pred_noises, noises)

            return loss.item()

        def plot_images(self, epoch=None, logs=None, num_rows=3, num_cols=6):
            # plot random generated images for visual evaluation of generation quality
            generated_images = self.generate(
                num_images=num_rows * num_cols,
                image_size=IMAGE_SIZE,
                diffusion_steps=20,
            ).cpu()
            show_image(generated_images[0])      

    return (DiffusionModel,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The training loop looks similar to the one used in the previous modules.
    """)
    return


@app.cell
def _(torch):
    from tqdm import tqdm
    import os

    def train_diffusion(model, train_loader, val_loader, optimizer, loss_fn, epochs=50, device='cuda', checkpoint_dir='checkpoints'):
        # Create checkpoint directory
        os.makedirs(checkpoint_dir, exist_ok=True)

        model.to(device)
        best_val_loss = float('inf')

        for epoch in range(epochs):
            model.train()
            train_losses = []
            loader_with_progress = tqdm(train_loader, ncols=120, desc=f'Epoch {epoch+1}/{epochs} [Train]')
            for images, _  in loader_with_progress:
                images = images.to(device)
                loss = model.train_step(images, optimizer, loss_fn)
                train_losses.append(loss)
                loader_with_progress.set_postfix(loss=f'{loss:.4f}')

            avg_train_loss = sum(train_losses) / len(train_losses)

            model.eval()
            model.plot_images()

            val_losses = []
            for images, _ in tqdm(val_loader, desc=f"Epoch {epoch+1} [Val]"):
                images = images.to(device)
                loss = model.test_step(images, loss_fn)
                val_losses.append(loss)

            avg_val_loss = sum(val_losses) / len(val_losses)
            loader_with_progress.set_postfix(loss=f'{avg_train_loss:.4f}')

            # Save checkpoint every epoch
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.network.state_dict(),
                'ema_model_state_dict': model.ema_network.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'normalizer_mean': model.normalizer_mean,
                'normalizer_std': model.normalizer_std
            }

            # Save latest checkpoint
            checkpoint_path = os.path.join(checkpoint_dir, f'diffusion_epoch_{epoch+1:03d}.pth')
            torch.save(checkpoint, checkpoint_path)

            # Save best model
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_checkpoint_path = os.path.join(checkpoint_dir, 'diffusion_best.pth')
                torch.save(checkpoint, best_checkpoint_path)
                print(f"New best model saved at epoch {epoch+1} with val_loss: {avg_val_loss:.4f}")

            print(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
            print(f"Checkpoint saved: {checkpoint_path}")

    def load_checkpoint(model, optimizer, checkpoint_path, device='cuda'):
        """
        Load a saved checkpoint and restore model, EMA, and optimizer states
        """
        checkpoint = torch.load(checkpoint_path, map_location=device)

        model.network.load_state_dict(checkpoint['model_state_dict'])
        model.ema_network.load_state_dict(checkpoint['ema_model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        # Restore normalizer settings
        model.normalizer_mean = checkpoint['normalizer_mean']
        model.normalizer_std = checkpoint['normalizer_std']

        print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
        print(f"Train Loss: {checkpoint['train_loss']:.4f}, Val Loss: {checkpoint['val_loss']:.4f}")

        return checkpoint['epoch']

    return load_checkpoint, train_diffusion


@app.cell
def _(
    BATCH_SIZE,
    DataLoader,
    DiffusionModel,
    IMAGE_SIZE,
    UNet,
    device,
    nn,
    offset_cosine_diffusion_schedule,
    test_loader,
    torch,
    train_dataset,
    train_diffusion,
    train_loader,
):
    NOISE_EMBEDDING_SIZE = 64
    NUM_CHANNELS = 3
    unet = UNet(IMAGE_SIZE, NUM_CHANNELS, NOISE_EMBEDDING_SIZE)
    diffusion_model = DiffusionModel(unet, offset_cosine_diffusion_schedule)

    optimizer = torch.optim.AdamW(diffusion_model.network.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = nn.L1Loss()

    # Calculate proper per-channel normalization statistics
    mean = torch.zeros(NUM_CHANNELS)
    std = torch.zeros(NUM_CHANNELS)
    total_samples = 0

    train_loader_for_stats = DataLoader(train_dataset, batch_size=BATCH_SIZE)
    for imgs, _ in train_loader_for_stats:
        batch_size = imgs.size(0)
        # Reshape to (batch_size, channels, height*width)
        imgs_flat = imgs.view(batch_size, NUM_CHANNELS, -1)

        # Calculate mean and std per channel across all pixels in this batch
        batch_mean = imgs_flat.mean(dim=(0, 2))  # Mean across batch and spatial dims
        batch_std = imgs_flat.std(dim=(0, 2))    # Std across batch and spatial dims

        # Accumulate statistics
        mean += batch_mean * batch_size
        std += batch_std * batch_size
        total_samples += batch_size

    # Finalize statistics
    mean /= total_samples
    std /= total_samples

    print("Normalization stats - Mean:", mean, "Std:", std)
    mean = mean.reshape(1, NUM_CHANNELS, 1, 1).to(device)
    std = std.reshape(1, NUM_CHANNELS, 1, 1).to(device)
    diffusion_model.set_normalizer(mean, std)
    # To load from a checkpoint, uncomment and specify the path:
    # load_checkpoint(diffusion_model, optimizer, 'checkpoints/diffusion_epoch_010.pth', device=device)

    train_diffusion(diffusion_model, train_loader, test_loader, optimizer, loss_fn, epochs=50, device=device)
    return diffusion_model, optimizer


@app.cell
def _(device, diffusion_model, load_checkpoint, optimizer):
    # Example: Load a specific checkpoint and generate images
    # Uncomment the lines below to load a checkpoint from an earlier epoch

    load_checkpoint(diffusion_model, optimizer, 'checkpoints/diffusion_epoch_048.pth', device=device)
    return


@app.cell
def _(IMAGE_SIZE, diffusion_model, show_image):
    # Generate images
    diffusion_model.eval()
    samples = diffusion_model.generate(num_images=1, image_size=IMAGE_SIZE, diffusion_steps=1000)  # returns tensor in [0, 1]

    # Convert to numpy for plotting
    image = samples[0].cpu()

    # Plot
    show_image(image)
    return


@app.cell
def _(device, torch):
    # Test EMA after training
    test_input = torch.randn(1, 3, 64, 64).to(device)
    test_noise_var = torch.tensor([[[[0.5]]]]).to(device)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
