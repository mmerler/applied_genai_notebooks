import marimo

__generated_with = "0.23.9"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # Module 7: Practical 1 - Energy Based Methods
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Theoretical Part

    First, some math... The derivations below are somewhat complex, so do not worry if you don't follow everything. Luckily, the final loss function is simple and intuitive. The derivation is not necessary to understand the code, but gives a very nice interpretation of the final loss function.

    This follows a nice derivation of Contrastive Divergence found in https://www.robots.ox.ac.uk/~ojw/files/NotesOnCD.pdf. If you reference the above, keep in mind a change of notation: our $E$ is the same as $-log(f)$ in the derivation (equivalently $f = e^{-E}$).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We will derive the loss function for our energy based models using maximum likelihood. Later we will see that the resulting loss function makes sense even if we don't transform the energy into a probability distribution.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    Start with an energy defined for all inputs $x$ and dependent on parameters $w$: $E(x, w)$. We use regular letters $x$ and $w$, but keep in mind that they represent vectors (many parameters and many input variables for each input). In addition, for many derivations and analytical results it is convenient to think of $x$ as being a continuous scalar variable instead of a high-dimensional vector. It should be clear from the context, i.e. if we integrate over $x$ it is a continuous variable, if we sum over values $x_i$ of $x$, it is a discrete vector.

    Note: in the practical part below, $x$ will be a tensor of image pixels and $w$ the weights in the neural network, in other words the neural network calculates the energy and we will see that we want to train it (tune the parameters $w$), so that it gives low energy to realistic images and high energy to fake images.

    We can transform $E$ into a probability by normalizing it as follows:

    $p(x,w) = \dfrac{1}{Z(w)} e^{-E(x,w)}$,

    where

    $Z(w) = \int e^{-E(x,w)}$.

    Here we are assuming that $x$ is a continuous variable, so the integration is analogous to summation in the case of $x$ being discrete valued. If you are not familiar with integration, just think about it being aproximated by "summing" the energy values over the possible values of $x$ (all possible images in our example). This should actually look familiar.  Remember the softmax activation?  In the discrete case, this is how we usually turn multiple outputs into a probability distribution used for classification in a neural network.

    The difference is that we are only using this form now to derive an appropriate loss function - with  energy models, our loss function will work directly with unnormalized energy values.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Now that we have a probability function, we can use the usual Maximum Likelihood Estimation.

    $loss = -\mathbb{E}_{x \sim data} \log p(x,w) = -\dfrac{1}{N} \sum \log p(x_i, w) = -\dfrac{1}{N} \sum \log \left( \dfrac{1}{Z(w)} e^{-E(x_i,w)} \right) = \dfrac{1}{N} \sum \left( \log Z(w) + E(x_i, w) \right)$.

    Since $Z$ doesn't depend on $i$ or $x$ this simplifies to

    $loss = \log Z(w) + \dfrac{1}{N} \sum E(x_i, w)$.

    To train a neural network using gradient descent requires calculating the gradient of the loss with respect to the parameters $w$:

    $\nabla_w loss = \nabla_w \log Z(w) + \dfrac{1}{N} \sum \nabla_w E(x_i, w) =  \nabla_w \log Z(w)  + \mathbb{E}_{x \sim data} \nabla_w E(x, w)$.

    We will obtain a form for the first term in terms of an expectation as well.

    $\nabla_w \log Z(w) = \dfrac{1}{Z(w)} \nabla_w Z(w) = \dfrac{1}{Z(w)} \nabla_w \int e^{-E(x,w)} dx =  -\dfrac{1}{Z(w)} \int e^{- E(x,w)}  \nabla_w E(x,w) dx = -\int p(x,w) \nabla_w E(x,w) dx = - \mathbb{E}_{x \sim model} \nabla_w E(x,w)$

    and the full gradient becomes

    $\nabla_w loss = \mathbb{E}_{x \sim data} \nabla_w E(x, w) - \mathbb{E}_{x \sim model} \nabla_w E(x,w)$.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    In other words, minimizing this loss is equivalent to finding parameters $w$ that decrease the energy on the data and increase the energy on the model.  This is a form of contrastive divergence, where we are trying to minimize the difference between the data and the model distributions.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Practical Part

    Now let's use the loss function we derived to implement an Energy Based model to generate synthetic images similar to the hand-written digits in the MNIST dataset. First the usual data preparation steps...
    """)
    return


@app.cell
def _():
    import random
    import numpy as np
    import matplotlib.pyplot as plt
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torchvision
    import torchvision.transforms as transforms
    from torch.utils.data import DataLoader
    import torch.nn.functional as F

    return DataLoader, nn, np, plt, random, torch, torchvision, transforms


@app.cell
def _(np, random, torch):
    # Check which GPU is available
    device = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    torch.manual_seed(42) 
    random.seed(42)
    np.random.seed(42)
    print(f'Using device: {device}')
    return (device,)


@app.cell
def _(numpy, random, torch):
    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        numpy.random.seed(worker_seed)
        random.seed(worker_seed)

    g = torch.Generator()
    g.manual_seed(0)

    # DataLoader(
    #     train_dataset,
    #     batch_size=batch_size,
    #     num_workers=num_workers,
    #     worker_init_fn=seed_worker,
    #     generator=g,
    # )
    return g, seed_worker


@app.cell
def _(torchvision, transforms):
    # Prepare the data
    transform = transforms.Compose([transforms.Pad(2, -1), transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

    # Load/download the datasets
    train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    return test_dataset, train_dataset


@app.cell
def _(DataLoader, g, seed_worker, test_dataset, train_dataset):
    BATCH_SIZE = 128
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, worker_init_fn=seed_worker, generator=g)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, worker_init_fn=seed_worker, generator=g)
    return test_loader, train_loader


@app.cell
def _(np, plt, train_dataset):
    _random_index = np.random.randint(len(train_dataset))
    _img, _label = train_dataset[_random_index]
    plt.imshow(_img.permute(1,2,0), cmap='gray')
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We now construct the neural network that approximates the energy $E(x,w)$. The architecture is a familiar convolutional neural network, with a small addition of using the `swish` activation function instead of ReLU that we've been using before. The convolutional layers reduce the image from 32 by 32 to 2 by 2 and increase the number of channels from 1 to 64. The final fully connected layers reduce the output to a single scalar value representing the energy of the input image.
    """)
    return


@app.cell
def _(nn, torch):
    # Swish activation function
    def swish(x):
        return x * torch.sigmoid(x)

    class EnergyModel(nn.Module):
        def __init__(self):
            super(EnergyModel, self).__init__()
            self.conv1 = nn.Conv2d(1, 16, kernel_size=5, stride=2, padding=2)
            self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
            self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
            self.conv4 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1)

            self.flatten = nn.Flatten()
            self.fc1 = nn.Linear(64 * 2 * 2, 64)
            self.fc2 = nn.Linear(64, 1)

        def forward(self, x):
            x = swish(self.conv1(x))
            x = swish(self.conv2(x))
            x = swish(self.conv3(x))
            x = swish(self.conv4(x))
            x = self.flatten(x)
            x = swish(self.fc1(x))
            return self.fc2(x)

    return (EnergyModel,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We next create a function that samples the low energy states. We assume that we have a neural network represented by `nn_model` that maps input images to energy (often you will see the neural network designed to model negative energy $-E$ instead of $E$ as a convention, in which case you should be careful with either switching the sign before doing gradient descent or doing gradient ascent instead by adding a multiple of the gradient instead of subtracting it.)
    """)
    return


@app.cell
def _(torch):
    def generate_samples(nn_energy_model, inp_imgs, steps, step_size, noise_std):
        nn_energy_model.eval()

        for _ in range(steps):
            # We add noise to the input images, but we will
            # need to calculate the gradients with the transformed
            # noisy images, so tell pytorch not to track the gradient 
            # yet, this way we can avoid unnecessary computations that
            # pytorch does in order to calculate the gradients later:
            with torch.no_grad():
                noise = torch.randn_like(inp_imgs) * noise_std
                inp_imgs = (inp_imgs + noise).clamp(-1.0, 1.0)

            inp_imgs.requires_grad_(True)

            # Compute energy and gradients
            energy = nn_energy_model(inp_imgs)

            # The gradient with respect to parameters is usually done automatically 
            # when we train a neural network as part of .backward() call.
            # Here we do it manually and specify that the gradient should be with 
            # respect to the input images, not the parameters.
            # In addition because energy contains energy values for each input image
            # in a batch, we need to specify an extra grad_outputs argument for the 
            # right gradients to be calculated for each input image.
            grads, = torch.autograd.grad(energy, inp_imgs, grad_outputs=torch.ones_like(energy))

            # Finally, apply gradient clipping for stabilizing the sampling
            with torch.no_grad():
                grads = grads.clamp(-0.03, 0.03)
                inp_imgs = (inp_imgs - step_size*grads).clamp(-1.0, 1.0)

        return inp_imgs.detach()

    # We can also use .backward() call to calculate the gradient instead of autograd.grad
    # The difference is that pytorch will automatically populate the gradients into
    # .grad attribute of the input tensor instead of returning it.
    # Compare the code below to the one above.

    # def generate_samples(nn_energy_model, inp_imgs, steps, step_size, noise_std):
    #     nn_energy_model.eval()

    #     # Energy: (x, w) => E(x, w)
    #     for w in nn_energy_model.parameters():
    #         w.requires_grad = False
    #     inp_imgs = inp_imgs.detach().requires_grad_(True)

    #     for _ in range(steps):
    #         with torch.no_grad():
    #             noise = torch.randn_like(inp_imgs) * noise_std
    #             inp_imgs = (inp_imgs + noise).clamp(-1.0, 1.0)

    #         inp_imgs.requires_grad_(True)

    #         # Compute energy and gradients
    #         energy = nn_energy_model(inp_imgs)
    #         energy.backward(torch.ones_like(energy))

    #         with torch.no_grad():     
    #             grads = inp_imgs.grad.clamp(-0.03, 0.03)
    #             inp_imgs = (inp_imgs - step_size*grads).clamp(-1.0, 1.0)

    #     for w in nn_energy_model.parameters():
    #         w.requires_grad = True

    #     return inp_imgs.detach()
    return (generate_samples,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The `generate_samples` function above uses a process called *Langevin Dynamics* to generate points based on the energy defined by our neural network model. It turns out that this works only after an initial amount of gradient steps referred to as mixing time. This makes generating samples from scratch every time inefficient. One way to address this is to keep a buffer of samples and sample from it instead of starting from random noise every time. This is implemented in the `Buffer` class below.
    """)
    return


@app.cell
def _(generate_samples, np, random, torch):
    class Buffer:
        def __init__(self, model, device):
            super().__init__()
            self.model = model
            self.device = device
            # start with random images in the buffer
            self.examples = [torch.rand((1, 1, 32, 32), device=self.device) * 2 - 1 for _ in range(128)]

        def sample_new_exmps(self, steps, step_size, noise):
            n_new = np.random.binomial(128, 0.05)

            # Generate new random images for around 5% of the inputs
            new_rand_imgs = torch.rand((n_new, 1, 32, 32),  device=self.device) * 2 - 1

            # Sample old images from the buffer for the rest
            old_imgs = torch.cat(random.choices(self.examples, k=128 - n_new), dim=0)

            inp_imgs = torch.cat([new_rand_imgs, old_imgs], dim=0)

            # Run Langevin dynamics
            new_imgs = generate_samples(self.model, inp_imgs, steps, step_size, noise)

            # Update buffer
            self.examples = list(torch.split(new_imgs, 1, dim=0)) + self.examples
            self.examples = self.examples[:8192]

            return new_imgs

    return (Buffer,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Finally define, the Energy class which will be responsible for tuning the weights of our energy model in order to increase the energy on the generated images and decrease it on the training data images.
    """)
    return


@app.cell
def _(Buffer, device, nn, torch):
    from collections import defaultdict

    class Metric:
        def __init__(self):
            self.reset()

        def update(self, val):
            self.total += val.item()
            self.count += 1

        def result(self):
            return self.total / self.count if self.count > 0 else 0.0

        def reset(self):
            self.total = 0.0
            self.count = 0

    class EBM(nn.Module):
        def __init__(self, model, alpha, steps, step_size, noise, device):
            super().__init__()
            self.device = device
            # define the nn energy model 
            self.model = model

            self.buffer = Buffer(self.model, device=device)

            # define the hyperparameters
            self.alpha = alpha
            self.steps = steps
            self.step_size = step_size
            self.noise = noise

            self.loss_metric = Metric()
            self.reg_loss_metric = Metric()
            self.cdiv_loss_metric = Metric()
            self.real_out_metric = Metric()
            self.fake_out_metric = Metric()

        def metrics(self):
            return {
                "loss": self.loss_metric.result(),
                "reg": self.reg_loss_metric.result(),
                "cdiv": self.cdiv_loss_metric.result(),
                "real": self.real_out_metric.result(),
                "fake": self.fake_out_metric.result()
            }

        def reset_metrics(self):
            for m in [self.loss_metric, self.reg_loss_metric, self.cdiv_loss_metric,
                      self.real_out_metric, self.fake_out_metric]:
                m.reset()

        def train_step(self, real_imgs, optimizer):
            real_imgs = real_imgs + torch.randn_like(real_imgs) * self.noise
            real_imgs = torch.clamp(real_imgs, -1.0, 1.0)


            fake_imgs = self.buffer.sample_new_exmps(
                steps=self.steps, step_size=self.step_size, noise=self.noise)

            inp_imgs = torch.cat([real_imgs, fake_imgs], dim=0)
            inp_imgs = inp_imgs.clone().detach().to(device).requires_grad_(False)

            out_scores = self.model(inp_imgs)

            real_out, fake_out = torch.split(out_scores, [real_imgs.size(0), fake_imgs.size(0)], dim=0)

            cdiv_loss = real_out.mean() - fake_out.mean() 
            reg_loss = self.alpha * (real_out.pow(2).mean() + fake_out.pow(2).mean())
            loss = cdiv_loss + reg_loss 

            optimizer.zero_grad()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.1)

            optimizer.step()

            self.loss_metric.update(loss)
            self.reg_loss_metric.update(reg_loss)
            self.cdiv_loss_metric.update(cdiv_loss)
            self.real_out_metric.update(real_out.mean())
            self.fake_out_metric.update(fake_out.mean())

            return self.metrics()

        def test_step(self, real_imgs):
            batch_size = real_imgs.shape[0]
            fake_imgs = torch.rand((batch_size, 1, 32, 32), device=self.device) * 2 - 1
            inp_imgs = torch.cat([real_imgs, fake_imgs], dim=0)

            with torch.no_grad():
                out_scores = self.model(inp_imgs)
                real_out, fake_out = torch.split(out_scores, batch_size, dim=0)
                cdiv = real_out.mean() - fake_out.mean()

            self.cdiv_loss_metric.update(cdiv)
            self.real_out_metric.update(real_out.mean())
            self.fake_out_metric.update(fake_out.mean())

            return {
                "cdiv": self.cdiv_loss_metric.result(),
                "real": self.real_out_metric.result(),
                "fake": self.fake_out_metric.result()
            }

    return (EBM,)


@app.cell
def _(plt, torch):
    @torch.no_grad()
    def clip_img(x):
        return torch.clamp((x + 1) / 2, 0, 1)  # scale from [-1,1] to [0,1]

    def plot_samples(samples, n=8):
        samples = clip_img(samples)
        samples = samples.cpu()
        fig, axes = plt.subplots(1, n, figsize=(n * 2, 2))
        for i in range(n):
            img = samples[i].permute(1, 2, 0).squeeze()  # CHW to HWC
            axes[i].imshow(img, cmap='gray' if img.ndim == 2 else None)
            axes[i].axis("off")
        plt.show()

    return (plot_samples,)


@app.cell
def _(
    EBM,
    EnergyModel,
    device,
    plot_samples,
    test_loader,
    torch,
    train_loader,
):
    # Initialize model, optimizer
    nn_energy_model = EnergyModel()
    nn_energy_model.to(device)
    ebm = EBM(nn_energy_model, alpha=0.1, steps=60, step_size=10, noise=0.005, device=device)
    optimizer = torch.optim.Adam(nn_energy_model.parameters(), lr=0.0001, betas=(0.0, 0.999))

    # Training loop
    for epoch in range(10):
        ebm.reset_metrics()
        for index, batch in enumerate(train_loader):
            real_imgs = batch[0].to(device)
            metrics = ebm.train_step(real_imgs, optimizer)

        plot_samples(torch.cat(ebm.buffer.examples[-8:]), n=8)
        print(f"Epoch {epoch+1} - " + ", ".join(f"{k}: {v:.4f}" for k, v in metrics.items()))

        # Validation step
        ebm.reset_metrics()
        for batch in test_loader:
            real_imgs = batch[0].to(device)
            val_metrics = ebm.test_step(real_imgs)

        print(f"Validation - " + ", ".join(f"{k}: {v:.4f}" for k, v in val_metrics.items()))
    return (nn_energy_model,)


@app.cell
def _(device, generate_samples, nn_energy_model, plot_samples, torch):
    # Generate and plot 8 grayscale 32x32 images
    x = torch.rand((8, 1, 32, 32), device=device) * 2 - 1  # Uniform in [-1, 1]
    new_imgs = generate_samples(nn_energy_model, x, steps=256, step_size=10.0, noise_std=0.01)

    plot_samples(new_imgs, n=8)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
