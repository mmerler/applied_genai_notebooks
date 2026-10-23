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
    # Module 7: Practical - Normalizing Flows
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Recall the two generative models we have seen so far that map a simple distribution to the data:

    - **VAEs** learn an encoder and a decoder, but the encoder only gives an *approximate* posterior, so we can only optimize a lower bound (the ELBO) of the likelihood.
    - **GANs** learn a generator that maps noise to images, but there is no encoder at all and no way to evaluate the likelihood $p(x)$ of a given image.

    **Normalizing Flows** take a different approach: we learn a single function $f$ that is **invertible**. The same network acts as the "encoder" ($z = f(x)$) and, run backwards, as the "decoder" ($x = f^{-1}(z)$). Because $f$ is invertible we can compute the **exact** likelihood of the data and train by plain maximum likelihood estimation, just like we did with simple probability distributions at the start of the course.

    This practical has two parts:

    1. **RealNVP**, a classic normalizing flow built from invertible coupling layers, trained on a 2D toy dataset where we can visualize everything.
    2. **Flow matching**, the continuous-time version of normalizing flows used by modern image generators. We first study it on the same 2D data (trajectories, few-step sampling, exact likelihood), then use it to generate images of flowers of a chosen species.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Theoretical Part

    ### Change of variables

    The key tool is the *change of variables* formula. Let's start in one dimension. Suppose $z$ is a random variable with a known density $p_Z(z)$ (for example a standard normal), and $x$ is obtained from $z$ through an invertible, differentiable function, $x = g(z)$, equivalently $z = f(x)$ with $f = g^{-1}$.

    Probability mass must be conserved: the probability of $x$ falling in a tiny interval $dx$ must equal the probability of $z$ falling in the corresponding interval $dz$:

    $p_X(x)\,|dx| = p_Z(z)\,|dz|$

    and therefore

    $p_X(x) = p_Z(f(x)) \left| \dfrac{df(x)}{dx} \right|$.

    The derivative term accounts for how much $f$ *stretches* or *squeezes* space. If $f$ squeezes a region (derivative smaller than 1 in absolute value), the density in $x$ must be lower, and vice versa.

    In $d$ dimensions the derivative becomes the $d \times d$ **Jacobian matrix** $J_f(x) = \dfrac{\partial f(x)}{\partial x}$, and the absolute value of the derivative becomes the absolute value of its **determinant**:

    $p_X(x) = p_Z(f(x)) \left| \det J_f(x) \right|$.

    Taking the logarithm, which is what we will actually optimize:

    $\log p_X(x) = \log p_Z(f(x)) + \log \left| \det J_f(x) \right|$.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Maximum likelihood

    Given training data $x_1, \dots, x_N$ and a flow $f$ with parameters $w$, the loss is the usual negative log-likelihood:

    $loss = -\dfrac{1}{N} \sum_i \left[ \log p_Z(f(x_i, w)) + \log \left| \det J_f(x_i, w) \right| \right]$.

    If $p_Z$ is the standard normal in $d$ dimensions, the first term is simply

    $\log p_Z(z) = -\dfrac{1}{2} \|z\|^2 - \dfrac{d}{2} \log(2\pi)$.

    So the loss has a nice interpretation: the first term pushes the data to land close to the origin in latent space, while the second term prevents the trivial solution of squashing everything to zero (squeezing space makes the determinant small and its log very negative).

    ### The catch: we need a special kind of network

    For this to work, $f$ has to satisfy three requirements:

    1. It must be **invertible**, so that we can sample by computing $x = f^{-1}(z)$.
    2. The **determinant of the Jacobian** must be cheap to compute. For a general $d \times d$ matrix this costs $O(d^3)$, which is prohibitive for images.
    3. It must still be **expressive** enough to turn a Gaussian into something complicated.

    A standard fully connected network fails the first two. The trick is to build $f$ as a composition of simple invertible layers $f = f_K \circ \dots \circ f_2 \circ f_1$ (hence "flow"). The log-determinant of a composition is just the sum of the log-determinants of the individual layers:

    $\log \left| \det J_f(x) \right| = \sum_{k=1}^{K} \log \left| \det J_{f_k}(h_{k-1}) \right|$, where $h_0 = x$ and $h_k = f_k(h_{k-1})$.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### RealNVP and affine coupling layers

    The model we will implement is **RealNVP** (Dinh et al., 2017, *"Density estimation using Real NVP"*). Its building block is the **affine coupling layer**. The input $x$ is split into two parts, $x_{1}$ and $x_{2}$, using a binary mask $b$. The first part passes through unchanged; the second part is scaled and shifted by amounts that depend **only on the first part**:

    $z_{1} = x_{1}$

    $z_{2} = x_{2} \odot \exp\big(s(x_{1})\big) + t(x_{1})$

    Here $s$ (scale) and $t$ (translation) are arbitrary neural networks and $\odot$ is element-wise multiplication. Now look at what this buys us:

    - **Invertibility for free.** Since $z_1 = x_1$, we know $x_1$ when going backwards, so we can recompute $s$ and $t$ and undo the transformation: $x_{2} = \big(z_{2} - t(z_{1})\big) \odot \exp\big(-s(z_{1})\big)$. We **never need to invert** $s$ or $t$, so they can be as complicated as we like.
    - **Cheap determinant.** The Jacobian is lower triangular:

    $J = \begin{bmatrix} I & 0 \\ \dfrac{\partial z_2}{\partial x_1} & \text{diag}\big(\exp(s(x_1))\big) \end{bmatrix}$

    and the determinant of a triangular matrix is the product of its diagonal. Therefore

    $\log \left| \det J \right| = \sum_j s_j(x_1)$,

    which is just a sum over the outputs of the scale network.

    Since a single coupling layer leaves $x_1$ untouched, we stack several layers and **alternate the mask**, so that every dimension gets transformed.

    Using a mask $b$ (1 for the part that is kept, 0 for the part that is transformed) we can write the whole layer compactly, which is exactly how we will implement it:

    $z = b \odot x + (1 - b) \odot \Big( x \odot \exp\big(s(b \odot x)\big) + t(b \odot x) \Big)$
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Practical Part

    ### Warm-up: change of variables in 1D

    Before building a neural network, let's convince ourselves that the change of variables formula works. Take $z \sim \mathcal{N}(0, 1)$ and the invertible function $x = g(z) = e^{z}$, so $z = f(x) = \log x$ and $\dfrac{df}{dx} = \dfrac{1}{x}$. According to the formula:

    $p_X(x) = p_Z(\log x) \cdot \dfrac{1}{x}$.

    Let's compare it with a histogram of transformed samples.
    """)
    return


@app.cell
def _():
    import math
    import random
    import numpy as np
    import matplotlib.pyplot as plt
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    from sklearn import datasets
    from tqdm import tqdm

    return (
        DataLoader,
        TensorDataset,
        datasets,
        math,
        nn,
        np,
        plt,
        random,
        torch,
        tqdm,
    )


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
def _(np, plt):
    def standard_normal_pdf(z):
        return np.exp(-0.5 * z**2) / np.sqrt(2 * np.pi)

    # Sample z from the base distribution and push it through g(z) = exp(z)
    _z_samples = np.random.randn(100000)
    _x_samples = np.exp(_z_samples)

    # Density predicted by the change of variables formula
    _x_grid = np.linspace(0.01, 6, 500)
    _p_x = standard_normal_pdf(np.log(_x_grid)) * (1 / _x_grid)

    _fig, _axes = plt.subplots(1, 2, figsize=(10, 4))

    _axes[0].hist(_z_samples, bins=100, density=True, alpha=0.6)
    _axes[0].plot(np.linspace(-4, 4, 200), standard_normal_pdf(np.linspace(-4, 4, 200)), 'r', linewidth=2)
    _axes[0].set_title(r"Base distribution: $z \sim \mathcal{N}(0,1)$")

    _axes[1].hist(_x_samples[_x_samples < 6], bins=100, density=True, alpha=0.6, label="samples of exp(z)")
    _axes[1].plot(_x_grid, _p_x, 'r', linewidth=2, label="change of variables")
    _axes[1].set_title(r"Transformed: $x = e^{z}$")
    _axes[1].legend()

    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The red curve computed with the formula matches the histogram (the histogram is slightly lower everywhere because we cut it at $x < 6$, but `density=True` renormalizes it over the visible range only).

    Notice how a simple symmetric bell curve became a skewed distribution with a long tail, just by applying an invertible function. A normalizing flow does exactly the same thing, but **learns** the function with a neural network.

    Why did we need the extra factor $\frac{1}{x}$? What would happen to the area under the red curve without it?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### The data: two moons

    To keep things fast and easy to visualize, we will work in 2D with the classic "two moons" dataset from scikit-learn. Each data point is a point $(x_1, x_2)$ in the plane. Our goal is to learn a flow that maps this distribution to a 2D standard normal, and then run it backwards to generate new points that look like moons.

    We standardize the data (zero mean and unit variance per dimension), which makes training easier, since the target latent distribution also has zero mean and unit variance.
    """)
    return


@app.cell
def _(datasets, np, torch):
    N_TRAIN = 10000
    N_TEST = 2000

    _train_np, _ = datasets.make_moons(n_samples=N_TRAIN, noise=0.05, random_state=42)
    _test_np, _ = datasets.make_moons(n_samples=N_TEST, noise=0.05, random_state=7)

    # Standardize using the statistics of the training data only
    data_mean = _train_np.mean(axis=0)
    data_std = _train_np.std(axis=0)

    train_data = torch.tensor((_train_np - data_mean) / data_std, dtype=torch.float32)
    test_data = torch.tensor((_test_np - data_mean) / data_std, dtype=torch.float32)

    print("Train data shape:", train_data.shape)
    print("Test data shape:", test_data.shape)
    print("Mean after normalization:", np.round(train_data.mean(dim=0).numpy(), 3))
    print("Std after normalization: ", np.round(train_data.std(dim=0).numpy(), 3))
    return N_TRAIN, test_data, train_data


@app.cell
def _(plt, train_data):
    plt.figure(figsize=(5, 5))
    plt.scatter(train_data[:, 0], train_data[:, 1], s=1, alpha=0.5)
    plt.title("Two moons (normalized)")
    plt.axis('equal')
    plt.show()
    return


@app.cell
def _(DataLoader, TensorDataset, test_data, train_data):
    BATCH_SIZE = 256
    # TensorDataset wraps a tensor so that the DataLoader can batch and shuffle it.
    # There are no labels in this problem, each item is a 1-tuple (point,)
    train_loader = DataLoader(TensorDataset(train_data), batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(TensorDataset(test_data), batch_size=BATCH_SIZE, shuffle=False)
    return test_loader, train_loader


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### The affine coupling layer

    Now we implement the coupling layer exactly as in the formula above. The scale network $s$ and translation network $t$ are small fully connected networks. Some important details:

    - The `mask` is stored with `register_buffer`: it is part of the module's state (it moves to the GPU with `.to(device)`), but it is **not** a trainable parameter.
    - The scale network ends with a `tanh`, which keeps $s$ in $[-1, 1]$. Without it, $\exp(s)$ can explode early in training and produce `NaN`s.
    - `forward` goes from data to latent ($x \to z$) and also returns the log-determinant. `inverse` goes from latent to data ($z \to x$) and is what we will use for sampling.
    - Multiplying the outputs of $s$ and $t$ by $(1 - b)$ makes sure that the masked (unchanged) part does not contribute to the log-determinant.
    """)
    return


@app.cell
def _(nn, torch):
    class CouplingLayer(nn.Module):
        def __init__(self, input_dim, hidden_dim, mask):
            super(CouplingLayer, self).__init__()
            self.register_buffer('mask', mask)

            # Scale network s(.)
            self.s_net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim), nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                nn.Linear(hidden_dim, input_dim), nn.Tanh(),
            )
            # Translation network t(.)
            self.t_net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim), nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                nn.Linear(hidden_dim, input_dim),
            )

        def forward(self, x):
            # Data -> latent direction:  z = b*x + (1-b)*(x*exp(s) + t)
            x_masked = x * self.mask
            s = self.s_net(x_masked) * (1 - self.mask)
            t = self.t_net(x_masked) * (1 - self.mask)
            z = x_masked + (1 - self.mask) * (x * torch.exp(s) + t)
            log_det = s.sum(dim=1)  # log|det J| = sum of the scales
            return z, log_det

        def inverse(self, z):
            # Latent -> data direction:  x = b*z + (1-b)*((z - t)*exp(-s))
            z_masked = z * self.mask
            s = self.s_net(z_masked) * (1 - self.mask)
            t = self.t_net(z_masked) * (1 - self.mask)
            x = z_masked + (1 - self.mask) * ((z - t) * torch.exp(-s))
            return x

    return (CouplingLayer,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Before going further, let's check two claims we made in the theory part on a single, untrained coupling layer:

    1. `inverse(forward(x))` gives back exactly `x`.
    2. The cheap log-determinant (sum of the scales) equals the log-determinant of the full Jacobian, which we can compute numerically with `torch.autograd.functional.jacobian`.
    """)
    return


@app.cell
def _(CouplingLayer, torch):
    _layer = CouplingLayer(input_dim=2, hidden_dim=16, mask=torch.tensor([1.0, 0.0]))
    _x = torch.randn(5, 2)

    with torch.no_grad():
        _z, _log_det = _layer(_x)
        _x_rec = _layer.inverse(_z)
    print("Max reconstruction error:", (_x - _x_rec).abs().max().item())

    # Full Jacobian of the layer at the first point
    _point = _x[0]
    _J = torch.autograd.functional.jacobian(lambda p: _layer(p.unsqueeze(0))[0].squeeze(0), _point)
    print("Jacobian:\n", _J)
    print("log|det J| (full Jacobian):", torch.log(torch.abs(torch.det(_J))).item())
    print("log|det J| (sum of scales):", _log_det[0].item())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Notice the zero in the top right of the Jacobian: the first output does not depend on the second input, which is exactly the triangular structure that makes the determinant easy. What is the top left entry, and why?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### The RealNVP model

    The full model stacks several coupling layers with alternating masks `[1, 0]` and `[0, 1]`. In `forward` we accumulate the log-determinants of each layer. In `inverse` we must apply the layers in **reverse order**, since $f^{-1} = f_1^{-1} \circ \dots \circ f_K^{-1}$.

    We also add three convenience methods:

    - `log_prob(x)` computes $\log p_X(x)$ with the change of variables formula. Its negative mean is our loss.
    - `sample(n)` draws $z$ from the standard normal and maps it to data space with `inverse`.
    - `forward_steps(x)` returns the intermediate results after each layer, which we will use to visualize how the flow transforms the data.
    """)
    return


@app.cell
def _(CouplingLayer, math, nn, torch):
    class RealNVP(nn.Module):
        def __init__(self, input_dim=2, hidden_dim=256, num_coupling_layers=6):
            super(RealNVP, self).__init__()
            self.input_dim = input_dim
            masks = [torch.tensor([1.0, 0.0]) if i % 2 == 0 else torch.tensor([0.0, 1.0])
                     for i in range(num_coupling_layers)]
            self.layers = nn.ModuleList([CouplingLayer(input_dim, hidden_dim, m) for m in masks])

        def forward(self, x):
            # x -> z, accumulating the log-determinants of every layer
            log_det_total = torch.zeros(x.shape[0], device=x.device)
            for layer in self.layers:
                x, log_det = layer(x)
                log_det_total = log_det_total + log_det
            return x, log_det_total

        def inverse(self, z):
            # z -> x, applying the layers in reverse order
            for layer in reversed(self.layers):
                z = layer.inverse(z)
            return z

        def base_log_prob(self, z):
            # log density of a standard normal in d dimensions
            return -0.5 * (z ** 2).sum(dim=1) - 0.5 * self.input_dim * math.log(2 * math.pi)

        def log_prob(self, x):
            z, log_det = self.forward(x)
            return self.base_log_prob(z) + log_det

        def sample(self, num_samples, temperature=1.0):
            device = next(self.parameters()).device
            z = torch.randn(num_samples, self.input_dim, device=device) * temperature
            return self.inverse(z)

        def forward_steps(self, x):
            steps = [x]
            for layer in self.layers:
                x, _ = layer(x)
                steps.append(x)
            return steps

    return (RealNVP,)


@app.cell
def _(RealNVP, device, torch):
    COUPLING_LAYERS = 6
    HIDDEN_DIM = 256
    LEARNING_RATE = 1e-3

    flow = RealNVP(input_dim=2, hidden_dim=HIDDEN_DIM, num_coupling_layers=COUPLING_LAYERS).to(device)
    optimizer = torch.optim.Adam(flow.parameters(), lr=LEARNING_RATE)

    print(flow.layers[0])
    print("Number of trainable parameters:", sum(p.numel() for p in flow.parameters() if p.requires_grad))
    return flow, optimizer


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Notice that there is no separate loss function object this time (like `BCELoss` for the VAE): the loss is simply `-flow.log_prob(x).mean()`, the negative log-likelihood we derived above.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Training the model
    """)
    return


@app.cell
def _(device, flow, optimizer, test_loader, torch, tqdm, train_loader):
    EPOCHS = 100

    def evaluate(model, loader, device):
        model.eval()
        total, count = 0.0, 0
        with torch.no_grad():
            for (batch,) in loader:
                batch = batch.to(device)
                total += -model.log_prob(batch).sum().item()
                count += batch.shape[0]
        return total / count

    def train_flow(model, optimizer, train_loader, test_loader, device, epochs=10):
        train_losses, test_losses = [], []
        # One progress bar over the epochs (each epoch here is very short)
        epochs_with_progress = tqdm(range(epochs), desc='Training')
        for epoch in epochs_with_progress:
            model.train()
            epoch_loss, count = 0.0, 0
            for (batch,) in train_loader:
                batch = batch.to(device)
                optimizer.zero_grad()
                loss = -model.log_prob(batch).mean()  # negative log-likelihood
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * batch.shape[0]
                count += batch.shape[0]

            train_losses.append(epoch_loss / count)
            test_losses.append(evaluate(model, test_loader, device))
            epochs_with_progress.set_postfix(train_nll=f'{train_losses[-1]:.4f}', test_nll=f'{test_losses[-1]:.4f}')

        print("Finished Training")
        return train_losses, test_losses

    train_losses, test_losses = train_flow(flow, optimizer, train_loader, test_loader, device, epochs=EPOCHS)
    return test_losses, train_losses


@app.cell
def _(plt, test_losses, train_losses):
    plt.figure(figsize=(7, 4))
    plt.plot(train_losses, label='train')
    plt.plot(test_losses, label='test')
    plt.xlabel('Epoch')
    plt.ylabel('Negative log-likelihood')
    plt.legend()
    plt.title('Training curve')
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Unlike the GAN, where the losses of the generator and discriminator are hard to interpret, here the loss has a clear meaning: it is the average negative log-likelihood of the data (in nats). Lower is better, and we can compare the train and test values directly to check for overfitting.

    Can the loss become negative? Why is that not a contradiction, given that probabilities are at most 1?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Results: both directions of the flow

    Now the most important plot. The top row goes **data → latent**: we push the test points through `forward` and check that they look like a standard normal. The bottom row goes **latent → data**: we sample fresh points from a standard normal and push them through `inverse` to generate new data.
    """)
    return


@app.cell
def _(device, flow, plt, test_data, torch):
    flow.eval()
    with torch.no_grad():
        _z_test, _ = flow(test_data.to(device))
        _z_test = _z_test.cpu()
        _z_new = torch.randn(len(test_data), 2)
        _x_new = flow.inverse(_z_new.to(device)).cpu()

    _fig, _axes = plt.subplots(2, 2, figsize=(10, 10))

    _axes[0, 0].scatter(test_data[:, 0], test_data[:, 1], s=2, color='tab:blue')
    _axes[0, 0].set_title("Data space: test data X")
    _axes[0, 1].scatter(_z_test[:, 0], _z_test[:, 1], s=2, color='tab:blue')
    _axes[0, 1].set_title("Latent space: f(X)")

    _axes[1, 0].scatter(_z_new[:, 0], _z_new[:, 1], s=2, color='tab:green')
    _axes[1, 0].set_title(r"Latent space: Z ~ N(0, I)")
    _axes[1, 1].scatter(_x_new[:, 0], _x_new[:, 1], s=2, color='tab:green')
    _axes[1, 1].set_title(r"Data space: generated $f^{-1}(Z)$")

    for _ax in _axes.flat:
        _ax.set_xlim(-4, 4)
        _ax.set_ylim(-4, 4)
        _ax.set_aspect('equal')

    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The generated samples should look like two moons, possibly with a few points stretched along a thin "bridge" between the moons. Why might that be? Hint: $f^{-1}$ is continuous, the Gaussian is a single connected blob, and the moons are two separate pieces. A continuous map cannot tear space apart, so the flow has to squeeze some probability mass into a thin connection.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Following the flow layer by layer

    Since the flow is a composition of simple layers, we can watch the data being transformed step by step. We color each point by its position along the moons so you can track where it goes. Notice how each layer only moves points along **one** axis (the one not masked), because the other coordinate is copied unchanged.
    """)
    return


@app.cell
def _(device, flow, np, plt, test_data, torch):
    with torch.no_grad():
        _steps = [s.cpu() for s in flow.forward_steps(test_data.to(device))]

    # Color points by their angle around the center so we can follow them
    _colors = np.arctan2(test_data[:, 1].numpy(), test_data[:, 0].numpy())

    _fig, _axes = plt.subplots(1, len(_steps), figsize=(3 * len(_steps), 3.8))
    for _i, (_ax, _s) in enumerate(zip(_axes, _steps)):
        _ax.scatter(_s[:, 0], _s[:, 1], s=1, c=_colors, cmap='viridis')
        _ax.set_title("Input" if _i == 0 else f"After layer {_i}\n(transforms {'x2' if _i % 2 == 1 else 'x1'})")
        _ax.set_xlim(-4, 4)
        _ax.set_ylim(-4, 4)
        _ax.set_aspect('equal')
        _ax.set_xticks([])
        _ax.set_yticks([])

    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### The learned density

    A big advantage of normalizing flows over GANs is that we can evaluate the exact density $p_X(x)$ at **any** point. Let's evaluate `log_prob` on a grid covering the plane and plot $p_X(x)$ as a heatmap.
    """)
    return


@app.cell
def _(device, flow, np, plt, torch):
    _grid_x, _grid_y = np.meshgrid(np.linspace(-3, 3, 300), np.linspace(-3, 3, 300))
    _grid = torch.tensor(np.stack([_grid_x.ravel(), _grid_y.ravel()], axis=1), dtype=torch.float32)

    with torch.no_grad():
        _log_p = flow.log_prob(_grid.to(device)).cpu().numpy()
    _density = np.exp(_log_p).reshape(_grid_x.shape)

    # Sanity check: the density should integrate to (roughly) 1 over the plane
    _cell_area = (6 / 299) ** 2
    print(f"Approximate integral of p(x) over the grid: {_density.sum() * _cell_area:.3f}")

    plt.figure(figsize=(6, 5))
    plt.pcolormesh(_grid_x, _grid_y, _density, shading='auto', cmap='magma')
    plt.colorbar(label='p(x)')
    plt.title("Density learned by RealNVP")
    plt.gca().set_aspect('equal')
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The integral being close to 1 is a nice confirmation that the model really defines a valid probability distribution: the Jacobian term in the loss is what guarantees that. Compare this with energy-based models, where computing the normalizing constant $Z(w)$ was intractable.

    ### Sampling temperature

    A common trick when sampling from flows (used for example in Glow to produce cleaner faces) is to sample $z$ from a normal distribution with a reduced standard deviation $T < 1$, called the *temperature*. This keeps samples closer to the high-density center of the latent space. Use the slider to see the effect.
    """)
    return


@app.cell
def _(mo):
    temperature_slider = mo.ui.slider(label="Temperature $T$", value=1.0, start=0.1, stop=2.0, step=0.1)
    temperature_slider
    return (temperature_slider,)


@app.cell
def _(flow, plt, temperature_slider, torch):
    flow.eval()
    with torch.no_grad():
        _samples = flow.sample(3000, temperature=temperature_slider.value).cpu()

    plt.figure(figsize=(5, 5))
    plt.scatter(_samples[:, 0], _samples[:, 1], s=1, alpha=0.5, color='tab:green')
    plt.title(f"Samples with temperature T = {temperature_slider.value:.1f}")
    plt.xlim(-4, 4)
    plt.ylim(-4, 4)
    plt.gca().set_aspect('equal')
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    What happens to the bridge between the moons as you lower the temperature? And what happens when $T > 1$?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Flow Matching: Normalizing Flows in Continuous Time

    RealNVP works, but its design is dictated by mathematical constraints: every layer must be invertible, with a triangular Jacobian, so half of the input is always copied unchanged. This limits how expressive each layer can be. **Flow matching** (Lipman et al., 2023; Liu et al., 2022, *"rectified flow"*; Albergo & Vanden-Eijnden, 2023) removes these constraints, and it is the training method behind recent image generators such as Stable Diffusion 3 and Flux.

    ### From layers to an ODE

    Imagine a flow with a very large number of very small layers, each moving every point a tiny bit: $x_{k+1} = x_k + \Delta t \, v(x_k, t_k)$. In the limit of infinitely many layers this becomes an **ordinary differential equation** (ODE):

    $\dfrac{dx}{dt} = v_\theta(x, t), \qquad t \in [0, 1]$.

    The neural network $v_\theta$ is a **velocity field**: for every point $x$ and time $t$ it says in which direction and how fast that point should move. We start from noise $x(0) \sim \mathcal{N}(0, I)$ and integrate to $t = 1$ to get a data sample. This is still a normalizing flow: integrating the ODE backwards takes us from data back to noise, so the map is invertible **for any (smooth) network** $v_\theta$. No masks, no special architecture.

    The change of variables formula also has a continuous version, the *instantaneous change of variables*:

    $\dfrac{d}{dt} \log p_t(x(t)) = -\nabla \cdot v_\theta(x(t), t) = -\text{tr}\left( \dfrac{\partial v_\theta}{\partial x} \right)$.

    The sum of log-determinants over layers became an integral of the *trace* of the Jacobian (the divergence) over time. Intuitively, where the velocity field spreads points apart (positive divergence), the density decreases.

    ### The flow matching objective

    We could train $v_\theta$ by maximum likelihood, but that requires solving an ODE (and computing traces) at every training step, which is slow. Flow matching uses a much simpler idea. Take a noise sample $x_0 \sim \mathcal{N}(0, I)$ and a data sample $x_1$, and connect them with a straight line:

    $x_t = (1 - t)\, x_0 + t\, x_1$.

    A point moving along this line has the constant velocity $\dfrac{dx_t}{dt} = x_1 - x_0$. So we simply train the network to predict this velocity, with a regression loss:

    $loss = \mathbb{E}_{t \sim \mathcal{U}[0,1],\; x_0 \sim \mathcal{N}(0,I),\; x_1 \sim data} \Big\| v_\theta(x_t, t) - (x_1 - x_0) \Big\|^2$.

    Each straight line goes from one specific noise point to one specific data point, and many lines cross through the same $x_t$. The network cannot know which line it is on, so the best it can do (the minimizer of the MSE) is to predict the **average** velocity of all the lines passing through $x_t$ at time $t$. The key result of the flow matching papers is that this averaged velocity field transports the Gaussian **exactly** onto the data distribution.

    Compared to RealNVP:

    | | RealNVP | Flow matching |
    |---|---|---|
    | Network | invertible coupling layers | any network |
    | Training | maximum likelihood | MSE regression, no ODE solving |
    | Sampling | one pass of `inverse` | many small ODE steps |
    | Exact likelihood | cheap | possible, but expensive (ODE + trace) |
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Flow matching on two moons

    The velocity network is a plain MLP that receives the point $x$ and the time $t$ (concatenated as an extra input) and outputs a 2D velocity. Note that, unlike the coupling layer, nothing about it is invertible.
    """)
    return


@app.cell
def _(nn, torch):
    class VelocityField(nn.Module):
        def __init__(self, dim=2, hidden_dim=256):
            super(VelocityField, self).__init__()
            self.net = nn.Sequential(
                nn.Linear(dim + 1, hidden_dim), nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
                nn.Linear(hidden_dim, dim),
            )

        def forward(self, x, t):
            # x: (B, dim), t: (B, 1)
            return self.net(torch.cat([x, t], dim=1))

    return (VelocityField,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The training function is written so that we can reuse it later: `get_pairs` returns a batch of $(x_0, x_1)$ pairs. For standard flow matching, $x_0$ is fresh Gaussian noise and $x_1$ a random batch of training points, drawn **independently**. Notice how short the training step is: sample $t$, build $x_t$, regress the velocity.
    """)
    return


@app.cell
def _(device, torch, tqdm, train_data):
    FM_BATCH_SIZE = 256
    moons_on_device = train_data.to(device)

    def independent_pairs(batch_size=FM_BATCH_SIZE):
        x1 = moons_on_device[torch.randint(len(moons_on_device), (batch_size,), device=device)]
        x0 = torch.randn_like(x1)
        return x0, x1

    def train_velocity_model(model, optimizer, get_pairs, num_steps):
        model.train()
        losses = []
        steps_with_progress = tqdm(range(num_steps), desc='Training')
        for step in steps_with_progress:
            x0, x1 = get_pairs()
            t = torch.rand(x0.shape[0], 1, device=x0.device)
            x_t = (1 - t) * x0 + t * x1          # point on the straight line
            target_velocity = x1 - x0             # its (constant) velocity
            loss = ((model(x_t, t) - target_velocity) ** 2).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            if step % 100 == 0:
                steps_with_progress.set_postfix(loss=f'{loss.item():.4f}')
        print("Finished Training")
        return losses

    return independent_pairs, train_velocity_model


@app.cell
def _(VelocityField, device, independent_pairs, torch, train_velocity_model):
    FM_STEPS = 5000
    torch.manual_seed(42)
    fm_model = VelocityField().to(device)
    _fm_optimizer = torch.optim.Adam(fm_model.parameters(), lr=1e-3)
    fm_losses = train_velocity_model(fm_model, _fm_optimizer, independent_pairs, num_steps=FM_STEPS)
    return (fm_model,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Why can't the loss go to zero here, unlike in a normal regression problem? (Hint: think about what the network is asked to predict at a point $x_t$ where many lines cross.)

    ### Sampling: solving the ODE

    To generate data we integrate the ODE from $t = 0$ to $t = 1$. The simplest solver is **Euler's method**: split $[0, 1]$ into `num_steps` steps of size $\Delta t$ and repeat

    $x_{t + \Delta t} = x_t + \Delta t \; v_\theta(x_t, t)$.

    This is exactly the "many small residual layers" picture from above. More steps give a more accurate solution but cost more network evaluations.
    """)
    return


@app.cell
def _(torch):
    @torch.no_grad()
    def sample_ode(model, x0, num_steps, return_trajectory=False):
        model.eval()
        dt = 1.0 / num_steps
        x = x0
        trajectory = [x0]
        for i in range(num_steps):
            t = torch.full((x.shape[0], 1), i * dt, device=x.device)
            x = x + dt * model(x, t)
            trajectory.append(x)
        return torch.stack(trajectory) if return_trajectory else x

    return (sample_ode,)


@app.cell
def _(device, fm_model, plt, sample_ode, torch):
    torch.manual_seed(0)
    _x0 = torch.randn(2000, 2, device=device)
    _traj = sample_ode(fm_model, _x0, num_steps=100, return_trajectory=True).cpu()

    _times = [0.0, 0.25, 0.5, 0.75, 1.0]
    _fig, _axes = plt.subplots(1, len(_times), figsize=(3 * len(_times), 3.2))
    for _ax, _t in zip(_axes, _times):
        _pts = _traj[int(_t * 100)]
        _ax.scatter(_pts[:, 0], _pts[:, 1], s=1, alpha=0.5, color='tab:purple')
        _ax.set_title(f"t = {_t}")
        _ax.set_xlim(-3.5, 3.5)
        _ax.set_ylim(-3.5, 3.5)
        _ax.set_aspect('equal')
        _ax.set_xticks([])
        _ax.set_yticks([])
    plt.suptitle("Flow matching: noise (t = 0) flows into data (t = 1)")
    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Compare this with the layer-by-layer plot of RealNVP: instead of a few discrete jumps along one axis at a time, the points now move smoothly in all directions at once.

    Let's look at the individual **trajectories** of a few hundred particles (green: start, blue: end).
    """)
    return


@app.cell
def _():
    def plot_trajectories(ax, trajectory, title, num_paths=300):
        trajectory = trajectory[:, :num_paths].cpu()
        for i in range(trajectory.shape[1]):
            ax.plot(trajectory[:, i, 0], trajectory[:, i, 1], color='gray', alpha=0.3, linewidth=0.7)
        ax.scatter(trajectory[0, :, 0], trajectory[0, :, 1], s=4, color='tab:green', zorder=3, label='t = 0')
        ax.scatter(trajectory[-1, :, 0], trajectory[-1, :, 1], s=4, color='tab:blue', zorder=3, label='t = 1')
        ax.set_xlim(-3.5, 3.5)
        ax.set_ylim(-3.5, 3.5)
        ax.set_aspect('equal')
        ax.set_title(title)

    def straightness(trajectory):
        # Total length traveled by all particles divided by the total straight-line
        # distance between their start and end points (1.0 = perfectly straight paths)
        path_length = (trajectory[1:] - trajectory[:-1]).norm(dim=-1).sum()
        distance = (trajectory[-1] - trajectory[0]).norm(dim=-1).sum()
        return (path_length / distance).item()

    return plot_trajectories, straightness


@app.cell
def _(
    device,
    fm_model,
    plot_trajectories,
    plt,
    sample_ode,
    straightness,
    torch,
):
    torch.manual_seed(1)
    _traj = sample_ode(fm_model, torch.randn(1000, 2, device=device), num_steps=100, return_trajectory=True)
    _fig, _ax = plt.subplots(figsize=(6, 6))
    plot_trajectories(_ax, _traj, f"Flow matching trajectories (path length / distance = {straightness(_traj):.3f})")
    _ax.legend()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We trained on **straight** lines, yet many of the learned paths are **curved**. Why? Two solutions of an ODE can never cross (at any point the velocity is unique), but the straight training lines cross all the time. The network averages the crossing directions, and the resulting paths bend around each other.

    Curved paths are a problem for sampling: Euler's method follows the tangent, so with few steps it "overshoots" the curves. Let's see how the samples depend on the number of steps.
    """)
    return


@app.cell
def _(device, fm_model, plt, sample_ode, torch):
    torch.manual_seed(2)
    _x0 = torch.randn(2000, 2, device=device)
    _step_counts = [1, 2, 4, 8, 100]
    _fig, _axes = plt.subplots(1, len(_step_counts), figsize=(3 * len(_step_counts), 3.2))
    for _ax, _n in zip(_axes, _step_counts):
        _x = sample_ode(fm_model, _x0, num_steps=_n).cpu()
        _ax.scatter(_x[:, 0], _x[:, 1], s=1, alpha=0.5, color='tab:purple')
        _ax.set_title(f"{_n} Euler step{'s' if _n > 1 else ''}")
        _ax.set_xlim(-3.5, 3.5)
        _ax.set_ylim(-3.5, 3.5)
        _ax.set_aspect('equal')
        _ax.set_xticks([])
        _ax.set_yticks([])
    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    With a single step every point moves straight to the *average* target, so all the samples collapse onto a blurry blob in the middle. We need several steps to follow the curves.

    ### Straightening the flow: Reflow

    The *rectified flow* paper proposes a neat trick to fix this. The trained model defines a deterministic map from each noise point $z$ to a data point $\text{ODE}(z)$. If we now train a **new** model on the pairs $(x_0, x_1) = (z, \text{ODE}(z))$ instead of independent random pairs, the training lines no longer cross (they come from non-crossing ODE paths), so there is nothing to average and the learned paths become (almost) straight. A straight path can be followed exactly with a **single** Euler step.

    This procedure is called **reflow**. Below we generate the coupled pairs with the first model (using many steps, so they are accurate), and train a second model initialized from the first one. The training code is exactly the same; only `get_pairs` changes.
    """)
    return


@app.cell
def _(
    N_TRAIN,
    VelocityField,
    device,
    fm_model,
    sample_ode,
    torch,
    train_velocity_model,
):
    REFLOW_STEPS = 3000

    # 1. Generate coupled pairs (z, ODE(z)) with the trained model
    torch.manual_seed(3)
    reflow_x0 = torch.randn(N_TRAIN, 2, device=device)
    reflow_x1 = sample_ode(fm_model, reflow_x0, num_steps=100)

    def coupled_pairs(batch_size=256):
        idx = torch.randint(N_TRAIN, (batch_size,), device=device)
        return reflow_x0[idx], reflow_x1[idx]

    # 2. Train a new model on the coupled pairs, starting from the weights of the first one
    reflow_model = VelocityField().to(device)
    reflow_model.load_state_dict(fm_model.state_dict())
    _reflow_optimizer = torch.optim.Adam(reflow_model.parameters(), lr=1e-3)
    reflow_losses = train_velocity_model(reflow_model, _reflow_optimizer, coupled_pairs, num_steps=REFLOW_STEPS)
    return (reflow_model,)


@app.cell
def _(
    device,
    fm_model,
    plot_trajectories,
    plt,
    reflow_model,
    sample_ode,
    straightness,
    torch,
):
    torch.manual_seed(1)
    _x0 = torch.randn(1000, 2, device=device)
    _fig, _axes = plt.subplots(1, 2, figsize=(12, 6))
    for _ax, _model, _name in zip(_axes, [fm_model, reflow_model], ["Flow matching", "After reflow"]):
        _traj = sample_ode(_model, _x0, num_steps=100, return_trajectory=True)
        plot_trajectories(_ax, _traj, f"{_name} (path length / distance = {straightness(_traj):.3f})")
    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Now use the slider to compare the samples of the two models with very few Euler steps. How many steps does each model need to produce good moons?
    """)
    return


@app.cell
def _(mo):
    ode_steps_slider = mo.ui.slider(label="Number of Euler steps", value=1, start=1, stop=20, step=1)
    ode_steps_slider
    return (ode_steps_slider,)


@app.cell
def _(
    device,
    fm_model,
    ode_steps_slider,
    plt,
    reflow_model,
    sample_ode,
    torch,
):
    torch.manual_seed(2)
    _x0 = torch.randn(2000, 2, device=device)
    _fig, _axes = plt.subplots(1, 2, figsize=(10, 5))
    for _ax, _model, _name in zip(_axes, [fm_model, reflow_model], ["Flow matching", "After reflow"]):
        _x = sample_ode(_model, _x0, num_steps=ode_steps_slider.value).cpu()
        _ax.scatter(_x[:, 0], _x[:, 1], s=1, alpha=0.5, color='tab:purple')
        _ax.set_title(f"{_name}: {ode_steps_slider.value} step(s)")
        _ax.set_xlim(-3.5, 3.5)
        _ax.set_ylim(-3.5, 3.5)
        _ax.set_aspect('equal')
    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Reflow is one of the ideas behind few-step image generators: a model that needs 1 to 4 network evaluations instead of 50 is dramatically cheaper to run.

    ### Exact likelihood, the continuous way

    Flow matching never computed a likelihood during training, but since the model is still a normalizing flow we *can* compute $\log p(x)$ exactly, using the instantaneous change of variables. Starting from a data point $x_1 = x$, we integrate the ODE **backwards** to $t = 0$ while accumulating the divergence:

    $\log p_1(x) = \log p_0(x_0) - \displaystyle\int_0^1 \nabla \cdot v_\theta(x_t, t) \, dt$.

    In 2D the divergence $\frac{\partial v_1}{\partial x_1} + \frac{\partial v_2}{\partial x_2}$ costs just two backward passes. In high dimensions computing the full trace is too expensive, and in practice it is *estimated* with random projections (Hutchinson's trick), which is why we don't train flow matching models this way.
    """)
    return


@app.cell
def _(math, torch):
    def ode_log_prob(model, x, num_steps=100):
        model.eval()
        dt = 1.0 / num_steps
        divergence_integral = torch.zeros(x.shape[0], device=x.device)
        for i in reversed(range(num_steps)):
            t = torch.full((x.shape[0], 1), (i + 1) * dt, device=x.device)
            with torch.enable_grad():
                x = x.detach().requires_grad_(True)
                v = model(x, t)
                # Exact divergence: sum of the diagonal entries of the Jacobian dv/dx
                divergence = sum(
                    torch.autograd.grad(v[:, d].sum(), x, retain_graph=True)[0][:, d] for d in range(x.shape[1])
                )
            x = (x - dt * v).detach()  # Euler step backwards in time
            divergence_integral = divergence_integral + dt * divergence.detach()
        log_p0 = -0.5 * (x ** 2).sum(dim=1) - 0.5 * x.shape[1] * math.log(2 * math.pi)
        return log_p0 - divergence_integral

    return (ode_log_prob,)


@app.cell
def _(device, fm_model, np, ode_log_prob, plt, torch):
    _grid_x, _grid_y = np.meshgrid(np.linspace(-3, 3, 150), np.linspace(-3, 3, 150))
    _grid = torch.tensor(np.stack([_grid_x.ravel(), _grid_y.ravel()], axis=1), dtype=torch.float32, device=device)
    _density = np.exp(ode_log_prob(fm_model, _grid).cpu().numpy()).reshape(_grid_x.shape)

    print(f"Approximate integral of p(x) over the grid: {_density.sum() * (6 / 149) ** 2:.3f}")

    plt.figure(figsize=(6, 5))
    plt.pcolormesh(_grid_x, _grid_y, _density, shading='auto', cmap='magma')
    plt.colorbar(label='p(x)')
    plt.title("Density learned by flow matching")
    plt.gca().set_aspect('equal')
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Compare it with the RealNVP density above. Is the "bridge" between the moons still there? The integral is only approximately 1 because both the ODE solver and the grid are discretized.

    ## Image Generation: Class-Conditional Flow Matching on Flowers

    Since flow matching works with **any** network, we can now use an ordinary convolutional neural network, like the ones from Module 4, as the velocity field $v_\theta(x_t, t)$, and generate real color images. We will train on the Oxford Flowers102 dataset at 32 by 32 pixels.

    We will also make the model **class-conditional**: Flowers102 has labels for 102 flower species, so we give the network the label $c$ as an extra input, $v_\theta(x_t, t, c)$, and ask it for, say, a sunflower or a rose.

    If you are on a fast GPU, try `IMG_SIZE = 64`.
    """)
    return


@app.cell
def _():
    import copy
    import torchvision
    import torchvision.transforms as transforms
    from torch.utils.data import ConcatDataset

    return ConcatDataset, copy, torchvision, transforms


@app.cell
def _(ConcatDataset, DataLoader, torchvision, transforms):
    IMG_SIZE = 32
    IMG_BATCH_SIZE = 64
    NUM_CLASSES = 102

    img_transform = transforms.Compose([
        transforms.CenterCrop(340),            # focus on the flower in the middle
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),     # simple data augmentation
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # pixel values in [-1, 1]
    ])

    # Generative models need a lot of data, so we use all three splits for training (~8,000 images)
    flowers_dataset = ConcatDataset([
        torchvision.datasets.Flowers102(root='./data', split=_split, download=True, transform=img_transform)
        for _split in ['train', 'val', 'test']
    ])
    img_loader = DataLoader(flowers_dataset, batch_size=IMG_BATCH_SIZE, shuffle=True, drop_last=True)

    # Recent versions of torchvision include the species names
    flower_names = list(getattr(torchvision.datasets.Flowers102, 'classes', [f'class {i}' for i in range(NUM_CLASSES)]))

    print("Number of images:", len(flowers_dataset))
    print("Image shape:", flowers_dataset[0][0].shape)
    return IMG_SIZE, NUM_CLASSES, flower_names, img_loader


@app.cell
def _(plt, torchvision):
    def to_display(images):
        # [-1, 1] -> [0, 1], move to the cpu
        return ((images.detach().cpu() + 1) / 2).clamp(0, 1)

    def show_image_grid(images, nrow=8, title=None):
        # make_grid tiles a batch of (N, C, H, W) images into a single image
        _grid = torchvision.utils.make_grid(to_display(images), nrow=nrow, padding=1, pad_value=1.0)
        plt.figure(figsize=(nrow * 1.2, max(1, len(images) // nrow) * 1.2))
        plt.imshow(_grid.permute(1, 2, 0))
        plt.axis('off')
        if title:
            plt.title(title)
        plt.show()

    return show_image_grid, to_display


@app.cell
def _(img_loader, show_image_grid):
    real_images, real_labels = next(iter(img_loader))
    show_image_grid(real_images[:32], title="Real flowers (32 x 32)")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### The velocity network: a simple CNN

    For the moons, the velocity network was an MLP that received $x$ and $t$ concatenated. For images we do exactly the same thing, but with convolutions:

    - The network is a stack of $3 \times 3$ convolutional layers that all keep the full $32 \times 32$ resolution (no pooling), since the output velocity must have the same shape as the input image.
    - The **time** $t$ and the **class** are turned into extra input channels: $t$ becomes one channel filled with the value $t$, and the class goes through an `nn.Embedding` (a learnable vector for each of the 102 species, plus one extra "null" class that we will need for guidance below), copied to every pixel. These conditioning channels are concatenated to the input of **every** layer, so each layer always knows the time and the flower species.

    Two small additions make this simple network work much better:

    - **Dilated convolutions.** A $3 \times 3$ filter only sees a pixel's immediate neighbors, so our stack of 9 layers would see a patch of only $19 \times 19$ pixels, too small to understand the shape of a whole flower. A convolution with dilation $d$ spreads the 9 filter weights $d$ pixels apart: it has the same number of parameters but looks further away. With dilations $1, 2, 4, 8, 4, 2, 1$ in the middle layers, each output pixel sees a $49 \times 49$ region, which covers the whole image.
    - **Residual connections.** Each layer *adds* its output to its input, `h = h + layer(h)`, instead of replacing it. Information (and gradients) can then flow straight through the network, which makes a deep stack of layers much easier to train. Try removing them (exercise 9): the loss barely decreases.

    Finally, the last layer is initialized to zero, so the network starts by predicting a velocity of zero everywhere.
    """)
    return


@app.cell
def _(nn, torch):
    class ConditionalCNN(nn.Module):
        def __init__(self, num_classes, num_channels=3, hidden_channels=64, class_dim=16, dilations=(1, 2, 4, 8, 4, 2, 1)):
            super().__init__()
            self.null_class = num_classes  # extra label meaning "no class given"
            self.class_embedding = nn.Embedding(num_classes + 1, class_dim)
            cond_channels = 1 + class_dim   # one channel for t, class_dim channels for the class

            self.first = nn.Conv2d(num_channels + cond_channels, hidden_channels, kernel_size=3, padding=1)
            # padding = dilation keeps the image size unchanged for a 3x3 dilated convolution
            self.layers = nn.ModuleList([
                nn.Conv2d(hidden_channels + cond_channels, hidden_channels, kernel_size=3, padding=d, dilation=d)
                for d in dilations
            ])
            self.final = nn.Conv2d(hidden_channels + cond_channels, num_channels, kernel_size=3, padding=1)
            nn.init.zeros_(self.final.weight)
            nn.init.zeros_(self.final.bias)

        def forward(self, x, t, labels):
            b, _, h, w = x.shape
            # Conditioning channels: t and the class embedding, copied to every pixel
            t_channel = t.view(b, 1, 1, 1).expand(b, 1, h, w)
            class_channels = self.class_embedding(labels)[:, :, None, None].expand(-1, -1, h, w)
            cond = torch.cat([t_channel, class_channels], dim=1)

            # SiLU (x * sigmoid(x)) is a smooth version of ReLU
            hidden = nn.functional.silu(self.first(torch.cat([x, cond], dim=1)))
            for layer in self.layers:
                hidden = hidden + nn.functional.silu(layer(torch.cat([hidden, cond], dim=1)))  # residual connection
            return self.final(torch.cat([hidden, cond], dim=1))

    return (ConditionalCNN,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Classifier-free guidance

    A purely conditional model often produces samples that only loosely match their class. **Classifier-free guidance** (Ho & Salimans, 2022), used by practically all modern text-to-image models, fixes this with a simple trick:

    1. During training, we replace the true label with the "null" label 10% of the time, so the **same** network learns both the conditional velocity $v_\theta(x, t, c)$ and the unconditional one $v_\theta(x, t, \varnothing)$.
    2. During sampling, we extrapolate away from the unconditional velocity, in the direction of the class:

    $v = v_\theta(x, t, \varnothing) + w \, \big( v_\theta(x, t, c) - v_\theta(x, t, \varnothing) \big)$.

    With $w = 0$ we get unconditional samples, $w = 1$ is the plain conditional model, and $w > 1$ exaggerates the features that distinguish the class, trading diversity for fidelity.

    We also keep an **exponential moving average** (EMA) of the weights, a running average of the model over the last few thousand training steps. Generating with the EMA model gives noticeably cleaner images, and it is standard practice for flow matching (and diffusion) models.
    """)
    return


@app.cell
def _(ConditionalCNN, NUM_CLASSES, copy, device, torch):
    torch.manual_seed(42)
    velocity_cnn = ConditionalCNN(num_classes=NUM_CLASSES).to(device)
    ema_cnn = copy.deepcopy(velocity_cnn)
    for _p in ema_cnn.parameters():
        _p.requires_grad_(False)

    print(velocity_cnn)
    print("Number of trainable parameters:", sum(p.numel() for p in velocity_cnn.parameters() if p.requires_grad))
    return ema_cnn, velocity_cnn


@app.cell
def _(torch):
    @torch.no_grad()
    def generate_images(model, labels, image_size, num_steps=20, guidance=2.0, seed=0, return_trajectory=False):
        model.eval()
        device = labels.device
        # Generate the noise on the cpu with a fixed seed, so the same seed gives the same images on any device
        _generator = torch.Generator().manual_seed(seed)
        x = torch.randn(len(labels), 3, image_size, image_size, generator=_generator).to(device)
        null_labels = torch.full_like(labels, model.null_class)
        trajectory = [x]

        dt = 1.0 / num_steps
        for i in range(num_steps):
            t = torch.full((len(labels),), i * dt, device=device)
            # Evaluate the conditional and unconditional velocities in a single batch
            v_cond, v_uncond = model(torch.cat([x, x]), torch.cat([t, t]), torch.cat([labels, null_labels])).chunk(2)
            v = v_uncond + guidance * (v_cond - v_uncond)
            x = x + dt * v
            trajectory.append(x)

        return torch.stack(trajectory) if return_trajectory else x

    return (generate_images,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Training

    The training step is the same five lines as for the moons: sample $t$, build $x_t$ on the line between noise and image, and regress the velocity $x_1 - x_0$. The only additions are the label dropout for guidance and the EMA update. Every few epochs we show samples for a few species: sunflower, rose, passion flower, water lily, bird of paradise and common dandelion.
    """)
    return


@app.cell
def _(flower_names):
    _chosen = ['sunflower', 'rose', 'passion flower', 'water lily', 'bird of paradise', 'common dandelion']
    SHOWCASE_CLASSES = [flower_names.index(_n) for _n in _chosen if _n in flower_names] or [53, 73, 76, 72, 7, 49]
    print({flower_names[_c]: _c for _c in SHOWCASE_CLASSES})
    return (SHOWCASE_CLASSES,)


@app.cell
def _(
    IMG_SIZE,
    SHOWCASE_CLASSES,
    device,
    ema_cnn,
    generate_images,
    img_loader,
    show_image_grid,
    torch,
    tqdm,
    velocity_cnn,
):
    IMG_EPOCHS = 50
    LABEL_DROPOUT = 0.1
    EMA_DECAY = 0.999

    def train_conditional_flow_matching(model, ema_model, loader, epochs, show_every=10):
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)
        epoch_losses = []
        global_step = 0
        for epoch in range(epochs):
            model.train()
            loader_with_progress = tqdm(loader, desc=f'Epoch {epoch+1}/{epochs}')
            total, n_batches = 0.0, 0
            for x1, labels in loader_with_progress:
                x1, labels = x1.to(device), labels.to(device)
                # Classifier-free guidance: sometimes hide the label
                drop = torch.rand(labels.shape, device=device) < LABEL_DROPOUT
                labels = torch.where(drop, torch.full_like(labels, model.null_class), labels)

                x0 = torch.randn_like(x1)
                t = torch.rand(x1.shape[0], device=device)
                x_t = (1 - t.view(-1, 1, 1, 1)) * x0 + t.view(-1, 1, 1, 1) * x1
                loss = ((model(x_t, t, labels) - (x1 - x0)) ** 2).mean()

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                # EMA update: ema = decay * ema + (1 - decay) * current
                # The decay ramps up from 0 to EMA_DECAY ("warmup"), otherwise early in training
                # the average would still be dominated by the random initial weights
                global_step += 1
                decay = min(EMA_DECAY, (1 + global_step) / (10 + global_step))
                with torch.no_grad():
                    for p_ema, p in zip(ema_model.parameters(), model.parameters()):
                        p_ema.lerp_(p, 1 - decay)
                    for b_ema, b in zip(ema_model.buffers(), model.buffers()):
                        b_ema.copy_(b)

                total += loss.item()
                n_batches += 1
                loader_with_progress.set_postfix(loss=f'{loss.item():.4f}')

            epoch_losses.append(total / n_batches)
            if (epoch + 1) % show_every == 0 or epoch == epochs - 1:
                _labels = torch.tensor(SHOWCASE_CLASSES, device=device).repeat_interleave(2)
                show_image_grid(generate_images(ema_model, _labels, IMG_SIZE, num_steps=20, guidance=2.0),
                                nrow=len(_labels), title=f"Epoch {epoch+1} - loss {epoch_losses[-1]:.4f}")

        print("Finished Training")
        return epoch_losses

    img_losses = train_conditional_flow_matching(velocity_cnn, ema_cnn, img_loader, epochs=IMG_EPOCHS)
    return (img_losses,)


@app.cell
def _(img_losses, plt):
    plt.figure(figsize=(7, 4))
    plt.plot(range(1, len(img_losses) + 1), img_losses)
    plt.xlabel('Epoch')
    plt.ylabel('Flow matching loss (MSE)')
    plt.title('Image flow matching training curve')
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Generating flowers

    Each row below is one species, and each column uses the same starting noise $x_0$, so when you move the sliders only the guidance strength or the number of steps changes. Things to try:

    - Set the guidance $w = 0$: what kind of images do you get, and do the rows still differ?
    - Increase $w$ to 4 or 5: what happens to the colors and to the diversity within a row?
    - Reduce the number of Euler steps to 1 or 2: does it look like the one-step moons from before?
    """)
    return


@app.cell
def _(mo):
    guidance_slider = mo.ui.slider(label="Guidance $w$", value=2.0, start=0.0, stop=6.0, step=0.5)
    img_steps_slider = mo.ui.slider(label="Euler steps", value=20, start=1, stop=50, step=1)
    mo.hstack([guidance_slider, img_steps_slider], justify='start', gap=2)
    return guidance_slider, img_steps_slider


@app.cell
def _(
    IMG_SIZE,
    SHOWCASE_CLASSES,
    device,
    ema_cnn,
    flower_names,
    generate_images,
    guidance_slider,
    img_steps_slider,
    plt,
    to_display,
    torch,
):
    _per_class = 8
    _labels = torch.tensor(SHOWCASE_CLASSES, device=device).repeat_interleave(_per_class)
    # Same seed for every class so that each column starts from comparable noise
    _images = torch.cat([
        generate_images(ema_cnn, _labels[_i * _per_class:(_i + 1) * _per_class], IMG_SIZE,
                        num_steps=img_steps_slider.value, guidance=guidance_slider.value, seed=0)
        for _i in range(len(SHOWCASE_CLASSES))
    ])
    _images = to_display(_images)

    _fig, _axes = plt.subplots(len(SHOWCASE_CLASSES), _per_class, figsize=(_per_class * 1.3, len(SHOWCASE_CLASSES) * 1.3))
    for _r, _c in enumerate(SHOWCASE_CLASSES):
        for _j in range(_per_class):
            _ax = _axes[_r, _j]
            _ax.imshow(_images[_r * _per_class + _j].permute(1, 2, 0))
            _ax.set_xticks([])
            _ax.set_yticks([])
        _axes[_r, 0].set_ylabel(flower_names[_c], rotation=0, ha='right', va='center', fontsize=9)
    plt.suptitle(f"Guidance w = {guidance_slider.value}, {img_steps_slider.value} Euler steps")
    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Watching an image flow

    Finally, let's look at the intermediate states $x_t$ along the ODE trajectory for one sample per species: the same picture as the moons snapshots, but now each point is a $3 \times 32 \times 32$ image. Notice that the coarse layout and colors are decided early, and fine details appear only near $t = 1$.
    """)
    return


@app.cell
def _(
    IMG_SIZE,
    SHOWCASE_CLASSES,
    device,
    ema_cnn,
    flower_names,
    generate_images,
    plt,
    to_display,
    torch,
):
    _num_steps = 20
    _labels = torch.tensor(SHOWCASE_CLASSES, device=device)
    _traj = generate_images(ema_cnn, _labels, IMG_SIZE, num_steps=_num_steps, guidance=2.0, seed=1, return_trajectory=True)
    _shown_steps = [0, 4, 8, 12, 16, 18, 20]

    _fig, _axes = plt.subplots(len(SHOWCASE_CLASSES), len(_shown_steps), figsize=(len(_shown_steps) * 1.4, len(SHOWCASE_CLASSES) * 1.4))
    for _r, _c in enumerate(SHOWCASE_CLASSES):
        for _j, _s in enumerate(_shown_steps):
            _ax = _axes[_r, _j]
            _ax.imshow(to_display(_traj[_s, _r]).permute(1, 2, 0))
            _ax.set_xticks([])
            _ax.set_yticks([])
            if _r == 0:
                _ax.set_title(f"t = {_s / _num_steps:.1f}", fontsize=9)
        _axes[_r, 0].set_ylabel(flower_names[_c], rotation=0, ha='right', va='center', fontsize=9)
    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Keep this picture in mind for the next module: **diffusion models** also turn noise into images through many small steps, and in fact a diffusion model can be rewritten as a flow along a *curved* path between noise and data. Flow matching chooses the straightest possible path, which is a big part of why it trains and samples efficiently.

    ## Exercises

    **RealNVP**

    1. **Number of layers.** Retrain with `COUPLING_LAYERS = 2`. Can the flow still produce two moons? Why do we need at least two layers, and why does having an even number make sense here?
    2. **Remove the `tanh`.** Delete the final `nn.Tanh()` from the scale network and retrain with a higher learning rate (e.g. `1e-2`). What happens to the loss?
    3. **Volume preservation.** Set the scale to zero (`s = torch.zeros_like(...)`) so the layers only translate. This is the original NICE model. What is the log-determinant now, and how does the quality of the samples change?
    4. **Other datasets.** Replace `make_moons` with `datasets.make_circles(n_samples=N_TRAIN, noise=0.05, factor=0.5)`. Is it harder to learn? Look at the "bridge" artifacts again and relate them to the topology of the data.

    **Flow matching**

    5. **Reflow again.** Apply reflow a second time, starting from `reflow_model`. Does the path straightness ratio get closer to 1, and do the one-step samples improve?
    6. **A different path.** Replace the straight line with $x_t = \cos(\frac{\pi}{2} t)\, x_0 + \sin(\frac{\pi}{2} t)\, x_1$ (this kind of curved path will come back with diffusion models). What is the new target velocity $\frac{dx_t}{dt}$? Compare how many Euler steps you need.
    7. **Better solver.** Replace Euler with the midpoint method: $x_{t+\Delta t} = x_t + \Delta t \, v\big(x_t + \frac{\Delta t}{2} v(x_t, t), t + \frac{\Delta t}{2}\big)$. Each step costs two network evaluations; is it worth it for the same total number of evaluations?
    8. **Flow between two datasets.** Flow matching does not require $x_0$ to be Gaussian noise. Train a model whose $x_0$ comes from `make_circles` and $x_1$ from `make_moons`. What does sampling now do?
    9. **Residual connections.** In `ConditionalCNN.forward`, replace `hidden = hidden + ...` with `hidden = ...` and retrain for a few epochs. Compare the loss curves.
    10. **Images.** Train at `IMG_SIZE = 64`, or make the model unconditional (always use the null label). How do the samples change?
    """)
    return


if __name__ == "__main__":
    app.run()
