import marimo

__generated_with = "0.21.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    import numpy as np
    import torch
    import torch.nn as nn
    import matplotlib.pyplot as plt

    return nn, np, plt, torch


@app.cell
def _(torch):
    device = (
        torch.device("mps")
        if torch.backends.mps.is_available()
        else torch.device("cuda")
        if torch.cuda.is_available()
        else torch.device("cpu")
    )
    print(f"Using device: {device}")
    return (device,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Module 7: Practical 3 - Safe Planning with Diffusion (SafeDiffuser)

    In the previous practical we trained a diffusion model to **generate images**. Diffusion
    models are also powerful **planners**: instead of pixels, the model generates a whole
    *trajectory* (a sequence of states) at once, conditioned on where we start and where we
    want to go. This is the idea behind *Diffuser* (Janner et al., 2022).

    There is one catch that matters a lot for robotics: a vanilla diffusion planner has
    **no safety guarantees**. It starts from pure Gaussian noise and denoises toward a
    plausible-looking path, but nothing stops that path from driving straight through a wall
    or an obstacle.

    In this notebook we reproduce, in miniature, the core idea of
    **SafeDiffuser** (Xiao, Wang, Gan, Rus, MIT 2023): we embed a *Control Barrier Function*
    (CBF) into the **denoising process itself** so that the generated trajectory is
    *guaranteed* to avoid an obstacle by the time denoising finishes — while staying as close
    as possible to what the diffusion model wanted to produce.

    Plan for the notebook:

    1. Train a tiny diffusion model that generates 2D paths from a start to a goal.
    2. Watch it happily cut **through** a circular obstacle (it was never told the obstacle exists).
    3. Add a CBF safety layer to each denoising step (the **Robust-Safe** diffuser) and watch
       the paths bend **around** the obstacle.
    4. Look at a hard case where this simple version gets *stuck* — motivating the more
       advanced variants in the paper.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. The planning problem

    A **trajectory** is a sequence of $H$ waypoints in the plane:

    $$\tau = (x_0, x_1, \dots, x_{H-1}), \qquad x_k \in \mathbb{R}^2.$$

    We will train a diffusion model to generate such trajectories, **conditioned** on the
    start point $x_0$ and the goal point $x_{H-1}$.

    For training data we synthesize smooth paths that go from the left edge of the box to the
    right edge, with a random sideways "wobble" so the model sees a variety of paths. Crucially,
    **the training data knows nothing about any obstacle** — the paths are free to cross the
    middle of the box. That is exactly the situation SafeDiffuser is designed for: the obstacle
    is a *constraint we impose at generation time*, not something baked into the data.
    """)
    return


@app.cell
def _(np, torch):
    torch.manual_seed(0)
    np.random.seed(0)

    H = 32  # waypoints per trajectory
    N_TRAJ = 4000  # number of training trajectories

    def make_dataset(n):
        t = np.linspace(0.0, 1.0, H)[None, :]  # (1, H)
        # start on the left edge, goal on the right edge, at varied heights
        start = np.stack(
            [np.random.uniform(-1.0, -0.6, n), np.random.uniform(-0.7, 0.7, n)], axis=1
        )
        goal = np.stack(
            [np.random.uniform(0.6, 1.0, n), np.random.uniform(-0.7, 0.7, n)], axis=1
        )
        # straight-line interpolation start -> goal
        base = start[:, None, :] * (1 - t[..., None]) + goal[:, None, :] * t[..., None]
        # smooth perpendicular wobble (random sign): varied, obstacle-UNAWARE prior
        direction = goal - start
        perp = np.stack([-direction[:, 1], direction[:, 0]], axis=1)
        perp = perp / (np.linalg.norm(perp, axis=1, keepdims=True) + 1e-9)
        amp = np.random.uniform(-0.35, 0.35, size=(n, 1))
        wobble = amp * np.sin(np.pi * t)  # zero at the endpoints
        traj = base + perp[:, None, :] * wobble[..., None]
        return traj.astype(np.float32), start.astype(np.float32), goal.astype(np.float32)

    traj_np, start_np, goal_np = make_dataset(N_TRAJ)
    traj = torch.tensor(traj_np)  # (N, H, 2)
    starts = torch.tensor(start_np)  # (N, 2)
    goals = torch.tensor(goal_np)  # (N, 2)
    cond_all = torch.cat([starts, goals], dim=-1)  # (N, 4) conditioning vector
    return H, N_TRAJ, cond_all, traj


@app.cell
def _(plt, traj):
    _fig, _ax = plt.subplots(figsize=(5, 5))
    for _k in range(0, 60):
        _ax.plot(traj[_k, :, 0], traj[_k, :, 1], lw=0.8, alpha=0.6)
    _ax.set_title("A sample of the (obstacle-unaware) training trajectories")
    _ax.set_xlim(-1.1, 1.1)
    _ax.set_ylim(-1.1, 1.1)
    _ax.set_aspect("equal")
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. A tiny trajectory diffusion model

    We reuse the DDPM machinery from the previous practical, just on trajectories instead of
    images. We pick a **cosine noise schedule** giving cumulative signal coefficients
    $\bar\alpha_i$, and train a small network $\epsilon_\theta(\tau^i, i, \text{cond})$ to predict
    the noise added to a clean trajectory $\tau^0$.

    For the network we use **1-D convolutions over the horizon axis** (a miniature of the temporal
    U-Net in the original Diffuser). Treating the trajectory as a length-$H$ signal with 2 channels
    means each output waypoint is computed from its *neighbours*, so the generated paths come out
    **smooth**. A plain MLP on the flattened trajectory has no such coupling and produces jagged
    paths. Start, goal, and the diffusion step are injected as an additive embedding at each block.

    Recall the two ingredients:

    - **Forward (noising):** $\;\tau^i = \sqrt{\bar\alpha_i}\,\tau^0 + \sqrt{1-\bar\alpha_i}\,\epsilon.$
    - **Training loss:** $\;\mathbb{E}\big[\,\lVert \epsilon - \epsilon_\theta(\tau^i, i, \text{cond})\rVert^2\,\big].$

    Note the two different "times" we will keep straight throughout: the **diffusion step** $i$
    (noise level, runs $N\!\to\!0$ during sampling) and the **planning index** $k$ (position
    along the trajectory, $0 \to H{-}1$).
    """)
    return


@app.cell
def _(device, np, torch):
    # Cosine noise schedule (Nichol & Dhariwal); keep the buffers on `device`.
    T = 100
    _t_grid = torch.linspace(0, T, T + 1)
    _abar = torch.cos(((_t_grid / T) + 0.008) / 1.008 * np.pi / 2) ** 2
    _abar = _abar / _abar[0]
    betas = (1 - _abar[1:] / _abar[:-1]).clamp(1e-4, 0.999).to(device)
    alphas = 1.0 - betas
    alphas_cum = torch.cumprod(alphas, dim=0)  # \bar{alpha}_i, shape (T,)
    sqrt_acum = torch.sqrt(alphas_cum)
    sqrt_1macum = torch.sqrt(1 - alphas_cum)
    return T, alphas, betas, sqrt_1macum, sqrt_acum


@app.cell
def _(nn, np, torch):
    def timestep_emb(t, d=32):
        """Sinusoidal embedding of the (integer) diffusion step t -> (B, d)."""
        half = d // 2
        freqs = torch.exp(
            -np.log(10000)
            * torch.arange(half, dtype=torch.float32, device=t.device)
            / (half - 1)
        )
        ang = t[:, None].float() * freqs[None, :]
        return torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)

    class ConvDenoiser(nn.Module):
        """Predicts the added noise with 1-D convolutions over the *horizon* axis, so neighbouring
        waypoints share features and the generated paths come out smooth. Start, goal and the
        diffusion step are injected as an additive (FiLM-style) embedding at each block."""

        def __init__(self, ch=64, temb=32, n_blocks=4):
            super().__init__()
            self.temb = temb
            self.cond_proj = nn.Sequential(
                nn.Linear(4 + temb, ch), nn.SiLU(), nn.Linear(ch, ch)
            )
            self.inp = nn.Conv1d(2, ch, kernel_size=5, padding=2)
            self.blocks = nn.ModuleList(
                [nn.Conv1d(ch, ch, kernel_size=5, padding=2) for _ in range(n_blocks)]
            )
            self.out = nn.Conv1d(ch, 2, kernel_size=5, padding=2)
            nn.init.zeros_(self.out.weight)  # start by predicting ~0 noise (stable)
            nn.init.zeros_(self.out.bias)
            self.act = nn.SiLU()

        def forward(self, x, t, cond):
            # x: (B, H, 2) -> treat the 2 coordinates as channels, H as the length
            h = x.transpose(1, 2)  # (B, 2, H)
            c = torch.cat([cond, timestep_emb(t, self.temb)], dim=-1)  # (B, 4 + temb)
            c = self.cond_proj(c)[:, :, None]  # (B, ch, 1), broadcast over time
            h = self.act(self.inp(h) + c)
            for conv in self.blocks:
                h = h + self.act(conv(h) + c)  # residual temporal block
            return self.out(h).transpose(1, 2)  # (B, H, 2)

    return (ConvDenoiser,)


@app.cell
def _(ConvDenoiser, N_TRAJ, T, cond_all, device, sqrt_1macum, sqrt_acum, torch, traj):
    # Instantiate AND train in the same cell, then return the trained `model`, so that the
    # sampling cells below genuinely depend on training having finished (marimo dataflow).
    torch.manual_seed(0)
    model = ConvDenoiser().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)

    ITERS = 4000
    BS = 256
    for _it in range(ITERS):
        _idx = torch.randint(0, N_TRAJ, (BS,))
        _x0 = traj[_idx].to(device)
        _c = cond_all[_idx].to(device)
        _t = torch.randint(0, T, (BS,), device=device)
        _noise = torch.randn_like(_x0)
        _xt = sqrt_acum[_t][:, None, None] * _x0 + sqrt_1macum[_t][:, None, None] * _noise
        _pred = model(_xt, _t, _c)
        _loss = ((_pred - _noise) ** 2).mean()
        opt.zero_grad()
        _loss.backward()
        opt.step()
        if (_it + 1) % 500 == 0:
            print(f"iter {_it + 1:4d}   loss {_loss.item():.4f}")
    print("training done")
    return (model,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2½. Sanity check: does the trained planner work?

    Before introducing any obstacle, let's confirm the model learned its basic job — turning
    Gaussian noise into smooth paths that connect the requested start and goal. We first define
    the reverse-diffusion **sampler**.

    (The sampler already contains a `safe=True` branch with a CBF safety layer — ignore that for
    now, it is explained in §4. With `safe=False` it is just an ordinary DDPM ancestral sampler.)
    """)
    return


@app.cell
def _(H, T, alphas, betas, device, model, sqrt_1macum, torch):
    @torch.no_grad()
    def sample(cond, safe=False, mode="ros", alpha=1.0, record_b=False, seed=0, eta=0.5,
               cbf_b=None, cbf_grad=None, w0=0.5, n_extra=20, max_step=None):
        """Reverse-diffusion sampler.

        safe=False -> ordinary DDPM ancestral sampling (the vanilla diffuser).
        safe=True  -> apply the CBF-QP safety projection at every step. Needs `cbf_b`/`cbf_grad`.
        mode       -> which safe diffuser (all three are the SAME half-space projection, §4/§7):
            "ros"  Robust-Safe        : hard constraint from the first step (§4).
            "res"  Relaxed-Safe       : the QP denominator is inflated by w(i)^2 with w(i) = w0*i/N
                                        decaying linearly to 0, followed by `n_extra` post-loop
                                        hard projection-only steps (the paper's "diffusion time <0"
                                        cleanup, with u_nom=0 so the denoiser doesn't fight back).
            "tvs"  Time-Varying-Safe  : enforce b(x) - gamma(i) >= 0 on a rising floor gamma(i)
                                        with gamma(N) <= b(x^N) and gamma(0) = 0, plus a -gamma_dot
                                        feed-forward term.
        eta        -> scale on the injected sampling noise; lower = smoother (less varied) paths.
        max_step   -> if set, cap |safety correction| per step (a trust region). nabla b -> 0 at the
                      obstacle centre, so the raw projection can blow up; the cap keeps the step
                      small (the paper's "small dtau") and avoids teleporting waypoints (chords).

        Noise is drawn with a CPU generator (reproducible on any device) and moved to `device`;
        outputs are returned on CPU so the plotting/metric cells need no device handling.
        """
        g = torch.Generator().manual_seed(seed)  # CPU RNG -> same paths on cpu/mps/cuda
        b = cond.shape[0]
        cond = cond.to(device)
        st, gl = cond[:, :2], cond[:, 2:]
        x = torch.randn(b, H, 2, generator=g).to(device)
        x[:, 0], x[:, -1] = st, gl  # condition on start/goal
        # TVS floor: gamma(N) = clamp(b(x^N), max=0) <= b(x^N), per waypoint, rises to 0 at i=0.
        g_N = cbf_b(x).clamp(max=0.0) if (safe and mode == "tvs" and cbf_b is not None) else None
        b_hist = []
        for i in reversed(range(T)):
            t = torch.full((b,), i, dtype=torch.long, device=device)
            eps = model(x, t, cond)
            mean = (x - betas[i] / sqrt_1macum[i] * eps) / torch.sqrt(alphas[i])
            if i > 0:
                noise = torch.randn(b, H, 2, generator=g).to(device)
                x_prop = mean + eta * torch.sqrt(betas[i]) * noise
            else:
                x_prop = mean

            if safe:
                # treat the vanilla step as the nominal control u_nom = x_prop - x
                u = x_prop - x
                grad = cbf_grad(x)  # (b, H, 2)
                bval = cbf_b(x)  # (b, H)
                gu = (grad * u).sum(-1)  # nabla b . u_nom
                denom = (grad**2).sum(-1) + 1e-6
                if mode == "res":
                    # w(i) decays linearly to 0; gentle early so the path commits to a side
                    # instead of being pinned to the boundary, then hard at the end.
                    w = w0 * i / (T - 1)
                    lhs = gu + alpha * bval  # same numerator as RoS ...
                    denom = denom + w**2  # ... but inflated denom -> gentler correction
                elif mode == "tvs":
                    gamma_now = g_N * ((i + 1) / (T - 1))  # rising floor at current level
                    gamma_next = g_N * (i / (T - 1))  # ... and at the next level
                    gamma_dot = gamma_next - gamma_now  # rate the floor rises (>= 0)
                    lhs = gu - gamma_dot + alpha * (bval - gamma_now)
                else:  # "ros"
                    lhs = gu + alpha * bval  # CBF condition h(u) >= 0
                viol = lhs < 0
                lam = torch.where(viol, (-lhs) / denom, torch.zeros_like(lhs))
                corr = lam[..., None] * grad  # the safety correction
                if max_step is not None:  # trust region: cap |correction| (small-dtau)
                    cnrm = corr.norm(dim=-1, keepdim=True) + 1e-9
                    corr = corr * (max_step / cnrm).clamp(max=1.0)
                x = x + u + corr  # nominal denoising + (clamped) safety correction
            else:
                x = x_prop

            x[:, 0], x[:, -1] = st, gl  # re-apply start/goal conditioning
            if record_b and cbf_b is not None:
                b_hist.append(cbf_b(x).min(dim=1).values.clone())  # worst waypoint

        # ReS only: n_extra "diffusion time < 0" cleanup steps. u_nom = 0 (no fresh denoising),
        # so the conv prior doesn't fight back -- this is what certifies the final hard b >= 0.
        if safe and mode == "res" and cbf_b is not None:
            for _ in range(n_extra):
                grad = cbf_grad(x)
                bval = cbf_b(x)
                lhs = alpha * bval  # u_nom = 0
                viol = lhs < 0
                denom = (grad**2).sum(-1) + 1e-6
                lam = torch.where(viol, (-lhs) / denom, torch.zeros_like(lhs))
                corr = lam[..., None] * grad
                if max_step is not None:
                    cnrm = corr.norm(dim=-1, keepdim=True) + 1e-9
                    corr = corr * (max_step / cnrm).clamp(max=1.0)
                x = x + corr
                x[:, 0], x[:, -1] = st, gl
                if record_b:
                    b_hist.append(cbf_b(x).min(dim=1).values.clone())

        if record_b:
            return x.cpu(), torch.stack(b_hist, 0).cpu()  # (steps, b)
        return x.cpu()

    return (sample,)


@app.cell
def _(plt, sample, torch):
    # Three left->goal pairs at different heights (the middle one runs through where the
    # obstacle will later sit). For each we draw 3 samples to show the model learned a
    # *distribution* of plausible paths, not one fixed answer. A small eta keeps them legible.
    _demo = torch.tensor(
        [
            [-0.9, 0.45, 0.9, 0.45],
            [-0.9, 0.00, 0.9, 0.00],
            [-0.9, -0.45, 0.9, -0.45],
        ]
    )
    _fig, _ax = plt.subplots(figsize=(5, 5))
    for _j in range(_demo.shape[0]):
        for _s in range(3):
            _p = sample(_demo[_j : _j + 1], safe=False, seed=_s, eta=0.25)[0]
            _ax.plot(_p[:, 0], _p[:, 1], "-", lw=1.3, alpha=0.8, color=f"C{_j}")
        _ax.scatter(*_demo[_j, :2], c="limegreen", s=70, zorder=5, edgecolor="k")
        _ax.scatter(*_demo[_j, 2:], c="red", s=70, zorder=5, edgecolor="k")
    _ax.set_title("Generated paths, no obstacle\n(green = start, red = goal; 3 samples per pair)")
    _ax.set_xlim(-1.1, 1.1)
    _ax.set_ylim(-1.1, 1.1)
    _ax.set_aspect("equal")
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. The obstacle and the Control Barrier Function

    We place a circular obstacle of radius $R$ at the center $c$ and define a **safety function**

    $$b(x) = \lVert x - c \rVert^2 - R^2.$$

    The **safe set** is $\;C = \{x : b(x) \ge 0\}$, i.e. everything *outside* the circle.
    A trajectory is safe when **every** waypoint satisfies $b(x_k) \ge 0$.

    Because the waypoints are discrete, the straight segment *between* two waypoints sitting
    right on the circle would still clip the obstacle. To account for this we plan against a
    slightly **inflated** radius `R_PLAN = R + margin` — a standard trick (it plays the role of
    a robot radius). The gradient we will need is simply $\nabla b(x) = 2(x - c)$.
    """)
    return


@app.cell
def _(torch):
    OBS_C = torch.tensor([0.0, 0.0])
    OBS_R = 0.35  # true obstacle radius (what we draw / score against)
    MARGIN = 0.07  # safety margin for discrete waypoints
    R_PLAN = OBS_R + MARGIN  # radius used inside the CBF

    def cbf_b(x):
        # x: (..., 2);  b >= 0  <=>  outside the inflated obstacle (safe)
        return ((x - OBS_C.to(x.device)) ** 2).sum(-1) - R_PLAN**2

    def cbf_grad(x):
        # db/dx = 2 (x - c)
        return 2 * (x - OBS_C.to(x.device))

    return OBS_C, OBS_R, cbf_b, cbf_grad


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. SafeDiffuser: a CBF in *diffusion time*

    Here is the key insight of the paper. Normally a CBF is applied along a robot's *physical*
    dynamics. SafeDiffuser instead applies it along the **denoising dynamics**. Think of one
    reverse diffusion step as a small move

    $$\dot\tau = u, \qquad u_{\text{nom}} = \tau^{\,\text{vanilla next}} - \tau^{\,\text{current}},$$

    where $u_{\text{nom}}$ is the move the vanilla diffuser *wanted* to make. We are free to
    replace it with a slightly different $u$, as long as we don't drift far from the model.

    For each waypoint $x_k$ we require the **discrete CBF condition**

    $$\nabla b(x_k)^\top u_k + \alpha\, b(x_k) \;\ge\; 0, \qquad \alpha \in (0, 1].$$

    Why does this give a guarantee? Substituting $u_k = x_k^{\text{next}} - x_k$ and using a
    first-order expansion, the condition is exactly

    $$b(x_k^{\text{next}}) \;\ge\; (1-\alpha)\, b(x_k).$$

    - If $x_k$ is already safe ($b\ge 0$): then $b^{\text{next}} \ge (1-\alpha)b \ge 0$ — it
      **stays** safe.
    - If $x_k$ is unsafe ($b < 0$, as it is at the start, when $\tau^N$ is pure noise): then
      $b$ improves by at least $\alpha\,|b|$ every step, so it reaches the safe set in a
      **finite** number of diffusion steps and stays there.

    That is the paper's **finite-time diffusion invariance**: even though we start outside the
    safe set, we are guaranteed to be inside it by the end of denoising.

    ### Solving for $u$: a one-line quadratic program

    We want to stay as close as possible to the diffuser's intended move, so we solve, at every
    diffusion step, the **minimum-deviation QP**

    $$u^\star = \arg\min_u \tfrac12 \lVert u - u_{\text{nom}}\rVert^2
    \quad \text{s.t.} \quad \nabla b(x_k)^\top u_k + \alpha\, b(x_k) \ge 0 \;\; \forall k.$$

    Because each waypoint has its own independent constraint, this decouples into one tiny QP
    per waypoint, each of which is just a **projection onto a half-space** with a closed form:

    $$u_k^\star = u_{\text{nom},k} + \frac{\big[-(\nabla b^\top u_{\text{nom}} + \alpha b)\big]_+}{\lVert \nabla b\rVert^2}\,\nabla b.$$

    So no QP solver is needed — when a waypoint's nominal move would violate safety, we add the
    smallest radial correction that restores the constraint (and otherwise leave it untouched).
    This is the paper's **Robust-Safe (RoS)** diffuser.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4½. The finite-time invariance proof, made more careful

    The paragraph above promised that the discrete CBF condition
    $b(x^{\text{next}}) \ge (1-\alpha)\,b(x)$ gives "finite-time diffusion invariance with
    probability $\approx 1$." The paper's appendix proof of Theorem 3.2 sketches this in three
    moves — Lyapunov contraction, Gaussian crossing, Nagumo lock-in — but works in continuous
    time and reports its conclusion as if the discrete step index were continuous time. Here is
    a tightened version, in **discrete time**, with $\Delta\tau$ carried through and the
    standing assumptions called out explicitly.

    ### Standing assumptions (the small print the paper omits)

    **(A1) Step size $\Delta\tau$.** One reverse-diffusion step advances $\tau^{j+1} \to \tau^j$
    by Euler-stepping the dynamics $\dot\tau = u$, i.e. $\tau^j = \tau^{j+1} + u^j\,\Delta\tau$.
    The paper introduces $\Delta\tau$ (in eq. 7 and the QP) but quietly absorbs it into the
    contraction rate when stating Theorem 3.2.

    **(A2) Linear class-$\mathcal K$ gain.** $\alpha_{\text{cts}}(s) = \alpha_0\,s$ with
    $\alpha_0 > 0$ and the **step-size stability condition** $\alpha_0\,\Delta\tau \le 1$
    (otherwise the discrete contraction overshoots and the analysis breaks). We will abbreviate

    $$\alpha \;:=\; \alpha_0\,\Delta\tau \;\in\; (0, 1],$$

    which is the **per-step gain** — exactly what the `alpha` slider in §5 controls.

    *Why linear specifically, when the paper says "extended class $\mathcal K$"?* Theorem 3.2 is
    **stated** for an arbitrary extended class-$\mathcal K$ function, but its proof handwaves
    the general case with *"by Lyapunov stability theory"* (asymptotic, no rate) and then writes
    *"Specifically, when $\alpha$ is a linear function..."* before producing the **only**
    explicit decay bound in the proof. The Gaussian-crossing step downstream needs that rate to
    argue $V_i$ shrinks fast enough for $p_i$ to remain bounded below. So the theorem's stated
    generality is **unproven** for non-linear extended class $\mathcal K$, and (A2) is the
    honest hypothesis under which the proof actually closes. The demo code is linear in $\alpha$
    (a scalar multiplying `bval`), placing it squarely inside (A2).

    **(A3) Convex barrier.** $\nabla^2 b \succeq 0$. (Our $b(x) = \|x-c\|^2 - R^2$ satisfies this
    with $\nabla^2 b = 2I$.) The QP enforces a *linearised* condition; the quadratic remainder
    $\tfrac12 \Delta\tau^2\, (u^j)^\top \nabla^2 b\, u^j$ is **non-negative** under (A3), so the
    linearisation is *conservative* — the actual $b^{\text{next}}$ improves at least as much as
    the linearisation predicts. Without (A3) one needs a Lipschitz bound on $\nabla b$ and the
    final inequalities pick up an explicit $O(\Delta\tau^2)$ slack.

    **Setup.** Fix a waypoint $k$ and write $b_j := b(x_k^j)$, $V_j := -b_j$. The QP enforces
    $\nabla b\cdot u^j + \alpha_0\,b \ge 0$ at the current $x_k^{j+1}$. Substituting
    $u^j\,\Delta\tau = x_k^j - x_k^{j+1}$, linearising $b$, and using (A3):

    $$\boxed{\;b_j \;\ge\; (1 - \alpha_0\,\Delta\tau)\,b_{j+1} \;=\; (1-\alpha)\,b_{j+1}\;} \tag{$\star$}$$

    — rigorously, under (A1)–(A3). Equivalently $V_j \le (1-\alpha)\,V_{j+1}$. In the demo code
    (see §4's `sample`) the line `u = x_prop - x` is a delta, *not* a velocity, so we are running
    at $\Delta\tau = 1$ in the paper's notation and the slider's `alpha` directly equals the
    per-step gain $\alpha = \alpha_0\Delta\tau$ defined here. The code's CBF constraint
    $\nabla b\cdot u + \alpha\,b \ge 0$ (with our `u` and `alpha`) is therefore exactly $(\star)$.

    ### Case 1 — $b_N \ge 0$: deterministic.

    Apply $(\star)$ with $b_{j+1} \ge 0$: $b_j \ge (1-\alpha)b_{j+1} \ge 0$. By induction,
    $b_j \ge 0$ for every $j \in \{0,\dots,N\}$. No probability, no asymptotics, no caveats —
    under (A1)–(A3) the projection alone proves it. *This piece of the original proof is solid.*

    ### Case 2 — $b_N < 0$: the contraction step.

    Iterating $(\star)$ as a recursion on $V$ gives an explicit, finite-$N$ bound:

    $$\boxed{\;V_j \;\le\; (1-\alpha)^{\,N-j}\, V_N \;=\; (1 - \alpha_0\,\Delta\tau)^{\,N-j}\, V_N.\;} \tag{$\dagger$}$$

    Taking the continuous limit $\Delta\tau \to 0$ with $N\Delta\tau = T_{\text{tot}}$ fixed,
    $(1-\alpha_0\Delta\tau)^{N-j} \to e^{-\alpha_0\,(T_{\text{tot}} - t_j)}$ — the paper's
    $e^{-\varepsilon(N-j)}$, with the paper's $\varepsilon$ revealed as $\alpha_0\,\Delta\tau$
    (= our $\alpha$). The unit-step shortcut hidden in the paper's notation now lives in plain
    sight as $\alpha = \alpha_0\Delta\tau$. Two finite-$N$ observations the paper underplays:

    1. **$V_0 \le (1-\alpha)^N V_N$ is exponentially small but *strictly positive*.** Lyapunov
       contraction gives "very close to the boundary," not "inside the safe set." The paper's
       phrase *"stabilized to the boundary $b=0$"* is exactly right, and exactly why this step
       alone is not enough.

    2. **Without (A2), no rate at all.** A general extended class-$\mathcal K$ function gives
       only $V_j \le V_{j+1} - \alpha_{\text{cts}}(V_{j+1})\,\Delta\tau$, whose convergence
       depends on the shape of $\alpha$ near $0$: **linear** → geometric (above), **super-linear**
       like $\alpha(s)=cs^p,\,p>1$ → sub-geometric (slow as $V\to 0$), **sub-linear** like
       $\alpha(s)=c\sqrt{s}$ → *genuinely* finite-time to zero. The third case is the only one
       that would justify the theorem's "finite-time" label without invoking Gaussian jumps at
       all — the paper does not work it out.

    ### Case 2, continued — the probabilistic crossing.

    To bridge "$V_0$ exponentially small" $\to$ "$V_0 = 0$" (i.e. $b_0 \ge 0$) the paper invokes
    the random Gaussian transitions: at each step there is *some* probability $p$ that the
    next waypoint lands just inside the safe set. The argument as written needs three patches.

    **Patch 1 — $p$ is not constant.** In DDPM the injected noise scale $\sqrt{\beta_i}$
    shrinks as $i \to 0$, exactly when $(\dagger)$ says the state is closest to the boundary.
    Let $p_i$ be the per-step crossing probability at step $i$. The right statement is

    $$P(\text{never crossed during steps } N{-}1, \dots, j) \;\le\; \prod_{i=j}^{N-1}(1 - p_i).$$

    **Patch 2 — independence is false.** The events "$b_l \ge 0$" at different steps are not
    independent (each $x^j$ depends on the whole history). The Bernoulli formula
    $1-(1-p)^j$ in the paper is at best a *lower bound* via the union bound. The right
    machinery is Borel–Cantelli (second), which gives

    $$\sum_{i=0}^{\infty} p_i \;=\; \infty \;\;\Longrightarrow\;\; P\big(\text{cross at least once}\big) = 1.$$

    For our cosine schedule with $\beta_i \in [10^{-4}, 0.999]$ this is satisfied — but the
    paper does not state, let alone check, the condition.

    **Patch 3 — what "$p_i$" actually means.** A clean sufficient condition: at step $i$,
    given history, $p_i \ge p_{\min}(\beta_i, V_i, \|\nabla b\|)$ where $p_{\min}$ is the
    probability that a one-dimensional Gaussian with scale $\sqrt{\beta_i}\|\nabla b\|$
    exceeds the radial distance to the boundary, $V_i / \|\nabla b\|$. By $(\dagger)$ this
    distance shrinks geometrically, so $p_{\min}$ stays bounded below by a positive sequence
    even as $\beta_i$ shrinks — provided $\beta_i$ does not shrink *faster* than $V_i$. This
    is the bookkeeping the paper omits.

    ### Case 2, continued — the lock-in.

    Suppose at some step $\ell \le N-1$ we have $b_\ell \ge 0$ (Borel–Cantelli says this
    happens almost surely). Apply Case 1 from step $\ell$: $b_r \ge 0$ for all $r \le \ell$.
    **Deterministically**, by the same projection. So once safe, the waypoint stays safe.
    This is what makes the probabilistic guarantee survive all the way to $j = 0$.

    ### The honest finite-$N$ statement.

    Combining $(\dagger)$, the union bound, and the lock-in:

    $$\boxed{\;P\big(b_0 < 0\big) \;\le\; \prod_{i=0}^{N-1}(1 - p_i),\;}$$

    which tends to $0$ as $N \to \infty$ whenever $\sum_i p_i = \infty$. This is the actual
    content of *"with almost probability 1"* — it is **not** a deterministic finite-time
    result, despite the theorem's name.

    ### The gap the proof never closes — per-waypoint vs. per-trajectory.

    Everything above treats *one* waypoint in isolation. The QP decouples across waypoints,
    and the conclusion is **per-waypoint**: every $x_k^0$ lands in $\{b \ge 0\}$ almost
    surely. But a *trajectory* is only safe if every **segment** $[x_k^0, x_{k+1}^0]$ also
    avoids the obstacle, and the proof says nothing about segments.

    This is not academic. In §7 you will see a Relaxed-Safe run in which every individual
    waypoint satisfies $\min_k b_k = +0.054$ — Theorem 3.2's conclusion holds — while the
    segments chord straight through the obstacle. The standard fix is to plan against an
    inflated radius `R_PLAN = R + MARGIN` (which we do); the *guarantee itself*, however,
    does not cover segments.

    ### What is actually proved (one sentence)

    > Under **(A1)–(A3)** — Euler step of size $\Delta\tau$, linear $\alpha = \alpha_0\Delta\tau \le 1$,
    > convex barrier — and a noise schedule with $\sum_i p_i = \infty$, every individual waypoint
    > $x_k^0$ lies in $\{b \ge 0\}$ with probability tending to $1$ as $N \to \infty$. Trajectory
    > (segment) safety requires a separate argument or an inflated barrier.

    That is *"finite-time diffusion invariance, almost surely, per waypoint, under (A1)–(A3)"* —
    a useful and non-trivial guarantee, but the original phrasing oversells it on **four** counts:

    1. **Continuous- vs. discrete-time**: $\Delta\tau$ silently disappears into the contraction rate.
    2. **Deterministic vs. probabilistic**: the result is *almost surely*, not surely.
    3. **Asymptotic-in-$N$ vs. finite-$N$**: $P(b_0 < 0) \to 0$ only as $N \to \infty$.
    4. **Waypoint vs. trajectory**: per-waypoint safety does not imply segment safety.

    The same caveats are inherited by every variant the paper proposes (ReS, TVS, and the
    high-relative-degree extensions).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. Vanilla diffuser vs. SafeDiffuser

    We test on six left-to-right paths at different heights $y_0$. Whenever $|y_0| < R$ the
    straight path runs into the obstacle. Use the slider to change the CBF gain $\alpha$ (how
    aggressively safety is enforced per step) and watch the safe paths bend around the obstacle.
    """)
    return


@app.cell
def _(mo):
    alpha_slider = mo.ui.slider(start=0.1, stop=1.0, step=0.1, value=1.0, label="CBF gain α")
    alpha_slider
    return (alpha_slider,)


@app.cell
def _(OBS_C, OBS_R, alpha_slider, cbf_b, cbf_grad, sample, torch):
    # six horizontal crossings; |y0| < OBS_R means the straight path hits the obstacle.
    # NOTE: y0 = +/-0.12 is deliberately near the center -> a hard "local trap" case.
    y0 = torch.tensor([-0.40, -0.25, -0.12, 0.12, 0.25, 0.40])
    test_start = torch.stack([torch.full_like(y0, -0.9), y0], -1)
    test_goal = torch.stack([torch.full_like(y0, 0.9), y0], -1)
    test_cond = torch.cat([test_start, test_goal], -1)

    van, bhist_van = sample(test_cond, safe=False, record_b=True, seed=1, cbf_b=cbf_b)
    saf, bhist_saf = sample(
        test_cond, safe=True, alpha=alpha_slider.value, record_b=True, seed=1,
        cbf_b=cbf_b, cbf_grad=cbf_grad,
    )

    def _collision_stats(x):
        # score against the TRUE obstacle radius (not the inflated planning radius)
        b = ((x - OBS_C) ** 2).sum(-1) - OBS_R**2
        return (b < 0).any(1).float().mean().item(), b.min().item()

    _vc, _vm = _collision_stats(van)
    _sc, _sm = _collision_stats(saf)
    print(f"VANILLA : fraction of paths colliding = {_vc:.2f},  min b = {_vm:+.3f}")
    print(f"SAFE    : fraction of paths colliding = {_sc:.2f},  min b = {_sm:+.3f}")
    return bhist_saf, bhist_van, saf, van


@app.cell
def _(OBS_C, OBS_R, T, bhist_saf, bhist_van, plt, saf, van):
    _fig, _ax = plt.subplots(1, 3, figsize=(15, 5))
    for _a, _X, _title in [
        (_ax[0], van, "Vanilla diffuser"),
        (_ax[1], saf, "SafeDiffuser (Robust-Safe)"),
    ]:
        _a.add_patch(plt.Circle(OBS_C.tolist(), OBS_R, color="crimson", alpha=0.3))
        for _k in range(_X.shape[0]):
            _a.plot(_X[_k, :, 0], _X[_k, :, 1], "-o", ms=2)
        _a.set_xlim(-1.1, 1.1)
        _a.set_ylim(-1.1, 1.1)
        _a.set_aspect("equal")
        _a.set_title(_title)

    _ax[2].axhline(0, color="k", lw=0.8, ls="--")
    _steps = list(range(T, 0, -1))
    _ax[2].plot(_steps, bhist_van.mean(1), color="crimson", lw=2, label="vanilla")
    _ax[2].plot(_steps, bhist_saf.mean(1), color="seagreen", lw=2, label="safe")
    _ax[2].set_xlabel("diffusion step  (N → 0)")
    _ax[2].set_ylabel("mean of (min b over waypoints)")
    _ax[2].set_title("Finite-time diffusion invariance")
    _ax[2].legend()
    _ax[2].invert_xaxis()
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6. What to notice (and the hard case)

    - **Vanilla** paths whose height falls inside the obstacle cut right through it — the
      green/red curve in the invariance plot stays **below zero**.
    - **SafeDiffuser** keeps every waypoint outside the obstacle. In the invariance plot the
      worst waypoint's $b$ is driven up and held at $\ge 0$ — finite-time diffusion invariance
      in action.

    ### The local trap

    Look at the near-center paths ($y_0 \approx \pm 0.12$, and especially the dead-center
    $y_0 = 0$ case we will add in §7). The Robust-Safe diffuser pins their mid-waypoints to the
    inflated boundary and produces a **sharp pointy peak** at the apex rather than the smooth bow
    it produces for off-center paths. Each waypoint is technically pushed out of the obstacle,
    but the path is kinked — and on a perfectly symmetric input it can get worse, with neighboring
    waypoints projected to opposite arcs and a chord between them slicing through the obstacle
    (the **local trap**).

    This is not a bug in our code; it is the exact limitation the paper calls out. Because the
    hard constraint is enforced from the very first (pure-noise) step, a waypoint in a
    symmetric/ambiguous spot can get pinned to the boundary with no time to coordinate with its
    neighbours. The paper proposes two fixes, both of which we build in §7:

    - **Relaxed-Safe (ReS)** diffuser: add a relaxation variable to the constraint with a
      time-varying weight $w_k(j)$ that decays to $0$, plus a few extra diffusion steps, so the
      constraint is soft early (letting the path commit to one side) and hard only at the end.
    - **Time-Varying-Safe (TVS)** diffuser: replace $b(x)\ge 0$ with
      $b(x) - \gamma_k(j) \ge 0$ where $\gamma_k(N)\le b(x^N)$ and $\gamma_k(0)=0$, gradually
      tightening the spec to its true value as denoising finishes.

    ### Two honest caveats

    1. **Discrete vs. continuous safety.** The CBF only constrains the discrete waypoints. The
       straight segments between them are kept clear by the inflated `R_PLAN` margin; for true
       continuous-time guarantees the paper pairs this with a lower-level CBF controller during
       execution.
    2. **The α trade-off.** Large $\alpha$ enforces safety fast but can distort the path; small
       $\alpha$ is gentler but lets waypoints drift further before being corrected. Try the
       slider to feel this trade-off.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 7. Escaping the local trap: Relaxed-Safe and Time-Varying-Safe

    The local trap comes from enforcing the **hard** constraint $b\ge 0$ from the very first
    pure-noise step: every waypoint is slammed onto the boundary immediately and can't commit to
    a side. Both of the paper's fixes loosen the constraint *early* in diffusion and tighten it to
    the true spec *only at the end* — and remarkably, each is the **same closed-form projection**
    from §4 with one term changed. (Their finite-time invariance proofs, Thms. 3.3 / 3.4, both
    reduce to the RoS proof once the relaxation / floor reaches its final value.)

    ### Relaxed-Safe (ReS)

    Add a per-waypoint relaxation variable $r_k$ to the constraint and penalise it in the cost:

    $$\min_{u,r}\; \tfrac12\lVert u-u_{\text{nom}}\rVert^2 + \tfrac12 r^2
    \quad\text{s.t.}\quad \nabla b^\top u + \alpha b - w(i)\,r \ge 0 .$$

    Solving the KKT conditions, the relaxation just **inflates the denominator** of the projection:

    $$\lambda = \frac{\big[-(\nabla b^\top u_{\text{nom}} + \alpha b)\big]_+}{\lVert\nabla b\rVert^2 + w(i)^2},
    \qquad u = u_{\text{nom}} + \lambda\,\nabla b .$$

    With $w(i)=w_0\,\tfrac{i}{N}$ large early, $\lambda$ is small — the path mostly follows the
    diffuser and **picks a side** instead of getting pinned to the boundary. As $w(i)\to 0$ the
    constraint becomes hard again (recovering RoS), and a handful of extra hard steps at
    "diffusion time $<0$" certify that every waypoint ends up safe.

    ### Time-Varying-Safe (TVS)

    Keep a hard constraint, but on a **rising floor** $\gamma(i)$ instead of $0$:

    $$b(x_k) - \gamma_k(i) \ge 0,\qquad \gamma_k(N)\le b(x_k^N),\quad \gamma_k(0)=0 .$$

    The floor starts low enough that the initial noisy waypoint already satisfies it (no early
    slamming), then rises to $0$ so the true spec is restored exactly at the end. The CBF condition
    gains a feed-forward term for the moving floor, $-\dot\gamma$, and is again one projection:

    $$\lambda = \frac{\big[-(\nabla b^\top u_{\text{nom}} - \dot\gamma + \alpha (b-\gamma))\big]_+}{\lVert\nabla b\rVert^2}.$$

    Below we run all three on **seven crossings** (the $\alpha$ slider still applies). We
    deliberately include the genuinely-symmetric $y_0 = 0$ path: at dead-center the noise prior
    has no preferred bow direction, so the radial CBF push is sign-ambiguous waypoint by waypoint
    — that is the local trap.

    What to watch:

    - **RoS** handles every off-center path cleanly (the conv prior already breaks symmetry), but
      on the dead-center path it produces a sharp pointy peak at the apex — waypoints snap to
      whichever side they happened to be nearest, with no time to coordinate. The trap.
    - **ReS** is structurally different (an inflated QP denominator + post-loop hard cleanup),
      but on this geometry the conv prior is strong enough that it ends up looking like RoS:
      same pointy peak. The cleanup steps to the right of step 0 in the invariance plot are
      what certify final safety after the relaxed middle phase.
    - **TVS** is the visible winner here: the rising floor coaxes the path out *gradually*
      throughout denoising, so the dead-center path bows over with a graceful peak instead of a
      kink, and the invariance curve traces the floor cleanly.

    All three finish at the same `min b ≈ +0.054` (every waypoint outside the inflated obstacle).
    The differences are in **path quality**, not safety — exactly the trade-off the paper studies.
    """)
    return


@app.cell
def _(OBS_C, OBS_R, alpha_slider, cbf_b, cbf_grad, sample, torch):
    # Same six crossings as §5 plus the genuinely-symmetric y0 = 0 case, which is the one
    # that exposes RoS's local trap: at dead-center the noise prior has no preferred bow
    # direction, so the radial CBF push is sign-ambiguous waypoint by waypoint.
    _y0 = torch.tensor([-0.40, -0.25, -0.12, 0.00, 0.12, 0.25, 0.40])
    _cond = torch.cat(
        [
            torch.stack([torch.full_like(_y0, -0.9), _y0], -1),
            torch.stack([torch.full_like(_y0, 0.9), _y0], -1),
        ],
        -1,
    )
    _a = alpha_slider.value
    _kw = dict(safe=True, alpha=_a, record_b=True, seed=1, cbf_b=cbf_b, cbf_grad=cbf_grad)
    ros_x, ros_b = sample(_cond, mode="ros", **_kw)
    res_x, res_b = sample(_cond, mode="res", **_kw)
    tvs_x, tvs_b = sample(_cond, mode="tvs", **_kw)

    def _stats(x):
        # score against the TRUE obstacle radius (not the inflated planning radius)
        _b = ((x - OBS_C) ** 2).sum(-1) - OBS_R**2
        return (_b < 0).any(1).float().mean().item(), _b.min().item()

    for _name, _x in [("RoS", ros_x), ("ReS", res_x), ("TVS", tvs_x)]:
        _c, _m = _stats(_x)
        print(f"{_name}: fraction colliding = {_c:.2f},  min b = {_m:+.3f}")
    return res_b, res_x, ros_b, ros_x, tvs_b, tvs_x


@app.cell
def _(OBS_C, OBS_R, T, plt, res_b, res_x, ros_b, ros_x, tvs_b, tvs_x):
    _fig, _ax = plt.subplots(2, 2, figsize=(11, 10))
    for _a, _X, _title in [
        (_ax[0, 0], ros_x, "Robust-Safe (RoS) — pointy peak at dead-center"),
        (_ax[0, 1], res_x, "Relaxed-Safe (ReS) — similar to RoS on this geometry"),
        (_ax[1, 0], tvs_x, "Time-Varying-Safe (TVS) — graceful arc at dead-center"),
    ]:
        _a.add_patch(plt.Circle(OBS_C.tolist(), OBS_R, color="crimson", alpha=0.3))
        for _k in range(_X.shape[0]):
            _a.plot(_X[_k, :, 0], _X[_k, :, 1], "-o", ms=2)
        _a.set_xlim(-1.1, 1.1)
        _a.set_ylim(-1.1, 1.1)
        _a.set_aspect("equal")
        _a.set_title(_title)

    _inv = _ax[1, 1]
    _inv.axhline(0, color="k", lw=0.8, ls="--")
    for _b, _col, _lab in [(ros_b, "C0", "RoS"), (res_b, "C1", "ReS"), (tvs_b, "C2", "TVS")]:
        # ReS records `n_extra` extra "diffusion time < 0" cleanup steps; place them past 0.
        _L = _b.shape[0]
        _steps = list(range(T, 0, -1)) + list(range(0, -(_L - T), -1))
        _inv.plot(_steps, _b.mean(1), color=_col, lw=2, label=_lab)
    _inv.set_xlabel("diffusion step  (N → 0, then extra steps < 0)")
    _inv.set_ylabel("mean of (min b over waypoints)")
    _inv.set_title("Finite-time diffusion invariance")
    _inv.legend()
    _inv.invert_xaxis()
    _fig.tight_layout()
    _fig
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
