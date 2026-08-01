# IIC

> **Research preview.** Copyright 2026 Chris van der Heide. This repository is
> temporarily available for noncommercial evaluation under the
> [PolyForm Noncommercial License 1.0.0](LICENSE).

`iic` is an experimental Python package for computing the hard-limit
interpolation information criterion (IIC), its finite-penalty soft extension,
and the component quantities needed to diagnose a fitted model. Its first
end-to-end experiment studies a family of reaction--diffusion physics-informed
neural networks (PINNs).

The hard-limit IIC is the primary object. Soft-IIC and curvature-only outputs
are explicit extensions or ablations; they are not substitutes silently used
when the hard-limit assumptions fail.

The core interfaces are intended to extend to grokking trajectories and
LoRA/adapter parameter subspaces. Those are future experiment families, not
currently documented as complete IIC workflows.

## Hard-limit problem

Let $\theta\in\mathbb R^p$ parameterise a predictor, let
$h:\mathbb R^p\rightarrow\mathbb R^N$ be its vector of scalar interpolation
constraints, and let $R:\mathbb R^p\rightarrow\mathbb R$ be the full
regularizer. The interpolating manifold is

$$
\mathcal M=\{\theta:h(\theta)=0\}.
$$

The hard-limit calculation compares a trained parameter $\theta_\star$ on
$\mathcal M$, interpreted as a constrained minimum of $R$, with an
unconstrained regularizer minimum

$$
\theta_0\in\operatorname*{arg\,min}_{\theta}R(\theta).
$$

Writing $A_\star=Dh(\theta_\star)$, the core package defines

$$
H_\star=\nabla_\theta^2R(\theta_\star),
\qquad
H_0=\nabla_\theta^2R(\theta_0).
$$

Both are Hessians of the same full regularizer. A least-squares multiplier may
be computed to diagnose constrained stationarity, but it is never inserted
into either core Hessian. Multiplier-dependent AIIC curvature is a separate,
not-yet-public evaluation path. The metric-adjusted constraint kernel is

$$
K_H=A_\star H_\star^{-1}A_\star^\mathsf{T}.
$$

For $N=\dim h$, the implemented hard-limit score is

$$
\operatorname{IIC}_{\mathrm{hard}}
=
\underbrace{\log\!\left(R(\theta_\star)-R(\theta_0)\right)}_{
  \text{energy gap}}
+\underbrace{\frac{\log\det K_H}{N}}_{
  \text{constraint geometry}}
+\underbrace{\frac{\log\det H_\star-\log\det H_0}{N}}_{
  \text{Hessian-volume gap}}
-\underbrace{\log N}_{\text{dataset correction}}.
$$

The Hessian-volume gap may equivalently be read as
$N^{-1}\log\det(H_\star H_0^{-1})$ when the required inverses and
determinants are valid. The package stores the optional direct tangent-space
`iic`, reduced hard-limit `hiic`, and finite-penalty `siic` separately,
together with their component terms. Legacy `hard_iic` and `soft_iic` aliases
remain available.

A theory-valid hard score requires, among other recorded conditions, an
interpolating $\theta_\star$, constrained stationarity, a positive
regularizer gap,
full row rank of $A_\star$, suitable definiteness of the Hessians and kernel,
and a valid reference point. The dense implementation evaluates all trained
models, including noninterpolating and failed-regime models, but distinguishes
the resulting `hard_iic_candidate` from a theory-valid `hard_iic`. It never
silently drops a model or replaces an indefinite determinant by a certified
one.

## Soft-IIC

The finite-penalty extension replaces the hard constraint-kernel determinant
by a regularised determinant. Using $\kappa>0$ for the penalty parameter, so
that it is not confused with the reaction coefficient in the PINN below, the
implemented score is

$$
\operatorname{IIC}_{\mathrm{soft}}(\kappa)
=
\log\!\left(R(\theta_\star)-R(\theta_0)\right)
+\frac{
  \log\det\!\left(K_H+\kappa^{-1}I\right)
  +\log\det H_\star
  -\log\det H_0
}{N}
-\log N.
$$

The hard kernel is recovered as $\kappa\rightarrow\infty$ when the limit is
well defined. Finite $\kappa$ regularises weak or null constraint directions
and supports analysis away from exact interpolation, while retaining the same
energy and Hessian-volume terms. Configuration and output fields currently use
the name `finite_penalty_rhos` for these $\kappa$ values; that field name is
legacy terminology and does not denote the PDE reaction coefficient.

## Reaction--diffusion PINNs

The included end-to-end experiment solves the logistic reaction--diffusion
equation

$$
u_t(x,t)=\nu u_{xx}(x,t)+\rho_{\mathrm{PDE}}u(x,t)
\left(1-u(x,t)\right),
\qquad (x,t)\in[0,2\pi)\times[0,1],
$$

or, equivalently, uses the residual

$$
r_\theta(x,t)
=u_{\theta,t}-\nu u_{\theta,xx}
-\rho_{\mathrm{PDE}}u_\theta
+\rho_{\mathrm{PDE}}u_\theta^2.
$$

The initial condition and periodic boundary conditions are

$$
u(x,0)
=\exp\!\left[-\frac{1}{2}
\left(\frac{x-\pi}{\pi/4}\right)^2\right],
\qquad
u(0,t)=u(2\pi,t).
$$

For $\nu>0$, the constraint map additionally enforces
$u_x(0,t)=u_x(2\pi,t)$. This derivative-matching block is omitted when
$\nu=0$, so the $\nu=0$ and $\nu>0$ cases have different constraint
dimensions and are reported as distinct estimands.

### Constraint map

For $n_x$ initial-condition samples and $n_t$ boundary times, the PINN
constraint map concatenates

$$
\begin{aligned}
h_{\mathrm{initial},i}(\theta)
&=\sqrt{\frac{2}{n_x}}\left(u(x_i,0)-u_\theta(x_i,0)\right),\\
h_{\mathrm{periodic},j}(\theta)
&=\sqrt{\frac{2}{n_t}}\left(u_\theta(0,t_j)-u_\theta(2\pi,t_j)\right),\\
h_{\mathrm{derivative},j}(\theta)
&=\sqrt{\frac{2}{n_t}}\left(u_{\theta,x}(0,t_j)
-u_{\theta,x}(2\pi,t_j)\right),\qquad \nu>0.
\end{aligned}
$$

The scaling is chosen so that

$$
\frac{1}{2}\lVert h(\theta)\rVert^2
=L_{\mathrm{initial}}(\theta)+L_{\mathrm{boundary}}(\theta),
$$

where each included block contributes its mean squared residual. The PDE
residual is deliberately not part of $h$: it is an explicit,
data-dependent regularizer. With the example grid sizes $n_x=256$ and
$n_t=100$, this gives $N=356$ for $\nu=0$ and $N=456$ for
$\nu>0$.

### Model and data

The predictor $u_\theta(x,t)$ is a scalar-output fully connected `tanh`
network. The 13-by-13 failure-grid example uses four hidden layers of width 50,
giving 7,851 trainable parameters. Weights in a layer with fan-in $d_\ell$
are sampled from $\mathcal N(0,2/d_\ell)$, and biases are sampled from
$\mathcal N(0,1)$.

The deterministic comparison solution is generated on the periodic grid by
Strang splitting: an exact logistic reaction step and a Fourier diffusion
step. Interior PDE collocation points are sampled without replacement from the
noninitial grid using an independently recorded collocation seed. The dense
grid solution is used only to report relative prediction error; it is not
inserted into the IIC constraint map beyond the initial condition.

### Training objective and regularizer

The training objective is

$$
L_{\mathrm{train}}(\theta)
=\frac{1}{2}\lVert h(\theta)\rVert^2
+R_{\mathrm{PDE}}(\theta)
+R_{\mathrm{wd}}(\theta),
$$

with enabled terms

$$
R_{\mathrm{PDE}}(\theta)
=\lambda_{\mathrm{PDE}}\frac{1}{n_f}
\sum_{k=1}^{n_f}r_\theta(x_k,t_k)^2,
\qquad
R_{\mathrm{wd}}(\theta)
=\frac{\lambda_{\mathrm{wd}}}{2}\lVert\theta\rVert^2.
$$

The full regularizer used by IIC is assembled independently as

$$
R
=R_{\mathrm{init}}+R_{\mathrm{PDE}}+R_{\mathrm{wd}},
$$

with only enabled components included. The initialization regularizer is the
negative log-density, up to additive constants, of the Gaussian distribution
actually used to initialize the network:

$$
R_{\mathrm{init}}(\theta)
=\frac{1}{2}\sum_\ell
\left(\frac{d_\ell}{2}\lVert W_\ell\rVert_F^2
+\lVert b_\ell\rVert_2^2\right).
$$

This initialization term contributes to the IIC regularizer but is not added
to the optimization loss merely because the parameters were sampled from that
distribution. The current failure-grid configuration uses zero weight decay,
so its full regularizer is $R=R_{\mathrm{init}}+R_{\mathrm{PDE}}$.

The same assembled $R$ is used without modification to solve numerically for
$\theta_0$, form the energy gap, and construct $H_\star$ and $H_0$.
The current nonlinear reference solver performs deterministic multistart
gradient descent with Armijo backtracking. It records a stationary candidate
and does not claim to certify a global minimum.

## Numerical backends

The constraint kernel $K_H$ is always materialised as an $N\times N$
matrix and its determinant is evaluated explicitly. Applying
$H_\star^{-1}$ to the Jacobian right-hand sides can use a dense solve, an
explicitly requested Moore--Penrose pseudoinverse, or matrix-free CG. The
dense exact path attempts a Cholesky factorisation first and reuses it for both
the log determinant and kernel solve. If Cholesky fails, the package computes
the full spectrum, records the failure, and retains the existing signed or
pseudodeterminant continuation where possible. It never turns that fallback
into a positive-definite certification. CG stops and records nonpositive
curvature instead of treating an indefinite system as SPD.

The Hessian-volume gap has four backends:

- `exact`: dense Hessians with Cholesky-first exact factors and spectral
  diagnostics on fallback;
- `first_order`: $\operatorname{tr}[H_0^{-1}(H_\star-H_0)]$;
- `path`: quadrature of the log-determinant derivative along the straight
  Hessian path from $H_0$ to $H_\star$;
- `slq`: correlated-probe estimates of
  $\operatorname{tr}\log H_\star-\operatorname{tr}\log H_0$.

The approximate paths use HVPs, Hutchinson probes, CG, and, where selected,
Lanczos quadrature. They report estimator uncertainty and solver health and do
not invent unavailable minimum eigenvalues. Spectral diagnostics deliberately
separate two quantities: `spectral_absolute_floor`, denoted `tau_abs`, controls
the analysis convention for pseudo-rank and pseudodeterminants; the separately
reported numerical-resolution scale is `epsilon_mach * s_max`, where `s_max`
is the largest absolute eigenvalue or singular value. No matrix-dimension
factor is applied, and the roundoff estimate is not folded into `tau_abs`.

`spectral_absolute_floor` defaults to `1e-14`. Roundoff does not silently zero
a spectral direction. Raw eigenvalues, signed log-absolute determinants,
eigenpair residuals, solve residuals, and numerical sign-resolution flags are
retained. The legacy configuration name `tolerance` remains accepted as an
alias. `numerical_jitter` is a separate, default-zero control and never
changes the recorded training weight decay.

Dense Hessian rows can be constructed in bounded-memory chunks with
`hessian_chunk_size`; smaller chunks reduce peak accelerator memory at the
cost of additional runtime.
Regularizer coefficients, including initialization and implicit terms, are
used exactly as configured; the evaluator does not infer a relative scaling or
add an unrecorded ridge.

Exact evaluation retains indefinite and singular cases. Conventional
determinants populate the IIC fields only when their validity conditions hold.
Signed-log-absolute and pseudo-log-determinant continuations are stored under
`diagnostic_continuations`, with matrix flags, and are never labelled
theory-valid IIC values.

## Current implementation

This is a deliberately narrow research implementation:

- exact and matrix-free full-score differentiation for reaction-diffusion
  PINNs;
- deterministic multi-process sharding, phase-separated resumption, and strict
  JSON merge validation, but no cluster-specific scheduler;
- exact, first-order, path-integral, and SLQ Hessian-volume backends;
- no FLODANCE constraint-kernel backend yet;
- nonlinear reference points are numerical multistart candidates, not
  certified global minima.

The complete hard-limit score is the default estimand. `--curvature-only`
provides an explicit ablation that skips $\theta_0$, $H_0$, and the energy
gap.
Numerically complete candidate scores remain available even when stationarity,
interpolation, or global-minimum certification has not been established; each
condition has a separate status flag. Every successfully trained PINN is
evaluated, including expected failure-regime models. Values outside the
hard-limit assumptions are reported as numerical candidates rather than being
silently discarded or promoted to theory-valid hard IIC. Indefinite or
singular matrices are retained as diagnostics but are never silently treated
as valid determinants.

`hard_iic_candidate` and `soft_iic_candidate` retain computable formula values
for descriptive all-model analysis. The unqualified `hard_iic` and `soft_iic`
fields are populated only when their recorded theory-validity conditions hold.

The repository contains no manuscript, paper results, pretrained weights, or
experimental datasets. The included PINN workflow is a computational
demonstration, and no claim is made that its diagnostics predict
generalisation. The API is unstable while the underlying numerical programme
is being developed.

## Install

```bash
python -m pip install -e ".[dev]"
```

The distribution is named `interpolating-iic`; the import and command names are
both `iic`.

## Run

Validate a run without training:

```bash
iic pinn run \
  --config configs/pinn-smoke.json \
  --output runs/pinn-smoke \
  --dry-run
```

Execute the complete smoke workflow:

```bash
iic pinn run \
  --config configs/pinn-smoke.json \
  --output runs/pinn-smoke
```

The smoke is an explicit CPU float64 calibration and also computes the direct
tangent-space IIC. It is intentionally small.

The command validates the configuration, generates deterministic PDE data,
trains each configured model, records the declared regime gate, solves for a
regularizer-reference candidate, computes every dense IIC term, and writes
manifest-complete JSON records. A failed regime gate is a recorded warning and
does not censor checkpoint evaluation.

To compute only the curvature ablation:

```bash
iic pinn run \
  --config configs/pinn-smoke.json \
  --output runs/pinn-smoke-curvature \
  --curvature-only
```

The larger `configs/pinn-pilot.example.json` is an L-BFGS illustrative
protocol, not a frozen paper experiment. It trains in float32 and evaluates
curvature in float64. Its fixed collocation seed is independent of the model
initialization seeds.

`configs/pinn-failure-grid.example.json` expands a decimal 13-by-13
reaction--diffusion grid with five model seeds (845 runs), the four-by-50
network, 1,000 fixed collocation points, L-BFGS training, and
`R_init + R_PDE` evaluation. It is an executable design example, not an
authorization to spend the corresponding compute or a frozen paper protocol.
Its matched He-Gaussian initialization is a deliberate change from the
historical implementation's PyTorch default initialization.

### Shards and resumption

Large plans can be partitioned deterministically without changing the
configuration:

```bash
iic pinn run \
  --config configs/pinn-pilot.example.json \
  --output runs/pinn-pilot-shard-0 \
  --num-shards 4 \
  --shard-index 0
```

`num_shards` is the number of deterministic partitions of the configured run
list. It is not the number of GPUs or concurrent processes. `workers` controls
how many shards execute concurrently. Using 845 shards for the 845-run failure
grid gives one model per shard, which makes calibration runs directly reusable
and gives the finest restart granularity.

Run the remaining zero-based shard indices with the same `--num-shards`.
Training rows and parameter checkpoints are written before dense evaluation.
If evaluation is interrupted, rerun the same command with `--resume`; validated
successful rows are skipped and failed or missing evaluations are retried
without retraining. A resume also requires the current source fingerprint to
match the source that wrote the shard and its parameter checkpoints. The
fingerprint includes the Git revision and, for a dirty checkout, a digest of
the working-tree changes. `--allow-source-mismatch` is an explicit emergency
override for a compatibility decision made outside the package; it should not
be part of a routine cluster command.

For a single full-size timing run on one GPU, shard 420 is the grid member
$(\nu,\rho,\mathrm{seed})=(3,3,0)$:

```bash
iic pinn launch \
  --config configs/pinn-failure-grid.example.json \
  --output runs/pinn-failure-grid \
  --stage training \
  --num-shards 845 \
  --shard-indices 420 \
  --workers 1 \
  --cuda-devices 0

iic pinn launch \
  --config configs/pinn-failure-grid.example.json \
  --output runs/pinn-failure-grid \
  --stage evaluation \
  --resume \
  --num-shards 845 \
  --shard-indices 420 \
  --workers 1 \
  --cuda-devices 0 \
  --hessian-chunk-size 16
```

An eight-GPU calibration uses shards
`0 61 67 128 420 451 782 843`. These are actual failure-grid runs spanning
both constraint dimensions and low, intermediate, and high PDE settings.
Running them in the eventual full-run output directory allows the later
845-shard launch to reuse them.

Training and evaluation can be separated:

```bash
iic pinn run \
  --config configs/pinn-pilot.example.json \
  --output runs/pinn-pilot-shard-0 \
  --num-shards 4 \
  --shard-index 0 \
  --stage training

iic pinn run \
  --config configs/pinn-pilot.example.json \
  --output runs/pinn-pilot-shard-0 \
  --num-shards 4 \
  --shard-index 0 \
  --stage evaluation
```

Every checkpoint, reference, and JSON aggregate is written atomically. On an
eviction-prone machine, point `--output` at durable network-mounted storage;
the package does not assume an SSH endpoint that has not been configured. Each
shard writes `stage_status.json` before and after training and evaluation.
The multi-process launcher streams each worker's standard output and error to
`logs/shard-NNNN/<stage>.stdout.log` and
`logs/shard-NNNN/<stage>.stderr.log` under the launch root, so progress and
failures remain inspectable while a worker is still running.

Inspect resource mapping without training or evaluation:

```bash
iic pinn inventory --config configs/pinn-failure-grid.example.json
```

After calibration, launch every failure-grid shard with `--resume`. Existing
calibration shards are reused; missing shard directories are created normally:

```bash
iic pinn launch \
  --config configs/pinn-failure-grid.example.json \
  --output runs/pinn-failure-grid \
  --stage training \
  --resume \
  --num-shards 845 \
  --workers 8 \
  --cuda-devices 0 1 2 3 4 5 6 7
```

The `cpu` profile keeps autodiff and linear algebra on CPU in float64. The
`mixed` profile performs autodiff/HVP work on GPU and transfers explicit
matrices to CPU float64 linear algebra. The `gpu` profile keeps both phases on
GPU. Worker density is a machine calibration parameter, not a fixed
two-process limit. Evaluation can use the same 845 shards with a lower
`--workers` value without changing shard identity.

After every shard completes:

```bash
iic pinn merge \
  --config configs/pinn-pilot.example.json \
  --inputs \
    runs/pinn-pilot-shard-0 \
    runs/pinn-pilot-shard-1 \
    runs/pinn-pilot-shard-2 \
    runs/pinn-pilot-shard-3 \
  --output runs/pinn-pilot-merged
```

The merge rejects inconsistent configurations, duplicate shards, missing shard
indices, duplicate run identities, and incomplete training or evaluation
coverage.

### Regime-aware analysis

The dependency-light analysis command reports descriptive all-model
correlations separately from interpolating and noninterpolating subsets. It
also reports correlations between PDE-cell medians and median within-cell
Kendall tau-b. The `nu=0` and `nu>0` estimands are always reported separately
because derivative boundary matching changes the constraint dimension:

```bash
iic pinn analyze \
  --input runs/pinn-pilot-merged \
  --scores hard_iic_candidate soft_iic_candidate.10 soft_iic_candidate.100 \
  --target relative_error \
  --output runs/pinn-pilot-analysis.json
```

The all-model table is deliberately labelled as candidate-score analysis.
Hard-limit claims must use the recorded interpolation, constrained
stationarity, reference, rank, and definiteness statuses.

## Safety and reproducibility

- Output directories are non-overwriting by default.
- Training and evaluation device/precision are explicit and independent.
- Model and collocation seeds are explicit and independent.
- Full configuration and runtime provenance are recorded.
- Git revision, dirty-tree digest, package version, CUDA runtime, accelerator
  identity, and source fingerprint are recorded where available.
- Dense-memory estimates are checked before Hessian construction.
- Evaluator and pipeline phase timings, process peak RSS, and resettable CUDA
  allocator peaks are recorded with each run.
- Trained and reference parameter records have independent fingerprints and
  manifests.
- Reference convergence, constrained stationarity, interpolation,
  definiteness, and global certification are reported independently.
- Checkpoints and generated results are ignored by Git.
- Resumed PINN evaluation accepts only the package's non-pickle NPZ parameter
  format after validating parameter names, shapes, data identity, configuration
  identity, source identity, and content fingerprints.

## Literature

- [Interpolation Information Criterion](https://arxiv.org/abs/2307.07785)
- [Characterizing possible failure modes in physics-informed neural networks](https://arxiv.org/abs/2109.01050)

## Licence

Copyright 2026 Chris van der Heide.

This research preview is licensed under the
[PolyForm Noncommercial License 1.0.0](LICENSE). It may be inspected, tested,
modified, and shared only for purposes permitted by that licence. The intended
licence for a later publication-ready release is MIT.
