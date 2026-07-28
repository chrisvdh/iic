# IIC

`iic` is an experimental Python package for computing the interpolation
information criterion and its component diagnostics. The public surface
contains a dense exact implementation, a one-command reaction-diffusion PINN
experiment, and lightweight infrastructure for classification trajectories
and adapter subspaces.

For a trained point \(\theta_\star\), an unconstrained regularizer reference
\(\theta_0\), and \(N = \dim h\), the primary hard score is

\[
\operatorname{IIC}
=
\log\!\left(R(\theta_\star)-R(\theta_0)\right)
+ \frac{
  \log\det K_H
  + \log\det H_\star
  - \log\det H_0
}{N}
- \log N,
\]

where

\[
K_H=A_\star H_\star^{-1}A_\star^\mathsf{T},
\qquad
A_\star=Dh(\theta_\star).
\]

The package-facing `EvaluationProblem` holds \(\theta_\star\), \(h\), and
\(R\). The energy gap, constraint-curvature term, Hessian-volume gap, and dataset
correction are stored separately. Sharpness and relative curvature are also
reported as an equivalent decomposition of the geometric contribution.
Finite-penalty scores replace \(K_H\) by
\(K_H+\rho^{-1}I\).

The PINN adapter uses scaled data and periodic-boundary residuals as `h`, so

\[
\frac{1}{2}\lVert h(\theta)\rVert^2
= L_{\mathrm{data}}(\theta) + L_{\mathrm{boundary}}(\theta).
\]

The PDE residual is an explicit data-dependent regularizer. Baseline
configurations use
\(R=R_{\mathrm{init}}+R_{\mathrm{PDE}}\). The optional
backward-error-analysis contribution is formed from the objective actually
optimized by full-batch gradient descent and is evaluated as a separate
optimizer-dependent ablation.

## Status

This is a deliberately narrow research implementation:

- dense exact full-score differentiation for reaction-diffusion PINNs;
- blockwise exact `J H^{-1} J^T` construction for supplied diagonal or
  operator metrics without a dense parameter Hessian or retained full
  Jacobian;
- seeded modular-addition training trajectories, deterministic on the tested
  CPU path;
- generic parameter-subspace helpers and a no-network LoRA stand-in;
- full-channel, simplex-tangent, and frozen top-k classification maps;
- no distributed sweep machinery;
- no FLODANCE or other scalable log-determinant backend yet;
- no real language-model integration yet;
- nonlinear reference points are numerical multistart candidates, not
  certified global minima.

The complete score is the default estimand. `--curvature-only` provides an
explicit ablation that skips \(\theta_0\), \(H_0\), and the energy gap.
Numerically complete candidate scores remain available even when stationarity,
interpolation, or global-minimum certification has not been established; each
condition has a separate status flag. Indefinite or singular matrices are
retained as diagnostics but are never silently treated as valid determinants.

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

The command validates the configuration, generates deterministic PDE data,
trains each configured model, applies the declared training gate, solves for a
regularizer-reference candidate, computes every dense IIC term, and writes
manifest-complete JSON records. This baseline deliberately excludes BEA so the
core energy and geometry computation can be validated independently.

Exercise the higher-order BEA path separately:

```bash
iic pinn run \
  --config configs/pinn-smoke-bea.json \
  --output runs/pinn-smoke-bea
```

The BEA smoke uses the same model, data, training trajectory, and score
settings as the baseline. Its only mathematical difference is the addition of
\[
R_{\mathrm{BEA}}(\theta)
=
\frac{\eta}{4}
\left\|\nabla_\theta L_{\mathrm{train}}(\theta)\right\|^2
\]
to the regularizer used for \(\theta_0\), \(H_\star\), \(H_0\), and the energy
gap. It is a differentiation stress test, not the default scientific
estimand.

To compute only the curvature ablation:

```bash
iic pinn run \
  --config configs/pinn-smoke.json \
  --output runs/pinn-smoke-curvature \
  --curvature-only
```

The larger `configs/pinn-pilot.example.json` is a BEA-free illustrative
protocol, not a frozen paper experiment.

### Grokking trajectories

Validate the seeded modular-addition training plan without training:

```bash
iic grokking train \
  --config configs/grokking-smoke.json \
  --dry-run
```

The corresponding smoke training command is:

```bash
iic grokking train \
  --config configs/grokking-smoke.json
```

The harness uses an explicitly recorded layerwise Gaussian initialization and
AdamW. It writes lightweight evaluation snapshots separately from optional
resume checkpoints. Exact BEA is unavailable for these trajectories because
AdamW does not satisfy the full-batch, zero-momentum gradient-descent
assumptions.

This command trains and records a trajectory only. It does not yet compute a
grokking IIC score.

## Classification kernels

For \(n\) observations and \(m\) retained probability channels, the
full-channel map contains \(mn\) scalar outputs. Thus true/false uses \(2n\)
and four-option multiple choice uses \(4n\). A target-versus-rest partition
retains the target as one channel and aggregates every other class into the
second channel.

Full probability channels have one exact normalization null direction per
observation. The package records this structural nullity and also provides an
orthonormal simplex-tangent representation of dimension \((m-1)n\). That
projection uses all classes and removes only the known normalization modes, so
model-dependent near-singular behaviour remains available for diagnosis.
One-hot probability residuals cannot equal zero at finite logits, so their
metadata marks them as near-interpolation or finite-penalty constraints rather
than silently treating them as exact hard equalities.

Top-k partitions are frozen from a declared reference state before comparing
checkpoints. Their selection rule, reference identifier, reference-logit
fingerprint, target fingerprint where applicable, and selected-index
fingerprint are retained. The full 97-way modular-addition map is retained as a target
representation, but its scalable log determinant still requires a future
FLODANCE integration or controlled approximation.

The operator-kernel API computes \(JH^{-1}J^\mathsf{T}\) for a supplied
diagonal precision or inverse-metric operator. This is a curvature primitive,
not automatically a complete IIC: the regularizer energy gap and compatible
Hessian-volume terms remain separate requirements.

## Safety and reproducibility

- Output directories are non-overwriting by default.
- Device selection is explicit and occurs at runtime.
- Full configuration and runtime provenance are recorded.
- Evaluation snapshots and resumable optimizer checkpoints use distinct,
  schema-versioned formats with identity, model, optimizer, and RNG-state
  fingerprints. Resume checkpoints restore Python, NumPy, Torch CPU, and
  available CUDA RNG streams.
- Dense-memory estimates are checked before Hessian construction.
- Trained and reference parameter records have independent fingerprints and
  manifests.
- Reference convergence, KKT stationarity, interpolation, definiteness, and
  global certification are reported independently.
- Checkpoints and generated results are ignored by Git.
- Checkpoints should be treated as untrusted input; this initial workflow does
  not load arbitrary checkpoint files.

## Literature

- [Interpolation Information Criterion](https://arxiv.org/abs/2307.07785)

## Licence

No open-source licence has been selected yet. Choose and add a licence before
making this repository public.
