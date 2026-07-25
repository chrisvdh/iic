# IIC

`iic` is an experimental Python package for computing the interpolation
information criterion and its component diagnostics. The initial public
surface contains a dense exact implementation and a one-command
reaction-diffusion PINN experiment.

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

The PDE residual is an explicit data-dependent regularizer. When enabled, the
backward-error-analysis contribution is formed from the objective actually
optimized by full-batch gradient descent.

## Status

This is a deliberately narrow research implementation:

- dense exact differentiation and linear algebra only;
- reaction-diffusion PINNs only;
- no distributed sweep machinery;
- no approximation backends;
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
manifest-complete JSON records.

To compute only the curvature ablation:

```bash
iic pinn run \
  --config configs/pinn-smoke.json \
  --output runs/pinn-smoke-curvature \
  --curvature-only
```

The larger `configs/pinn-pilot.example.json` is an illustrative protocol, not
a frozen paper experiment.

## Safety and reproducibility

- Output directories are non-overwriting by default.
- Device selection is explicit and occurs at runtime.
- Full configuration and runtime provenance are recorded.
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
