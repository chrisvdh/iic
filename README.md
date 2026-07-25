# IIC

`iic` is an experimental Python package for computing exact, data-space
curvature diagnostics associated with the interpolation information criterion.
The initial public surface contains a small dense reference implementation and
an end-to-end reaction-diffusion PINN pilot.

The package currently computes

\[
K_H = A_\star H_\star^{-1} A_\star^\mathsf{T},
\qquad
A_\star = D h(\theta_\star),
\]

together with hard and finite-penalty log-determinant diagnostics. The PINN
adapter uses scaled data and periodic-boundary residuals as `h`, so that

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
- no full IIC values unless regularizer reference gaps are supplied.

Outputs from the included workflow are therefore labelled `curvature_only`.
Indefinite or singular matrices are retained as diagnostics but are never
silently certified as valid IIC terms.

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
trains each configured model, applies the declared training gate, computes the
dense curvature diagnostics, and writes manifest-complete JSON records. The
larger `configs/pinn-pilot.example.json` is an illustrative protocol, not a
frozen paper experiment.

## Safety and reproducibility

- Output directories are non-overwriting by default.
- Device selection is explicit and occurs at runtime.
- Full configuration and runtime provenance are recorded.
- Dense-memory estimates are checked before Hessian construction.
- Checkpoints and generated results are ignored by Git.
- Checkpoints should be treated as untrusted input; this initial workflow does
  not load arbitrary checkpoint files.

## Literature

- [Interpolation Information Criterion](https://arxiv.org/abs/2307.07785)

## Licence

No open-source licence has been selected yet. Choose and add a licence before
making this repository public.

