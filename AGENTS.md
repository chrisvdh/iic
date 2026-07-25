# Agent Instructions

This repository is the small public-facing implementation of IIC curvature
diagnostics. It is intentionally separate from private paper development.

Before changing code:

1. Read `README.md`.
2. Run the focused test suite.
3. Preserve the public/private boundary below.

## Mathematical invariants

- The PINN constraint map contains scaled data and periodic-boundary residuals.
- `0.5 * ||h(theta)||^2` equals the configured data and boundary MSE.
- The PDE residual is an explicit data-dependent regularizer, not a constraint.
- `nu = 0` and `nu > 0` are distinct estimands because derivative matching differs.
- Exact BEA is available only for full-batch, zero-momentum gradient descent and
  uses coefficient `learning_rate / 4`.
- The KKT Hessian differentiates one scalar Lagrangian with fixed multipliers.
- Curvature-only output must never populate a full IIC field.
- Indefinite and singular quantities are retained and flagged, never silently
  converted into certified values.

## Public boundary

Do not add:

- paper drafts, claims, proof plans, or unpublished results;
- checkpoints, generated results, figures, or machine-specific paths;
- private-repository commit hashes or Git history;
- grokking, language-model, approximation, or distributed-sweep code before a
  concrete public requirement exists.

Do not create branches. Work directly on `main` unless the repository owner
explicitly changes this policy.

