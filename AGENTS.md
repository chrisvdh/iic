# Agent Instructions

This repository is the public-facing implementation of IIC curvature
diagnostics and the reaction--diffusion PINN experiment.

Before changing code:

1. Read `README.md`.
2. Run the focused test suite.
3. Preserve the public/private boundary below.

## Mathematical invariants

- The PINN constraint map contains scaled data and periodic-boundary residuals.
- `0.5 * ||h(theta)||^2` equals the configured data and boundary MSE.
- The PDE residual is an explicit data-dependent regularizer, not a constraint.
- `nu = 0` and `nu > 0` are distinct estimands because derivative matching differs.
- Every successfully trained PINN checkpoint is evaluated. Interpolation,
  stationarity, definiteness, and reference validity are reported as statuses
  and must not be used to silently delete rows.
- Hard-score formula values outside the hard-limit assumptions are numerical
  candidates, not theory-valid hard IIC values. Store them in candidate fields;
  unqualified `hard_iic` and `soft_iic` fields require their recorded
  theory-validity conditions.
- Model-initialization and collocation-data seeds are independent provenance
  fields. Replicate comparisons must not vary both accidentally.
- The regularizer is the sum of all enabled initialization, PDE, and explicit
  components; the same sum defines `theta0`, the energy gap, `Hstar`, and `H0`.
- The full hard score is
  `log(Rstar - R0) + (logdet(KH) + logdet(Hstar) - logdet(H0)) / N - log(N)`.
- `N` is the scalar output dimension of the constraint map.
- `Hstar` is the Hessian of the full regularizer at the evaluation point.
- `H0` is the Hessian of the same regularizer at the reference candidate.
- Constraint multipliers may be used for stationarity diagnostics but must not
  enter either core Hessian.
- Numerical reference candidates and numerical scores must remain distinct
  from globally certified references and theory-valid scores.
- Curvature-only output must never populate a full IIC field.
- Indefinite and singular quantities are retained and flagged, never silently
  converted into certified values.
- A kernel computed for a supplied diagonal or operator metric is a curvature
  primitive. It is not a complete IIC unless the energy gap and compatible
  `Hstar`/`H0` volume terms are also computed.

## Repository boundary

Do not add:

- paper drafts, claims, proof plans, or unpublished results;
- checkpoints, generated results, figures, or machine-specific paths;
- unrelated repository commit hashes or Git history;
- unpublished experiment configurations or paper-specific analysis;
- unvalidated FLODANCE claims, cluster-specific launchers, or private sweep
  configurations. Generic deterministic sharding and strict merge validation
  are permitted.

Do not create branches. Work directly on `main` unless the repository owner
explicitly changes this policy.
