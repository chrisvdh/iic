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
- Baseline smoke and pilot configurations exclude BEA. BEA must be enabled in
  an explicitly named stress test or experimental ablation.
- The regularizer is the sum of all enabled initialization, PDE, explicit, and
  BEA components; the same sum defines `theta0`, the energy gap, `Hstar`, and
  `H0`.
- The full hard score is
  `log(Rstar - R0) + (logdet(KH) + logdet(Hstar) - logdet(H0)) / N - log(N)`.
- `N` is the scalar output dimension of the constraint map.
- The KKT Hessian differentiates one scalar Lagrangian with fixed multipliers.
- `H0` is the unconstrained Hessian of the same regularizer at the reference
  candidate.
- Numerical reference candidates and numerical scores must remain distinct
  from globally certified references and theory-valid scores.
- Curvature-only output must never populate a full IIC field.
- Indefinite and singular quantities are retained and flagged, never silently
  converted into certified values.
- Classification observation counts and scalar channel counts are separate.
  A binary full-channel map has `2 * observations` scalar outputs and an
  `m`-choice full-channel map has `m * observations`.
- Full probability channels retain every class and therefore have one exact
  softmax-normalization null direction per observation. The simplex-tangent
  representation removes only those known structural modes and uses all
  classes.
- Frozen top-k partitions must be selected once and reused across compared
  states. Target-versus-rest maps must force the target class into the
  partition.
- AdamW trajectories are not eligible for exact BEA. Decoupled AdamW weight
  decay must not be described as a gradient-descent objective term.
- A kernel computed for a supplied diagonal or operator metric is a curvature
  primitive. It is not a complete IIC unless the energy gap and compatible
  `Hstar`/`H0` volume terms are also computed.

## Public boundary

Do not add:

- paper drafts, claims, proof plans, or unpublished results;
- checkpoints, generated results, figures, or machine-specific paths;
- private-repository commit hashes or Git history;
- private experiment configurations or unpublished paper-specific analysis;
- real language-model downloads or heavyweight adapter dependencies before a
  public base model and task are selected;
- unvalidated FLODANCE claims or distributed-sweep machinery.

Do not create branches. Work directly on `main` unless the repository owner
explicitly changes this policy.
