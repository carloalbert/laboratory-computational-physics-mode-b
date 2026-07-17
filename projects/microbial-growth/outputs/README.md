# Outputs

This folder contains the retained results from the microbial-growth analysis.

```text
summary/
```

Aggregate diagnostics for the current nine-sugar log-logistic
endpoint-residual model, together with Glu-focused diagnostics.

```text
model_comparison/<sugar>/loglogistic_endpoint_residual/
```

The current per-sugar fit for all nine environments.

```text
model_comparison/Glu/
```

All retained Glu-specific model fits and comparisons, including the
alpha-scaled time hazard and the lineage-specific `u_l` and `omega2_l`
hierarchies.

The superseded `by_sugar/` results and the pre-endpoint Ace adder-sizer
comparison outputs are intentionally not retained.
