# Promote GJR-GARCH to advisory-primary vol forecaster

**Date:** 2026-06-06
**Status:** building
**Charter tie:** the research→promote gate in `forecast/vol.py` — "a model is only
worth promoting if it beats the incumbent out-of-sample." HAR earned advisory-
primary by beating EWMA (DM p≈0.01). GARCH gap #3 produced a model that now beats
HAR on the same bar.

## Evidence

`walk_forward_eval(include_garch=True)` on SPY (3967 OOS days, one-day-ahead,
QLIKE-primary): **GJR-GARCH wins the six-model race** (mean QLIKE 1.531, lowest),
plain GARCH second (1.567), both ahead of EWMA (1.627), rolling (1.693), HAR
(1.720). **DM(GARCH vs HAR) = −2.004, p = 0.045** → GARCH significantly beats the
HAR incumbent. The leverage term (GJR) is what tips it, exactly as equity-index
asymmetry predicts.

That clears the same gate HAR did. So GJR-GARCH is promoted from shadow to
**advisory-primary** in the live one-day-ahead forecast.

## Change

`compute_vol_forecast` cascade becomes **GJR-GARCH → HAR → EWMA**:

1. GJR-GARCH if there are ≥ the model's min observations of returns;
2. else HAR (today's primary) if it fits;
3. else EWMA.

The fallbacks are byte-identical to today's HAR→EWMA path, so any series too short
for GJR behaves exactly as before — this is a *strictly additive* primary, not a
rewrite. `VolForecast.model` can now read `"gjr"`/`"garch"` in addition to
`"har"`/`"ewma"`. `OOS_SKILL_SPY` updated to the new evidence string.

## Scope / honesty

- **Still advisory.** `compute_vol_forecast` / `live_vol_forecast` feed the analyst
  context, render lines, and logs — **no sizing, tilt, or order path consumes the
  forecast.** Promotion to *sizing* remains a separate, deliberate gate (unchanged).
- Conscious, evidence-gated, user-authorized promotion — not an automatic flip.
- PIT + deterministic + dependency-free (GJR fit is the same hand-rolled QMLE).

## Tests

- `compute_vol_forecast` returns `model="gjr"` on a long GARCH-DGP series (primary
  now GJR), with the forecast still positive/annualized/regime-valid.
- Short series (< GJR min obs) still falls back to HAR/EWMA exactly as before.
- Render + live fail-open paths unchanged.
