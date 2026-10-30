# CybORG zone-level attack-risk prediction v2

This is a new, isolated package. It does not modify the legacy `prediction_module`, MAPPO implementation, existing datasets, or checkpoints. It predicts future attack risk per blue-owned zone rather than the legacy package's next red action class.

## Current implementation status

Implemented: label specification; five-zone mapping; action/session event adapter; step-level future labels for 3/5/10-step horizons; five-step event merging; zone-local observation/history features; episode-grouped train/validation/test split; base-rate, logistic-regression, random-forest, and histogram-gradient-boosting baselines; metrics and inference API; synthetic unit tests.

Not yet verified: live CybORG true-state/session schema, red action target extraction for every action type, seed replay determinism, exact mission phase transitions, collection across held-out red strategies/topologies, counterfactuals, MAPPO integration, and runtime communication of other-zone risks. The simulator collector is deliberately not executed during this build. Do not report collected model performance until those checks are completed.

## Environment

Run from the repository root in the project's supported Python environment. The existing repository appears to target Python 3.10. Install the extra packages if absent:

```powershell
py -3.10 -m pip install numpy pandas scikit-learn joblib
```

## Safe checks (no simulator execution)

```powershell
py -3.10 -m compileall -q CybORG/mappo/prediction_v2
py -3.10 -m unittest CybORG.mappo.prediction_v2.tests.test_labels CybORG.mappo.prediction_v2.tests.test_features
```

## Collection (simulator execution; run only after approving the simulator verification step)

```powershell
py -3.10 -m CybORG.mappo.prediction_v2.collect_data --episodes 40 --steps 100 --seed 42
```

The collector currently uses the environment's default red strategy. It captures actions from all discovered `red_agent*` interfaces, but strategy/topology holdouts and per-strategy scenario construction remain a required next integration step before the plan's generalisation gate is satisfied. The collector uses random high-level blue actions, not MAPPO, to avoid relying on potentially incompatible legacy checkpoints.

## Train and evaluate

After a dataset has been collected and inspected:

```powershell
py -3.10 -m CybORG.mappo.prediction_v2.train
py -3.10 -m CybORG.mappo.prediction_v2.evaluate
```

Artifacts go into this package's `data/`, `models/`, and `reports/` directories. Existing legacy prediction artifacts remain untouched.

## Data and label safeguards

- Inputs use the individual blue agent's 256-D observation, prior observation summaries, and public step/phase metadata.
- Simulator-derived foothold status is explicitly excluded from model features and is only an evaluation stratum.
- Labels cover future steps `t+1..t+w`, not the current action.
- The primary target combines attack attempts, new red sessions and damage. Scanning is a separate low-severity target.
- Contractor-network hosts are intentionally unmapped rather than silently assigned to HQ. Confirm how this initial foothold should be treated before trusting reported results.
- Event target extraction can be incomplete when a red action object does not expose its target hostname directly. The collector must be validated against actual captured actions before results are accepted.
- Calibrated probabilities and high-quality generalisation are not guaranteed just because a model trains successfully.

## Next required gates

1. Run the approved read-only checksum/label-spec step.
2. Separately approve simulator verification on a copy; verify host/session extraction, action targets, seed replay and phase timing.
3. Expand collection to explicit red-strategy and topology holdouts before any predictor gate is considered passed.
4. Compare all models against a runtime-valid persistence/alert-count baseline, then evaluate lead time and event-level metrics by clean/compromised stratum.
5. Only after the predictor beats baseline on held-out strategies should MAPPO integration begin.
