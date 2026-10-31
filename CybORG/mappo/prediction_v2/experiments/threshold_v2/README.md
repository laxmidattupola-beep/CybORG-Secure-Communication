# Threshold Experiment v2

Threshold v2 is isolated from threshold_v1 and production prediction/MAPPO code. The pinned policy and CybORG seed behavior were validated in Stage 1. The frozen 300-episode split is recorded in `split_manifest_20261101.json`.

## Pinned Policy

The only accepted checkpoint is `CybORG/mappo/mappo_model.pt`, SHA-256:

```text
6fab6f9d6cf79c20d2cce677077dd92eca352d39e7a013e34696656b6eb01170
```

`policy_adapter.py` verifies this hash before deserializing the checkpoint, checks dimensions and all actor/critic tensor shapes against the existing MAPPO architecture, loads with strict state-dict matching, sets evaluation mode, and calls the existing deterministic action-selection method with current action masks. It never falls back to another checkpoint.

## Validation Commands

Run from the repository root in an environment with the existing CybORG/MAPPO dependencies installed:

```powershell
py -3.10 -m CybORG.mappo.prediction_v2.experiments.threshold_v2.validate_policy
py -3.10 -m CybORG.mappo.prediction_v2.experiments.threshold_v2.validate_seeding --master-seed 20261031
```

The policy check loads only the pinned checkpoint, resets one environment, and validates one masked deterministic action selection without stepping the simulator. The seeding preflight uses three fresh environments and reset-only scenarios: two resets with the same derived seed must yield identical scenario and RNG fingerprints; a second distinct seed must yield a different RNG-state fingerprint. It does not require distinct seeds to produce distinct topologies and does not perform full data collection.

## Seeding Contract

`CybORGMAPPOEnv.reset()` does not accept a seed. Before each reset, the collector must call `environment.env.set_seed(episode_seed)` on the underlying CybORG instance immediately before `environment.reset()`. CybORG propagates the resulting RNG to the simulation controller; controller reset supplies it to `EnterpriseScenarioGenerator` and agent objects.

The split uses `SeedSequence(20261101).spawn(300)`. Episode ID equals the child's spawn index. Partitions are fixed to IDs 0-134 training, 135-179 calibration, 180-239 threshold selection, and 240-299 final testing. Every episode records its child seed, spawn key, master seed, and partition. Partition-specific episode files are written below `data/full_300_master_20261101/episodes/<partition>/`.

Threshold_v1 did not record per-episode CybORG simulator seeds, so independence from its unknown simulator RNG draws cannot be proven. The v2 manifest excludes the known threshold_v2 preflight seeds and records this limitation explicitly.

## Split Validation and Collection

Validate the frozen manifest and pinned checkpoint before collection:

```powershell
.\.venv\Scripts\python.exe -m CybORG.mappo.prediction_v2.experiments.threshold_v2.validate_split_manifest
```

The collector refuses to run more than three episodes without the frozen manifest. With the manifest, it rejects conflicting episode count, master seed, run ID, or step count, validates checkpoint SHA-256, and refuses an existing output directory:

```powershell
.\.venv\Scripts\python.exe -m CybORG.mappo.prediction_v2.experiments.threshold_v2.collect_data --split-manifest CybORG\mappo\prediction_v2\experiments\threshold_v2\split_manifest_20261101.json --device cpu
```

Do not run collection until the split validator passes and the final-test data handling protocol is approved. No threshold_v1, production code, existing datasets/models, MAPPO code, XAI, or checkpoints are modified by this experiment.