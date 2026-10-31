"""Preflight CybORG episode reseeding and same-seed scenario reproducibility."""
from __future__ import annotations

import argparse
import hashlib
import json

import numpy as np

from CybORG.mappo.cyborg_env import CybORGMAPPOEnv


def _digest(value) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _fresh_seeded_reset(seed: int) -> dict[str, str]:
    environment = CybORGMAPPOEnv(steps=100)
    try:
        # This must remain immediately before reset: CybORGMAPPOEnv.reset()
        # forwards the underlying CybORG RNG into SimulationController.reset().
        environment.env.set_seed(seed)
        observations = environment.reset()

        controller = environment.env.environment_controller
        state = controller.state
        topology = {
            "subnets": sorted(
                (
                    str(subnet.name),
                    str(subnet.cidr),
                    tuple(sorted(str(host) for host in subnet.hosts)),
                )
                for subnet in state.subnets.values()
            ),
            "hosts": sorted(str(hostname) for hostname in state.hosts),
            "agents": sorted(str(agent_id) for agent_id in environment.agent_ids),
            "observation_shapes": sorted(
                (str(agent_id), tuple(np.asarray(observation).shape))
                for agent_id, observation in observations.items()
            ),
        }
        rng_state = environment.env.np_random.bit_generator.state
        return {
            "scenario_fingerprint": _digest(topology),
            "rng_state_fingerprint": _digest(rng_state),
        }
    finally:
        environment.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--master-seed", type=int, default=20261031)
    args = parser.parse_args()

    child_sequences = np.random.SeedSequence(args.master_seed).spawn(2)
    seeds = [
        int(sequence.generate_state(1, dtype=np.uint32)[0])
        for sequence in child_sequences
    ]
    if len(set(seeds)) != len(seeds):
        raise RuntimeError("SeedSequence produced duplicate episode seeds; aborting preflight.")

    first = _fresh_seeded_reset(seeds[0])
    repeated = _fresh_seeded_reset(seeds[0])
    independent = _fresh_seeded_reset(seeds[1])

    if first["scenario_fingerprint"] != repeated["scenario_fingerprint"]:
        raise AssertionError(
            "The same CybORG seed produced different scenario fingerprints in fresh environments."
        )
    if first["rng_state_fingerprint"] != repeated["rng_state_fingerprint"]:
        raise AssertionError("The same CybORG seed did not reproduce the RNG state after reset.")
    if first["rng_state_fingerprint"] == independent["rng_state_fingerprint"]:
        raise AssertionError("Distinct episode seeds produced identical post-reset RNG states.")

    print(json.dumps({
        "status": "passed",
        "seed_call_order": "underlying CybORG set_seed(seed) immediately before wrapper reset()",
        "same_seed_scenario_reproduced": True,
        "same_seed_rng_state_reproduced": True,
        "different_seed_rng_state_differs": True,
        "different_seed_topology_uniqueness_required": False,
        "seeds": seeds,
        "scenario_fingerprints": [first["scenario_fingerprint"], independent["scenario_fingerprint"]],
    }, indent=2))


if __name__ == "__main__":
    main()