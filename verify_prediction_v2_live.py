"""Live, short CybORG verification for prediction_v2; does not collect or save a dataset.
Run from repository root: python verify_prediction_v2_live.py
"""
from __future__ import annotations
import hashlib, json, random, sys, traceback
from pathlib import Path

import numpy as np

from CybORG.mappo.cyborg_env import CybORGMAPPOEnv
from CybORG.mappo.prediction_v2.simulator_adapter import (
    install_action_capture, true_state_red_sessions, events_from_captures,
)
from CybORG.mappo.prediction_v2.labels import action_target_host
from CybORG.mappo.prediction_v2.config import ZONE_TO_AGENT
from CybORG.Simulator.Scenarios.EnterpriseScenarioGenerator import EnterpriseScenarioGenerator

SEED = 731
STEPS = 12
PROBE_STEPS = 5

def digest_observations(obs):
    h = hashlib.sha256()
    for agent_id in sorted(obs):
        h.update(agent_id.encode())
        arr = np.asarray(obs[agent_id], dtype=np.float32)
        h.update(arr.tobytes())
    return h.hexdigest()

def run_replay():
    random.seed(SEED)
    np.random.seed(SEED)
    wrapper = CybORGMAPPOEnv(steps=STEPS)
    wrapper.env.debug_observations = False
    report = {"agent_ids": list(wrapper.agent_ids), "observation_shapes": {}, "step_digests": [],
              "red_interface_count": 0, "captured_action_count": 0, "targeted_action_count": 0,
              "classified_event_count": 0, "true_state_session_reads": 0, "session_hosts_seen": [],
              "terminated_after_steps": None}
    try:
        obs = wrapper.reset()
        report["observation_shapes"] = {k: list(np.asarray(v).shape) for k, v in obs.items()}
        report["step_digests"].append(digest_observations(obs))
        captures = install_action_capture(wrapper.env)
        report["red_interface_count"] = len(captures)
        sessions = true_state_red_sessions(wrapper.env)
        report["true_state_session_reads"] += 1
        report["session_hosts_seen"] = sorted(sessions)
        for i in range(PROBE_STEPS):
            for capture in captures.values():
                capture["action"] = None
            # Sleep action for blue agents minimizes unrelated blue-side intervention.
            next_obs, _rewards, terminated, _infos = wrapper.step({aid: 7 for aid in wrapper.agent_ids})
            actions = [c.get("action") for c in captures.values() if c.get("action") is not None]
            report["captured_action_count"] += len(actions)
            report["targeted_action_count"] += sum(action_target_host(a) is not None for a in actions)
            report["classified_event_count"] += len(events_from_captures(i + 1, captures))
            sessions = true_state_red_sessions(wrapper.env)
            report["true_state_session_reads"] += 1
            report["session_hosts_seen"] = sorted(set(report["session_hosts_seen"]) | set(sessions))
            report["step_digests"].append(digest_observations(next_obs))
            if terminated:
                report["terminated_after_steps"] = i + 1
                break
        return report
    finally:
        wrapper.close()

def main():
    results = {"python": sys.version.split()[0], "requested_seed": SEED, "configured_episode_steps": STEPS,
               "probe_steps": PROBE_STEPS, "checks": {}}
    # Source-level mission phase arithmetic, exercised through the scenario generator method.
    try:
        gen = EnterpriseScenarioGenerator(steps=STEPS)
        phases = list(gen._generate_mission_phases(STEPS))
        results["mission_phase_lengths"] = phases
        results["checks"]["phase_lengths_sum_to_episode"] = sum(phases) == STEPS
    except Exception as exc:
        results["checks"]["phase_lengths_sum_to_episode"] = False
        results["phase_error"] = f"{type(exc).__name__}: {exc}"

    try:
        first = run_replay()
        second = run_replay()
        results["first_run"] = first
        results["second_run"] = second
        results["checks"]["five_blue_zones_present"] = all(a in first["agent_ids"] for a in ZONE_TO_AGENT.values())
        results["checks"]["observation_shapes_valid"] = all(shape == [256] for shape in first["observation_shapes"].values())
        results["checks"]["red_interfaces_found"] = first["red_interface_count"] > 0
        results["checks"]["true_state_read_succeeded"] = first["true_state_session_reads"] >= 2
        results["checks"]["seed_replay_same_observations"] = first["step_digests"] == second["step_digests"]
        results["checks"]["actions_captured"] = first["captured_action_count"] > 0
        # This is diagnostic, not a hard failure: some short probes may not produce targeted attacks.
        results["checks"]["at_least_one_target_extracted"] = first["targeted_action_count"] > 0
    except Exception as exc:
        results["runtime_error"] = f"{type(exc).__name__}: {exc}"
        results["traceback"] = traceback.format_exc()
        results["checks"]["runtime_probe_completed"] = False

    print("\n=== PREDICTION V2 LIVE SIMULATOR VERIFICATION ===")
    print(json.dumps(results, indent=2, default=str))
    hard_checks = ["phase_lengths_sum_to_episode", "five_blue_zones_present", "observation_shapes_valid",
                   "red_interfaces_found", "true_state_read_succeeded", "seed_replay_same_observations", "actions_captured"]
    passed = all(results.get("checks", {}).get(k, False) for k in hard_checks)
    print(f"\nRESULT: {'PASS' if passed else 'FAIL / INVESTIGATE'} (hard checks: {sum(bool(results.get('checks', {}).get(k, False)) for k in hard_checks)}/{len(hard_checks)})")
    print("NOTE: Target extraction is reported diagnostically because a short probe may not trigger a targeted red action.")
    print("No dataset/model/checkpoint is created by this script.")
    sys.exit(0 if passed else 1)

if __name__ == "__main__":
    main()
