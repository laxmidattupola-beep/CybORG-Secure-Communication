"""Collect isolated zone-risk episodes under the pinned deterministic MAPPO policy."""
from __future__ import annotations

import argparse
import csv
from collections import deque
import hashlib
import ipaddress
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any

import numpy as np
import torch

from CybORG.mappo.cyborg_env import CybORGMAPPOEnv
from CybORG.mappo.prediction_v2.config import (
    ATTACK_ACTIONS,
    DAMAGE_ACTIONS,
    SCAN_ACTIONS,
    UNMAPPED_HOST_TOKENS,
    PredictionConfig,
    ZONE_TO_AGENT,
)
from CybORG.mappo.prediction_v2.features import feature_names, zone_features
from CybORG.mappo.prediction_v2.labels import (
    RedEvent,
    action_name,
    action_target_host,
    classify_action_event,
    future_window_labels,
    resolve_zone_for_target,
    set_subnet_inventory,
    zone_for_host,
)
from CybORG.mappo.prediction_v2.simulator_adapter import (
    events_from_captures,
    install_action_capture,
    new_session_events,
    true_state_red_sessions,
)

from .policy_adapter import (
    ACTION_DIM,
    AGENT_IDS,
    CHECKPOINT_PATH,
    EXPECTED_CHECKPOINT_SHA256,
    OBSERVATION_DIM,
    PinnedMAPPOPolicy,
)
from .split_manifest import (
    DEFAULT_SPLIT_MANIFEST,
    load_split_manifest,
    validate_manifest_data,
)

EXPERIMENT_DIR = Path(__file__).resolve().parent
DATA_ROOT = EXPERIMENT_DIR / "data"
HORIZONS = (3, 5, 10)
ZONES = tuple(ZONE_TO_AGENT)
FEATURE_COLUMNS = feature_names()
LABEL_COLUMNS = tuple(
    f"{prefix}_{horizon}"
    for horizon in HORIZONS
    for prefix in (
        "attack_risk",
        "compromise_risk",
        "scan_risk",
        "damage_risk",
        "label_valid",
    )
)
DATA_FIELDS = (
    "episode_id",
    "episode_seed",
    "master_seed",
    "partition",
    "checkpoint_sha256",
    "policy_deterministic",
    "episode_complete",
    "step",
    "zone",
    *FEATURE_COLUMNS,
    *LABEL_COLUMNS,
)
EVENT_FIELDS = (
    "episode_id",
    "episode_seed",
    "master_seed",
    "partition",
    "step",
    "zone",
    "category",
    "source",
    "host",
    "success",
)
ACTION_FIELDS = (
    "episode_id",
    "episode_seed",
    "master_seed",
    "partition",
    "checkpoint_sha256",
    "policy_deterministic",
    "action_source",
    "step",
    "agent_id",
    "action_index",
    "action_mask",
    "transition_completed",
)
RED_ACTION_AUDIT_FIELDS = (
    "episode_id",
    "episode_seed",
    "master_seed",
    "partition",
    "step",
    "red_agent",
    "action_class",
    "raw_target_fields",
    "extracted_target",
    "extracted_from",
    "target_type",
    "resolved_zone",
    "classification_outcome",
    "exclusion_reason",
)
SESSION_AUDIT_FIELDS = (
    "episode_id",
    "episode_seed",
    "master_seed",
    "partition",
    "step",
    "hostname",
    "resolved_zone",
    "classification_outcome",
    "exclusion_reason",
)
MANIFEST_FIELDS = (
    "episode_id",
    "episode_seed",
    "master_seed",
    "partition",
    "seed_spawn_key",
    "checkpoint_sha256",
    "policy_deterministic",
    "status",
    "episode_complete",
    "episode_length",
    "scenario_fingerprint",
    "scenario_metadata",
    "red_actions_captured",
    "red_actions_without_mapped_event",
    "red_action_audit_rows",
    "discover_remote_systems_actions",
    "discover_remote_systems_scan_events",
    "new_red_sessions",
    "unmapped_new_red_sessions",
    "session_audit_rows",
    "mapped_red_events",
    "action_mask_refreshes",
    "action_selections",
    "feature_rows",
    "error",
    *(f"valid_label_rows_{horizon}" for horizon in HORIZONS),
    *(f"invalid_label_rows_{horizon}" for horizon in HORIZONS),
    *(f"positive_label_rows_{horizon}" for horizon in HORIZONS),
)


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temp_path = Path(stream.name)
            json.dump(value, stream, indent=2, sort_keys=True, default=str)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_path, path)
    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()


def _atomic_csv(path: Path, fieldnames: tuple[str, ...], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            newline="",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temp_path = Path(stream.name)
            writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="raise")
            writer.writeheader()
            writer.writerows(rows)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_path, path)
    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()


def _sha256_json(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _episode_seeds(master_seed: int, episode_count: int) -> list[tuple[int, tuple[int, ...]]]:
    children = np.random.SeedSequence(master_seed).spawn(episode_count)
    seeds = [
        (
            int(child.generate_state(1, dtype=np.uint32)[0]),
            tuple(int(part) for part in child.spawn_key),
        )
        for child in children
    ]
    values = [seed for seed, _ in seeds]
    if len(set(values)) != len(values):
        raise RuntimeError("SeedSequence generated duplicate episode seeds; refusing collection.")
    return seeds


def _current_action_masks(environment: CybORGMAPPOEnv) -> dict[str, np.ndarray]:
    masks = {
        agent_id: np.asarray(environment.get_action_mask(agent_id), dtype=np.float32).reshape(-1)
        for agent_id in environment.agent_ids
    }
    for agent_id, mask in masks.items():
        if mask.size != ACTION_DIM:
            raise ValueError(
                f"Action mask for {agent_id} has {mask.size} entries; expected {ACTION_DIM}."
            )
        if not np.any(mask > 0):
            raise ValueError(f"Action mask for {agent_id} has no available action.")
    return masks


def _scenario_metadata(environment: CybORGMAPPOEnv) -> dict[str, Any]:
    controller = environment.env.environment_controller
    state = controller.state
    subnets = sorted(
        (
            str(subnet.name),
            str(subnet.cidr),
            sorted(str(host) for host in (subnet.hosts or [])),
        )
        for subnet in state.subnets.values()
    )
    hosts = sorted(str(hostname) for hostname in state.hosts)
    red_agents = sorted(
        (str(agent_id), type(interface.agent).__name__)
        for agent_id, interface in controller.agent_interfaces.items()
        if str(agent_id).lower().startswith("red_agent")
    )
    topology = {"subnets": subnets, "hosts": hosts}
    return {
        "scenario_generator": type(environment.env.scenario_generator).__name__,
        "red_agents": red_agents,
        "host_count": len(hosts),
        "subnet_count": len(subnets),
        "scenario_fingerprint": _sha256_json(topology),
    }


def _event_row(
    episode_id: int,
    episode_seed: int,
    master_seed: int,
    partition: str,
    event,
) -> dict[str, Any]:
    if event.zone not in ZONES:
        raise ValueError(
            f"Red event {event.source!r} resolved to unknown zone {event.zone!r}; "
            "refusing to write a mislabeled event."
        )
    if event.host:
        resolved_zone = resolve_zone_for_target(event.host)
        if resolved_zone != event.zone:
            raise ValueError(
                f"Red event host {event.host!r} resolves to {resolved_zone!r}, "
                f"but its event zone is {event.zone!r}."
            )
    return {
        "episode_id": episode_id,
        "episode_seed": episode_seed,
        "master_seed": master_seed,
        "partition": partition,
        "step": int(event.step),
        "zone": event.zone,
        "category": event.category,
        "source": event.source,
        "host": event.host or "",
        "success": event.success,
    }


def _raw_target_fields(action) -> dict[str, str]:
    if isinstance(action, dict):
        attributes = action
    else:
        attributes = vars(action) if hasattr(action, "__dict__") else {}
    target_tokens = ("host", "target", "ip", "subnet", "address")
    return {
        str(key): "<None>" if value is None else str(value)
        for key, value in attributes.items()
        if any(token in str(key).lower() for token in target_tokens)
    }


def _target_type(target: str | None) -> str:
    if target is None:
        return "missing"
    try:
        ipaddress.ip_address(target)
        return "ip_address"
    except ValueError:
        pass
    try:
        ipaddress.ip_network(target, strict=False)
        return "cidr"
    except ValueError:
        pass
    if zone_for_host(target) is not None or any(
        token in target.lower() for token in UNMAPPED_HOST_TOKENS
    ):
        return "hostname"
    return "other_string"


def _target_subnet_names(target: str | None, controller) -> set[str]:
    if not target:
        return set()
    state = controller.state
    hostname_subnet_map = getattr(state, "hostname_subnet_map", {})
    if target in hostname_subnet_map:
        return {str(hostname_subnet_map[target])}

    try:
        address = ipaddress.ip_address(target)
    except ValueError:
        address = None
    try:
        network = ipaddress.ip_network(target, strict=False)
    except ValueError:
        network = None

    names = set()
    for subnet in state.subnets.values():
        cidr_value = getattr(subnet, "cidr", None)
        if cidr_value is None:
            continue
        try:
            cidr = ipaddress.ip_network(str(cidr_value), strict=False)
        except ValueError:
            continue
        if address is not None and address.version == cidr.version and address in cidr:
            names.add(str(subnet.name))
        elif network is not None and network.version == cidr.version:
            if network.subnet_of(cidr) or cidr.subnet_of(network):
                names.add(str(subnet.name))
    return names


def _exclusion_reason(
    action_class: str,
    target: str | None,
    resolved_zone: str | None,
    subnet_names: set[str],
) -> str:
    if action_class == "Sleep":
        return "intentionally_no_prediction_event"
    if target is None:
        return "target_field_not_extracted_or_missing"
    if resolved_zone is None:
        target_text = target.lower()
        subnet_text = " ".join(subnet_names).lower()
        if any(token in target_text or token in subnet_text for token in UNMAPPED_HOST_TOKENS):
            return "intentionally_out_of_scope_target"
        return "target_zone_unresolved"
    if action_class not in DAMAGE_ACTIONS | ATTACK_ACTIONS | SCAN_ACTIONS:
        return "action_type_not_in_prediction_event_sets"
    return ""


def _event_key(event) -> tuple:
    return (
        int(event.step),
        event.zone,
        event.category,
        event.source,
        event.host,
        event.success,
    )


def _classify_captured_red_actions(step: int, captures: dict, controller):
    """Apply existing labels to raw actions, with a v2-only subnet fallback."""
    events = []
    audit_rows = []
    legacy_events = []

    for red_agent, capture in captures.items():
        action = capture.get("action")
        if action is None:
            continue

        action_class = action_name(action)
        raw_fields = _raw_target_fields(action)
        target = action_target_host(action)
        extracted_from = "existing_action_target_host" if target else "none"

        if target is not None and target.strip().lower() in {"", "none", "null"}:
            target = None
        if action_class == "DiscoverRemoteSystems" and target is None:
            subnet_target = raw_fields.get("subnet")
            if subnet_target and subnet_target != "<None>":
                target = subnet_target
                extracted_from = "DiscoverRemoteSystems.subnet"

        target_kind = _target_type(target)
        resolved_zone = resolve_zone_for_target(target) if target else None
        event = classify_action_event(step, action, target)
        if event is not None:
            event = RedEvent(
                event.step,
                event.zone,
                event.category,
                f"{red_agent}:{event.source}",
                event.host,
                event.success,
            )
            events.append(event)

        legacy_events.extend(events_from_captures(step, {red_agent: capture}))

        outcome = event.category if event is not None else "no_event"
        exclusion_reason = "" if event is not None else _exclusion_reason(
            action_class,
            target,
            resolved_zone,
            _target_subnet_names(target, controller),
        )
        audit_rows.append({
            "step": int(step),
            "red_agent": str(red_agent),
            "action_class": action_class,
            "raw_target_fields": json.dumps(raw_fields, sort_keys=True),
            "extracted_target": target or "",
            "extracted_from": extracted_from,
            "target_type": target_kind,
            "resolved_zone": resolved_zone or "",
            "classification_outcome": outcome,
            "exclusion_reason": exclusion_reason,
        })

    custom_legacy_events = [
        event
        for event in events
        if not event.source.endswith(":DiscoverRemoteSystems")
    ]
    legacy_events = [
        event
        for event in legacy_events
        if not event.source.endswith(":DiscoverRemoteSystems")
    ]
    if sorted(_event_key(event) for event in custom_legacy_events) != sorted(
        _event_key(event) for event in legacy_events
    ):
        raise AssertionError(
            "Threshold-v2 target handling changed an existing non-DiscoverRemoteSystems event."
        )

    deduplicated = []
    seen = set()
    for event in events:
        key = _event_key(event)
        if key in seen:
            raise AssertionError(f"Duplicate Red action event generated: {key}")
        seen.add(key)
        deduplicated.append(event)
    return deduplicated, audit_rows


def _audit_new_sessions(
    step: int,
    previous_red_hosts: set[str],
    current_red_hosts: set[str],
    session_events: list,
    controller,
) -> list[dict[str, Any]]:
    mapped_by_host = {event.host: event for event in session_events}
    rows = []
    for hostname in sorted(set(current_red_hosts) - set(previous_red_hosts)):
        event = mapped_by_host.get(hostname)
        zone = event.zone if event is not None else None
        subnet_names = _target_subnet_names(hostname, controller)
        exclusion_reason = ""
        if event is None:
            lowered = hostname.lower()
            subnet_text = " ".join(subnet_names).lower()
            if any(token in lowered or token in subnet_text for token in UNMAPPED_HOST_TOKENS):
                exclusion_reason = "intentionally_out_of_scope_target"
            else:
                exclusion_reason = "hostname_does_not_match_monitored_zone_tokens"
        rows.append({
            "step": int(step),
            "hostname": hostname,
            "resolved_zone": zone or "",
            "classification_outcome": "compromise" if event is not None else "no_event",
            "exclusion_reason": exclusion_reason,
        })
    return rows


def _collect_episode(
    environment: CybORGMAPPOEnv,
    policy: PinnedMAPPOPolicy,
    episode_id: int,
    episode_seed: int,
    master_seed: int,
    spawn_key: tuple[int, ...],
    partition: str,
    episode_steps: int,
    history_steps: int,
) -> tuple[dict[str, Any], list[dict], list[dict], list[dict], list[dict], list[dict], bool, Exception | None]:
    captures = {}
    features: list[tuple[int, str, np.ndarray]] = []
    events = []
    action_rows = []
    red_action_audit_rows = []
    session_audit_rows = []
    action_mask_refreshes = 0
    action_selections = 0
    completed_steps = 0
    red_actions_captured = 0
    red_actions_without_mapped_event = 0
    discover_remote_systems_actions = 0
    discover_remote_systems_scan_events = 0
    new_red_sessions = 0
    unmapped_new_red_sessions = 0
    scenario = {}
    status = "failed"
    error_text = ""
    interrupted = False
    labels = []

    try:
        # Reseed the underlying CybORG instance immediately before wrapper reset.
        environment.env.set_seed(episode_seed)
        observations = environment.reset()
        set_subnet_inventory(environment.env.environment_controller.state.subnets)
        captures = install_action_capture(environment.env)
        scenario = _scenario_metadata(environment)
        histories = {zone: deque(maxlen=history_steps) for zone in ZONES}
        previous_red_hosts = true_state_red_sessions(environment.env)
        action_masks = _current_action_masks(environment)
        action_mask_refreshes += 1
        terminated = False

        for step in range(episode_steps):
            for zone, agent_id in ZONE_TO_AGENT.items():
                if agent_id not in observations:
                    raise KeyError(f"Observation missing for {agent_id} ({zone}).")
                observation = np.asarray(observations[agent_id], dtype=np.float32).reshape(-1)
                if observation.size != OBSERVATION_DIM:
                    raise ValueError(
                        f"Observation for {agent_id} has {observation.size} values; "
                        f"expected {OBSERVATION_DIM}."
                    )
                vector = zone_features(
                    observation,
                    zone,
                    step,
                    episode_steps,
                    list(histories[zone]),
                    expected_dim=OBSERVATION_DIM,
                )
                if vector.size != len(FEATURE_COLUMNS):
                    raise ValueError(
                        f"Features for {zone} have {vector.size} values; "
                        f"feature_names() defines {len(FEATURE_COLUMNS)}."
                    )
                features.append((step, zone, vector.copy()))
                histories[zone].append(observation.copy())

            # This calls only the pinned MAPPO adapter; there is no random-action path.
            with torch.no_grad():
                actions = policy.select_actions(observations, action_masks)
            action_selections += 1
            for agent_id in AGENT_IDS:
                mask = action_masks[agent_id]
                action_rows.append({
                    "episode_id": episode_id,
                    "episode_seed": episode_seed,
                    "master_seed": master_seed,
                    "partition": partition,
                    "checkpoint_sha256": policy.checkpoint_hash,
                    "policy_deterministic": True,
                    "action_source": "pinned_mappo_policy",
                    "step": step,
                    "agent_id": agent_id,
                    "action_index": int(actions[agent_id]),
                    "action_mask": json.dumps([int(value > 0) for value in mask]),
                    "transition_completed": 0,
                })

            for capture in captures.values():
                capture["action"] = None
            next_observations, _rewards, terminated, _infos = environment.step(actions)
            completed_steps += 1
            for row in action_rows[-len(AGENT_IDS):]:
                row["transition_completed"] = 1

            action_events, action_audit = _classify_captured_red_actions(
                completed_steps,
                captures,
                environment.env.environment_controller,
            )
            for row in action_audit:
                row.update({
                    "episode_id": episode_id,
                    "episode_seed": episode_seed,
                    "master_seed": master_seed,
                    "partition": partition,
                })
            red_action_audit_rows.extend(action_audit)
            captured_now = sum(capture.get("action") is not None for capture in captures.values())
            red_actions_captured += captured_now
            red_actions_without_mapped_event += max(0, captured_now - len(action_events))
            discover_remote_systems_actions += sum(
                row["action_class"] == "DiscoverRemoteSystems" for row in action_audit
            )
            discover_remote_systems_scan_events += sum(
                row["action_class"] == "DiscoverRemoteSystems"
                and row["classification_outcome"] == "scan"
                for row in action_audit
            )
            for event in action_events:
                _event_row(episode_id, episode_seed, master_seed, partition, event)
            events.extend(action_events)

            current_red_hosts = true_state_red_sessions(environment.env)
            new_hosts = set(current_red_hosts) - set(previous_red_hosts)
            session_events = new_session_events(
                completed_steps,
                previous_red_hosts,
                current_red_hosts,
            )
            session_audit = _audit_new_sessions(
                completed_steps,
                previous_red_hosts,
                current_red_hosts,
                session_events,
                environment.env.environment_controller,
            )
            for row in session_audit:
                row.update({
                    "episode_id": episode_id,
                    "episode_seed": episode_seed,
                    "master_seed": master_seed,
                    "partition": partition,
                })
            session_audit_rows.extend(session_audit)
            new_red_sessions += len(new_hosts)
            unmapped_new_red_sessions += max(0, len(new_hosts) - len(session_events))
            for event in session_events:
                _event_row(episode_id, episode_seed, master_seed, partition, event)
            events.extend(session_events)
            previous_red_hosts = current_red_hosts
            observations = next_observations

            # Refresh masks after every environment step, including terminal steps.
            action_masks = _current_action_masks(environment)
            action_mask_refreshes += 1
            if terminated:
                break

        status = "complete"
    except KeyboardInterrupt:
        status = "incomplete"
        interrupted = True
        error_text = "KeyboardInterrupt; retained completed transitions and marked episode incomplete."
    except Exception as error:
        status = "failed"
        error_text = f"{type(error).__name__}: {error}"

    if status == "complete" and action_selections != completed_steps:
        status = "failed"
        error_text = (
            f"Action selection count {action_selections} did not match "
            f"completed transition count {completed_steps}."
        )
    if status == "complete" and action_mask_refreshes != action_selections + 1:
        status = "failed"
        error_text = (
            f"Expected one initial mask and one refresh per transition; "
            f"got {action_mask_refreshes} refreshes for {action_selections} selections."
        )

    complete = status == "complete"
    # Keep only feature decisions whose environment transition completed.
    retained_features = [item for item in features if item[0] < completed_steps]
    if len(retained_features) != completed_steps * len(ZONES):
        if status == "complete":
            status = "failed"
            complete = False
            error_text = "Feature row count did not equal completed steps times five zones."
        retained_features = retained_features[: completed_steps * len(ZONES)]

    labels = future_window_labels(
        events=events,
        episode_steps=completed_steps,
        windows=HORIZONS,
        zones=ZONES,
    )
    expected_label_count = completed_steps * len(ZONES)
    if len(labels) != expected_label_count:
        status = "failed"
        complete = False
        error_text = (
            f"Label row count mismatch: expected {expected_label_count}, got {len(labels)}."
        )

    label_lookup = {(int(row["step"]), row["zone"]): row for row in labels}
    if len(label_lookup) != len(labels) or any(zone not in ZONES for _, zone in label_lookup):
        status = "failed"
        complete = False
        error_text = "Label rows contain duplicate step-zone keys or an unknown zone."

    data_rows = []
    for step, zone, vector in retained_features:
        label = label_lookup.get((step, zone))
        if label is None:
            status = "failed"
            complete = False
            error_text = f"Missing label row for step={step}, zone={zone}."
            continue
        row = {
            "episode_id": episode_id,
            "episode_seed": episode_seed,
            "master_seed": master_seed,
            "partition": partition,
            "checkpoint_sha256": policy.checkpoint_hash,
            "policy_deterministic": True,
            "episode_complete": complete,
            "step": step,
            "zone": zone,
        }
        row.update({name: float(value) for name, value in zip(FEATURE_COLUMNS, vector)})
        row.update({key: int(value) for key, value in label.items() if key not in {"step", "zone"}})
        data_rows.append(row)

    label_counts = {}
    for horizon in HORIZONS:
        valid_labels = [row for row in labels if bool(row[f"label_valid_{horizon}"])]
        label_counts[f"valid_label_rows_{horizon}"] = len(valid_labels)
        label_counts[f"invalid_label_rows_{horizon}"] = len(labels) - len(valid_labels)
        label_counts[f"positive_label_rows_{horizon}"] = sum(
            int(row[f"attack_risk_{horizon}"]) for row in valid_labels
        )

    manifest = {
        "episode_id": episode_id,
        "episode_seed": episode_seed,
        "master_seed": master_seed,
        "partition": partition,
        "seed_spawn_key": json.dumps(spawn_key),
        "checkpoint_sha256": policy.checkpoint_hash,
        "policy_deterministic": True,
        "status": status,
        "episode_complete": complete,
        "episode_length": completed_steps,
        "scenario_fingerprint": scenario.get("scenario_fingerprint", ""),
        "scenario_metadata": json.dumps(scenario, sort_keys=True),
        "red_actions_captured": red_actions_captured,
        "red_actions_without_mapped_event": red_actions_without_mapped_event,
        "red_action_audit_rows": len(red_action_audit_rows),
        "discover_remote_systems_actions": discover_remote_systems_actions,
        "discover_remote_systems_scan_events": discover_remote_systems_scan_events,
        "new_red_sessions": new_red_sessions,
        "unmapped_new_red_sessions": unmapped_new_red_sessions,
        "session_audit_rows": len(session_audit_rows),
        "mapped_red_events": len(events),
        "action_mask_refreshes": action_mask_refreshes,
        "action_selections": action_selections,
        "feature_rows": len(data_rows),
        "error": error_text,
        **label_counts,
    }
    event_rows = [
        _event_row(episode_id, episode_seed, master_seed, partition, event)
        for event in events
    ]
    return (
        manifest,
        data_rows,
        event_rows,
        action_rows,
        red_action_audit_rows,
        session_audit_rows,
        interrupted,
        None,
    )


def _write_manifest(path: Path, rows: list[dict]) -> None:
    _atomic_csv(path, MANIFEST_FIELDS, rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--master-seed", type=int)
    parser.add_argument("--run-id")
    parser.add_argument("--split-manifest", type=Path)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    split_manifest = None
    split_records = None
    if args.split_manifest is not None:
        split_manifest, split_records = load_split_manifest(args.split_manifest)
        split_metadata = split_manifest["metadata"]
        if args.episodes is not None and args.episodes != len(split_records):
            parser.error("--episodes conflicts with the frozen split manifest")
        if args.master_seed is not None and args.master_seed != split_metadata["master_seed"]:
            parser.error("--master-seed conflicts with the frozen split manifest")
        if args.run_id is not None and args.run_id != split_metadata["run_id"]:
            parser.error("--run-id conflicts with the frozen split manifest")
        if args.steps != split_metadata["steps_per_episode"]:
            parser.error("--steps conflicts with the frozen split manifest")
        episodes = len(split_records)
        master_seed = int(split_metadata["master_seed"])
        run_id = str(split_metadata["run_id"])
        seed_records = [
            (
                int(row["child_seed"]),
                tuple(int(part) for part in row["spawn_key"]),
                str(row["partition"]),
            )
            for row in split_records
        ]
    else:
        episodes = args.episodes if args.episodes is not None else 3
        master_seed = args.master_seed if args.master_seed is not None else 20261031
        run_id = args.run_id if args.run_id is not None else "preflight_3"
        if episodes < 1:
            parser.error("--episodes must be at least one")
        if episodes > 3:
            parser.error("Collections over three episodes require --split-manifest")
        seed_records = [
            (seed, spawn_key, "unassigned")
            for seed, spawn_key in _episode_seeds(master_seed, episodes)
        ]

    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", run_id):
        parser.error("run ID must contain only letters, digits, underscores, or hyphens")
    config = PredictionConfig(episode_steps=args.steps).validate()
    if tuple(config.windows) != HORIZONS:
        raise ValueError(f"Expected unchanged prediction horizons {HORIZONS}.")
    if tuple(ZONE_TO_AGENT.values()) != AGENT_IDS:
        raise ValueError("Configured zone-agent order does not match the pinned MAPPO actor order.")

    data_root = DATA_ROOT.resolve()
    run_dir = (data_root / run_id).resolve()
    if data_root not in run_dir.parents:
        raise ValueError(f"Collection output escaped threshold_v2/data: {run_dir}")
    if run_dir.exists():
        raise FileExistsError(
            f"Refusing to overwrite an existing collection run: {run_dir}"
        )

    policy = PinnedMAPPOPolicy.load(args.device)
    if policy.checkpoint_hash.lower() != EXPECTED_CHECKPOINT_SHA256:
        raise RuntimeError("Loaded policy checkpoint hash differs from the pinned hash.")
    if split_manifest is not None:
        split_records = validate_manifest_data(
            split_manifest,
            checkpoint_sha256=policy.checkpoint_hash,
        )
        seed_records = [
            (
                int(row["child_seed"]),
                tuple(int(part) for part in row["spawn_key"]),
                str(row["partition"]),
            )
            for row in split_records
        ]

    split_manifest_sha256 = (
        hashlib.sha256(Path(args.split_manifest).read_bytes()).hexdigest()
        if args.split_manifest is not None
        else None
    )
    manifest_rows = []
    run_metadata = {
        "experiment": "threshold_v2",
        "run_id": run_id,
        "status": "running",
        "episodes_requested": episodes,
        "master_seed": master_seed,
        "seed_generator": "numpy.random.SeedSequence.spawn",
        "episode_seeds": [seed for seed, _, _ in seed_records],
        "episode_spawn_keys": [list(spawn_key) for _, spawn_key, _ in seed_records],
        "episode_partitions": [partition for _, _, partition in seed_records],
        "split_manifest_path": str(Path(args.split_manifest).resolve()) if args.split_manifest else None,
        "split_manifest_sha256": split_manifest_sha256,
        "partition_counts": {
            partition: sum(row_partition == partition for _, _, row_partition in seed_records)
            for partition in ("training", "calibration", "threshold_selection", "final_testing")
        } if split_manifest is not None else {},
        "threshold_v1_simulator_seed_independence": (
            "not verified: threshold_v1 did not record per-episode CybORG seeds"
        ),
        "checkpoint_path": "CybORG/mappo/mappo_model.pt",
        "checkpoint_sha256": policy.checkpoint_hash,
        "policy_deterministic": True,
        "action_selection": "PinnedMAPPOPolicy.select_actions with current action masks",
        "observation_dim": OBSERVATION_DIM,
        "action_dim": ACTION_DIM,
        "agent_ids": list(AGENT_IDS),
        "zones": list(ZONES),
        "horizons": list(HORIZONS),
        "feature_count": len(FEATURE_COLUMNS),
        "feature_names_sha256": _sha256_json(FEATURE_COLUMNS),
        "label_source": "existing prediction_v2.labels.future_window_labels",
        "event_sources": [
            "existing prediction_v2.simulator_adapter.events_from_captures for legacy-compatible action types",
            "threshold_v2.collect_data subnet-aware DiscoverRemoteSystems handling",
            "existing prediction_v2.simulator_adapter.new_session_events",
        ],
        "audit_outputs": [
            "episodes/<partition>/episode_<id>_red_action_audit.csv",
            "episodes/<partition>/episode_<id>_session_audit.csv",
        ],
        "threshold_v1_data_reused": False,
        "threshold_v1_test_ids_used": False,
        "output_files": [],
    }
    environment = CybORGMAPPOEnv(steps=args.steps)
    environment.debug_observations = False
    if tuple(environment.agent_ids) != AGENT_IDS:
        environment.close()
        raise ValueError(
            f"Environment agent order mismatch: expected {AGENT_IDS}, got {tuple(environment.agent_ids)}."
        )
    if environment.observation_dim != OBSERVATION_DIM or environment.action_dim != ACTION_DIM:
        environment.close()
        raise ValueError(
            "Environment dimensions do not match the pinned policy: "
            f"observations={environment.observation_dim}, actions={environment.action_dim}."
        )

    run_dir.mkdir(parents=True, exist_ok=False)
    _atomic_json(run_dir / "metadata.json", run_metadata)

    interrupted = False
    failure = None
    try:
        for episode_id, (episode_seed, spawn_key, partition) in enumerate(seed_records):
            result = _collect_episode(
                environment=environment,
                policy=policy,
                episode_id=episode_id,
                episode_seed=episode_seed,
            master_seed=master_seed,
                spawn_key=spawn_key,
            partition=partition,
                episode_steps=args.steps,
                history_steps=config.history_steps,
            )
            (
                manifest,
                data_rows,
                event_rows,
                action_rows,
                red_action_audit_rows,
                session_audit_rows,
                episode_interrupted,
                error,
            ) = result
            manifest_rows.append(manifest)

            episode_prefix = f"episode_{episode_id:05d}"
            episode_dir = run_dir / "episodes" / partition
            _atomic_csv(episode_dir / f"{episode_prefix}.csv", DATA_FIELDS, data_rows)
            _atomic_csv(episode_dir / f"{episode_prefix}_events.csv", EVENT_FIELDS, event_rows)
            _atomic_csv(episode_dir / f"{episode_prefix}_actions.csv", ACTION_FIELDS, action_rows)
            _atomic_csv(
                episode_dir / f"{episode_prefix}_red_action_audit.csv",
                RED_ACTION_AUDIT_FIELDS,
                red_action_audit_rows,
            )
            _atomic_csv(
                episode_dir / f"{episode_prefix}_session_audit.csv",
                SESSION_AUDIT_FIELDS,
                session_audit_rows,
            )
            _write_manifest(run_dir / "episode_manifest.csv", manifest_rows)

            run_metadata["episodes_completed_or_recorded"] = len(manifest_rows)
            run_metadata["output_files"] = [
                str(path.relative_to(run_dir))
                for path in sorted(run_dir.rglob("*"))
                if path.is_file() and path.name != "metadata.json"
            ]
            _atomic_json(run_dir / "metadata.json", run_metadata)

            print(
                f"episode_id={episode_id} seed={episode_seed} status={manifest['status']} "
                f"steps={manifest['episode_length']} features={manifest['feature_rows']} "
                f"events={manifest['mapped_red_events']} "
                + " ".join(
                    f"valid_h{h}={manifest[f'valid_label_rows_{h}']} "
                    f"positive_h{h}={manifest[f'positive_label_rows_{h}']}"
                    for h in HORIZONS
                )
            )

            if manifest["status"] != "complete":
                if manifest["status"] == "failed":
                    failure = error or manifest["error"] or "Episode validation failed."
                interrupted = episode_interrupted
                break

    except KeyboardInterrupt:
        interrupted = True
        run_metadata["status"] = "interrupted_between_episodes"
    except Exception as error:
        failure = f"{type(error).__name__}: {error}"
    finally:
        environment.close()

    if failure:
        run_metadata["status"] = "failed"
        run_metadata["failure"] = failure
    elif interrupted:
        run_metadata["status"] = "interrupted"
    elif len(manifest_rows) == episodes:
        run_metadata["status"] = "complete"
    else:
        run_metadata["status"] = "incomplete"
    run_metadata["episodes_completed_or_recorded"] = len(manifest_rows)
    _atomic_json(run_dir / "metadata.json", run_metadata)

    print(f"Collection status: {run_metadata['status']}")
    print(f"Run directory: {run_dir}")
    print("Incomplete or failed episodes remain recorded and must be excluded from model fitting.")
    if failure:
        raise RuntimeError(f"Episode collection stopped after recording a failed episode: {failure}")
    if interrupted:
        raise SystemExit(130)


if __name__ == "__main__":
    main()