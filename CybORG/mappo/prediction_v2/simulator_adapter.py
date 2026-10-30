"""Thin, defensive adapter for the project's CybORG wrapper.

This file intentionally fails loudly when the simulator's true-state interface
is not as expected; it never fabricates attack labels.
"""
from __future__ import annotations
from collections import defaultdict
from .labels import action_name, action_target_host, classify_action_event, RedEvent


def get_red_interfaces(cyborg_env):
    controller = getattr(cyborg_env, "environment_controller", None)
    interfaces = getattr(controller, "agent_interfaces", None)
    if not isinstance(interfaces, dict):
        raise RuntimeError("CybORG environment_controller.agent_interfaces is unavailable")
    found = [(name, interface) for name, interface in interfaces.items() if str(name).lower().startswith("red_agent")]
    if not found:
        raise RuntimeError("No red_agent* interfaces found; cannot collect red-action labels")
    return found


def install_action_capture(cyborg_env):
    """Capture every red agent's returned action without altering its choice."""
    captures = {}
    for agent_id, interface in get_red_interfaces(cyborg_env):
        agent = getattr(interface, "agent", None)
        if agent is None or not callable(getattr(agent, "get_action", None)):
            raise RuntimeError(f"Red interface {agent_id} has no callable agent.get_action")
        captures[agent_id] = {"action": None}
        if getattr(agent, "_prediction_v2_capture_installed", False):
            agent._prediction_v2_capture = captures[agent_id]
            continue
        original = agent.get_action
        def wrapped(*args, __original=original, __agent=agent, **kwargs):
            result = __original(*args, **kwargs)
            active = getattr(__agent, "_prediction_v2_capture", None)
            if active is not None:
                active["action"] = result
            return result
        agent.get_action = wrapped
        agent._prediction_v2_capture_installed = True
        agent._prediction_v2_capture = captures[agent_id]
    return captures


def _all_hostnames(cyborg_env):
    controller = cyborg_env.environment_controller
    state = getattr(controller, "state", None)
    hosts = getattr(state, "hosts", None)
    if isinstance(hosts, dict):
        return list(hosts.keys())
    raise RuntimeError("Cannot enumerate simulator true-state hosts via controller.state.hosts")


def true_state_red_sessions(cyborg_env):
    """Return hostnames with at least one red session using simulator true state."""
    hostnames = _all_hostnames(cyborg_env)
    info = {host: {"Sessions": "All", "System info": "Hostname"} for host in hostnames}
    getter = getattr(cyborg_env, "get_true_state", None)
    if not callable(getter):
        raise RuntimeError("CybORG get_true_state(info) is unavailable")
    state = getter(info)
    if not isinstance(state, dict):
        raise RuntimeError(f"Unexpected true-state type: {type(state).__name__}")
    red_hosts = set()
    for host, record in state.items():
        if not isinstance(record, dict):
            continue
        sessions = record.get("Sessions", [])
        if isinstance(sessions, list):
            for session in sessions:
                if isinstance(session, dict) and "red" in str(session.get("agent", "")).lower():
                    red_hosts.add(str(host))
                    break
    return red_hosts


def events_from_captures(step, captures):
    events = []
    for red_id, capture in captures.items():
        action = capture.get("action")
        if action is None:
            continue
        event = classify_action_event(step, action, action_target_host(action))
        if event is not None:
            events.append(RedEvent(event.step, event.zone, event.category,
                                   f"{red_id}:{event.source}", event.host, event.success))
    return events


def new_session_events(step, previous_red_hosts, current_red_hosts):
    """Treat each newly observed red foothold as a compromise event."""
    from .labels import zone_for_host
    events = []
    for host in sorted(set(current_red_hosts) - set(previous_red_hosts)):
        zone = zone_for_host(host)
        if zone:
            events.append(RedEvent(step, zone, "compromise", "new_red_session", host, True))
    return events
