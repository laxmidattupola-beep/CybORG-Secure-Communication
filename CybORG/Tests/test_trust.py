import pytest

from CybORG.trust.trust_manager import TrustManager


def test_default_trust():
    manager = TrustManager()

    assert manager.get_trust("agent_1") == pytest.approx(0.5)


def test_register_agent():
    manager = TrustManager()

    manager.register_agent("agent_1")

    assert "agent_1" in manager
    assert manager.get_trust("agent_1") == pytest.approx(0.5)


def test_register_agent_with_initial_trust():
    manager = TrustManager()

    manager.register_agent(
        "agent_1",
        initial_trust=0.8,
    )

    assert manager.get_trust("agent_1") == pytest.approx(0.8)


def test_register_existing_agent_does_not_reset():
    manager = TrustManager()

    manager.register_agent("agent_1")

    manager.update_trust(
        "agent_1",
        successful_communication=True,
        authentication=True,
        behavior=True,
    )

    trust_before = manager.get_trust("agent_1")

    manager.register_agent(
        "agent_1",
        initial_trust=0.1,
    )

    assert manager.get_trust("agent_1") == pytest.approx(
        trust_before
    )


def test_register_multiple_agents():
    manager = TrustManager()

    manager.register_agents(
        [
            "agent_1",
            "agent_2",
            "agent_3",
        ]
    )

    assert len(manager) == 3

    assert manager.get_trust("agent_1") == pytest.approx(0.5)
    assert manager.get_trust("agent_2") == pytest.approx(0.5)
    assert manager.get_trust("agent_3") == pytest.approx(0.5)


def test_positive_trust_update():
    manager = TrustManager()

    trust = manager.update_trust(
        "agent_1",
        successful_communication=True,
        authentication=True,
        behavior=True,
    )

    assert trust > 0.5
    assert trust <= 1.0


def test_negative_trust_update():
    manager = TrustManager()

    trust = manager.update_trust(
        "agent_1",
        successful_communication=False,
        authentication=False,
        behavior=False,
    )

    assert trust < 0.5
    assert trust >= 0.0


def test_trust_factor_weights():
    manager = TrustManager()

    assert manager.trust_factors["communication"] == pytest.approx(
        0.3
    )

    assert manager.trust_factors["authentication"] == pytest.approx(
        0.3
    )

    assert manager.trust_factors["behavior"] == pytest.approx(
        0.4
    )


def test_weighted_update():
    manager = TrustManager()

    trust = manager.update_trust(
        "agent_1",
        successful_communication=1.0,
        authentication=1.0,
        behavior=0.0,
    )

    expected = (
        0.3 * 1.0
        + 0.3 * 1.0
        + 0.4 * 0.0
    )

    # History contains initial 0.5 and the new weighted score.
    expected_average = (0.5 + expected) / 2

    assert trust == pytest.approx(
        expected_average
    )


def test_history_is_stored():
    manager = TrustManager()

    manager.update_trust(
        "agent_1",
        True,
        True,
        True,
    )

    history = manager.get_history("agent_1")

    assert len(history) == 2


def test_history_averaging():
    manager = TrustManager()

    manager.update_trust(
        "agent_1",
        True,
        True,
        True,
    )

    manager.update_trust(
        "agent_1",
        False,
        False,
        False,
    )

    history = manager.get_history("agent_1")

    expected = sum(history) / len(history)

    assert manager.get_trust("agent_1") == pytest.approx(
        expected
    )


def test_repeated_positive_updates():
    manager = TrustManager()

    for _ in range(10):
        manager.update_trust(
            "agent_1",
            True,
            True,
            True,
        )

    assert manager.get_trust("agent_1") > 0.8


def test_repeated_negative_updates():
    manager = TrustManager()

    for _ in range(10):
        manager.update_trust(
            "agent_1",
            False,
            False,
            False,
        )

    assert manager.get_trust("agent_1") < 0.2


def test_is_trusted():
    manager = TrustManager()

    manager.register_agent(
        "trusted_agent",
        initial_trust=0.8,
    )

    manager.register_agent(
        "untrusted_agent",
        initial_trust=0.2,
    )

    assert manager.is_trusted(
        "trusted_agent"
    )

    assert not manager.is_trusted(
        "untrusted_agent"
    )


def test_custom_trust_threshold():
    manager = TrustManager()

    manager.register_agent(
        "agent_1",
        initial_trust=0.6,
    )

    assert manager.is_trusted(
        "agent_1",
        threshold=0.5,
    )

    assert not manager.is_trusted(
        "agent_1",
        threshold=0.7,
    )


def test_trust_level():
    manager = TrustManager()

    manager.register_agent(
        "agent_1",
        initial_trust=0.2,
    )

    assert manager.get_trust_level(
        "agent_1"
    ) == "untrusted"

    manager.register_agent(
        "agent_2",
        initial_trust=0.4,
    )

    assert manager.get_trust_level(
        "agent_2"
    ) == "low"

    manager.register_agent(
        "agent_3",
        initial_trust=0.6,
    )

    assert manager.get_trust_level(
        "agent_3"
    ) == "moderate"

    manager.register_agent(
        "agent_4",
        initial_trust=0.9,
    )

    assert manager.get_trust_level(
        "agent_4"
    ) == "high"


def test_message_weight():
    manager = TrustManager()

    manager.register_agent(
        "agent_1",
        initial_trust=0.75,
    )

    assert manager.get_message_weight(
        "agent_1"
    ) == pytest.approx(0.75)


def test_decay_high_trust_towards_neutral():
    manager = TrustManager(
        decay_rate=0.1,
    )

    manager.register_agent(
        "agent_1",
        initial_trust=1.0,
    )

    old_trust = manager.get_trust("agent_1")

    new_trust = manager.decay_trust(
        "agent_1"
    )

    assert new_trust < old_trust
    assert new_trust == pytest.approx(0.95)


def test_decay_low_trust_towards_neutral():
    manager = TrustManager(
        decay_rate=0.1,
    )

    manager.register_agent(
        "agent_1",
        initial_trust=0.0,
    )

    old_trust = manager.get_trust("agent_1")

    new_trust = manager.decay_trust(
        "agent_1"
    )

    assert new_trust > old_trust
    assert new_trust == pytest.approx(0.05)


def test_decay_neutral_trust():
    manager = TrustManager(
        decay_rate=0.1,
    )

    manager.register_agent(
        "agent_1",
        initial_trust=0.5,
    )

    new_trust = manager.decay_trust(
        "agent_1"
    )

    assert new_trust == pytest.approx(0.5)


def test_decay_all():
    manager = TrustManager(
        decay_rate=0.1,
    )

    manager.register_agent(
        "agent_1",
        initial_trust=1.0,
    )

    manager.register_agent(
        "agent_2",
        initial_trust=0.0,
    )

    manager.decay_all()

    assert manager.get_trust(
        "agent_1"
    ) == pytest.approx(0.95)

    assert manager.get_trust(
        "agent_2"
    ) == pytest.approx(0.05)


def test_statistics():
    manager = TrustManager()

    manager.update_trust(
        "agent_1",
        True,
        True,
        True,
    )

    statistics = manager.get_statistics(
        "agent_1"
    )

    assert statistics["agent"] == "agent_1"
    assert "trust" in statistics
    assert "trust_level" in statistics
    assert "history_length" in statistics


def test_state_dict():
    manager = TrustManager()

    manager.register_agent(
        "agent_1",
        initial_trust=0.8,
    )

    state = manager.state_dict()

    assert "trust_scores" in state
    assert "trust_history" in state
    assert "agent_1" in state["trust_scores"]


def test_load_state_dict():
    manager = TrustManager()

    manager.register_agent(
        "agent_1",
        initial_trust=0.8,
    )

    manager.update_trust(
        "agent_1",
        True,
        True,
        True,
    )

    state = manager.state_dict()

    new_manager = TrustManager()

    new_manager.load_state_dict(state)

    assert new_manager.get_trust(
        "agent_1"
    ) == pytest.approx(
        manager.get_trust("agent_1")
    )

    assert new_manager.get_history(
        "agent_1"
    ) == pytest.approx(
        manager.get_history("agent_1")
    )


def test_reset():
    manager = TrustManager()

    manager.register_agent("agent_1")
    manager.register_agent("agent_2")

    assert len(manager) == 2

    manager.reset()

    assert len(manager) == 0


def test_invalid_initial_trust():
    manager = TrustManager()

    with pytest.raises(ValueError):
        manager.register_agent(
            "agent_1",
            initial_trust=-0.1,
        )

    with pytest.raises(ValueError):
        manager.register_agent(
            "agent_1",
            initial_trust=1.1,
        )


def test_invalid_decay_rate():
    with pytest.raises(ValueError):
        TrustManager(
            decay_rate=-0.1
        )

    with pytest.raises(ValueError):
        TrustManager(
            decay_rate=1.1
        )


def test_invalid_trust_factor():
    manager = TrustManager()

    with pytest.raises(ValueError):
        manager.update_trust(
            "agent_1",
            successful_communication=2.0,
            authentication=True,
            behavior=True,
        )

    with pytest.raises(ValueError):
        manager.update_trust(
            "agent_1",
            successful_communication=True,
            authentication=-1.0,
            behavior=True,
        )

    with pytest.raises(ValueError):
        manager.update_trust(
            "agent_1",
            successful_communication=True,
            authentication=True,
            behavior=2.0,
        )