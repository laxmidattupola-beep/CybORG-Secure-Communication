import numpy as np

from CybORG.mappo.cyborg_env import CybORGMAPPOEnv


def count_host_records(raw_observation):
    """Count host records in a raw CybORG observation."""
    if not isinstance(raw_observation, dict):
        return 0

    excluded_keys = {
        "success",
        "action",
        "message",
        "msg",
    }

    return sum(
        1
        for key, value in raw_observation.items()
        if key not in excluded_keys and isinstance(value, dict)
    )


def main():
    print("=" * 60)
    print("CYBORG OBSERVATION DIAGNOSTIC TEST")
    print("=" * 60)

    # Create the existing environment.
    env = CybORGMAPPOEnv(steps=20)

    # Reset the environment.
    observations = env.reset()

    print("\nEnvironment reset successful.")
    print("Blue agents:", env.agent_ids)
    print("Observation dimension:", env.observation_dim)
    print("Action dimension:", env.action_dim)

    rng = np.random.default_rng(42)

    # Run five environment steps.
    for step in range(5):
        actions = {}

        # Select a valid random action for each agent.
        for agent_id in env.agent_ids:
            action_mask = env.get_action_mask(agent_id)
            valid_actions = np.flatnonzero(action_mask)

            if len(valid_actions) > 0:
                actions[agent_id] = int(
                    rng.choice(valid_actions)
                )
            else:
                actions[agent_id] = 7

        # Execute one environment step.
        observations, rewards, done, infos = env.step(actions)

        print("\n" + "=" * 60)
        print(f"STEP {step + 1}")
        print("=" * 60)

        for agent_id in env.agent_ids:
            encoded_observation = observations.get(agent_id)

            print(f"\nAgent: {agent_id}")
            print("Selected action:", actions[agent_id])

            # Check the encoded observation returned by the wrapper.
            if encoded_observation is not None:
                encoded_observation = np.asarray(
                    encoded_observation
                ).reshape(-1)

                active_features = np.count_nonzero(
                    encoded_observation
                )

                print(
                    "Encoded observation size:",
                    encoded_observation.size
                )

                print(
                    "Active encoded features:",
                    active_features
                )
            else:
                print("Encoded observation: MISSING")

            # Check whether the agent has an action in progress.
            try:
                in_progress = env.is_action_in_progress(
                    agent_id
                )
                print("Action in progress:", in_progress)

            except Exception as error:
                print("Action status unavailable:", error)

            # Separately inspect the environment's current
            # observation. This may differ from the observation
            # returned by parallel_step().
            try:
                raw_observation = env.env.get_observation(
                    agent_id
                )

                print(
                    "Current raw observation keys:",
                    list(raw_observation.keys())
                    if isinstance(raw_observation, dict)
                    else type(raw_observation).__name__
                )

                print(
                    "Current raw host records:",
                    count_host_records(raw_observation)
                )

            except Exception as error:
                print(
                    "Raw observation inspection failed:",
                    error
                )

        print("\nRewards:", rewards)
        print("Episode finished:", done)

    print("\n" + "=" * 60)
    print("DIAGNOSTIC TEST COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    main()