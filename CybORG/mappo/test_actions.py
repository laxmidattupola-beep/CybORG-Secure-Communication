from CybORG.mappo.cyborg_env import CybORGMAPPOEnv


def test_actions():

    print("=" * 60)
    print("             CYBORG ACTION TEST")
    print("=" * 60)

    env = CybORGMAPPOEnv(
        steps=10
    )

    agent_ids = env.get_agent_ids()

    print("Agents:", agent_ids)
    print()

    # Test all 8 MAPPO actions
    for action_index in range(env.get_action_dim()):

        print(
            f"Action {action_index}: "
            f"{env.ACTION_NAMES[action_index]}"
        )

        for agent_id in agent_ids:

            action = env._create_cyborg_action(
                agent_id,
                action_index
            )

            if action is None:

                print(
                    f"  {agent_id}: FAILED"
                )

            else:

                print(
                    f"  {agent_id}: "
                    f"{type(action).__name__}"
                )

        print()

    env.close()

    print("=" * 60)
    print("             ACTION TEST COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    test_actions()