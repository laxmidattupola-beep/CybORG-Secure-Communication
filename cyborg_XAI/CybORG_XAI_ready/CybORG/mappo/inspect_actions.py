from CybORG.mappo.cyborg_env import CybORGMAPPOEnv


def inspect_actions():

    print("=" * 60)
    print("          CYBORG ACTION SPACE INSPECTION")
    print("=" * 60)

    env = CybORGMAPPOEnv(steps=10)

    agent_id = env.get_agent_ids()[0]

    print("\nAgent:", agent_id)

    action_space = env.env.get_action_space(agent_id)

    print("\nACTION SPACE KEYS:")
    print(action_space.keys())

    print("\n" + "=" * 60)
    print("AVAILABLE ACTION CLASSES")
    print("=" * 60)

    action_options = action_space.get(
        "action",
        {}
    )

    for action_class, enabled in action_options.items():

        print(
            f"\n{action_class.__name__}"
        )

        print(
            "Enabled:",
            enabled
        )

        try:

            import inspect

            signature = inspect.signature(
                action_class
            )

            print(
                "Constructor:",
                signature
            )

        except Exception as error:

            print(
                "Could not inspect constructor:",
                error
            )

    print("\n" + "=" * 60)
    print("PARAMETER SPACES")
    print("=" * 60)

    for key, value in action_space.items():

        if key == "action":
            continue

        print(
            f"\n{key}:"
        )

        if isinstance(value, dict):

            count = 0

            for item, enabled in value.items():

                if enabled:

                    print(
                        "  ",
                        repr(item),
                        "->",
                        enabled
                    )

                    count += 1

                    if count >= 10:
                        print(
                            "   ... more values ..."
                        )
                        break

        else:

            print(
                "  ",
                repr(value)
            )

    env.close()

    print("\n" + "=" * 60)
    print("          INSPECTION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    inspect_actions()