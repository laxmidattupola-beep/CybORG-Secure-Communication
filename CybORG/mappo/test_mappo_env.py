import torch

from CybORG.mappo.cyborg_env import CybORGMAPPOEnv
from CybORG.mappo.mappo import MAPPO


print("===== MAPPO + CYBORG INTEGRATION TEST =====")

# --------------------------------------------------
# 1. Create environment
# --------------------------------------------------

env = CybORGMAPPOEnv(steps=10)

print("\nEnvironment created")
print("Agents:", env.get_agent_ids())
print("Observation dimension:", env.get_observation_dim())
print("Action dimension:", env.get_action_dim())


# --------------------------------------------------
# 2. Create MAPPO model
# --------------------------------------------------

model = MAPPO(
    observation_dim=env.get_observation_dim(),
    action_dim=env.get_action_dim(),
    num_agents=env.num_agents
)

print("\nMAPPO model created")


# --------------------------------------------------
# 3. Reset environment
# --------------------------------------------------

observations = env.reset()

print("\nEnvironment reset")


# --------------------------------------------------
# 4. Select actions using MAPPO actors
# --------------------------------------------------

actions = {}
log_probs = {}

for i, agent_id in enumerate(env.get_agent_ids()):

    observation = torch.tensor(
        observations[agent_id],
        dtype=torch.float32
    ).unsqueeze(0)

    action, log_probability = model.actors[i].get_action(
        observation
    )

    actions[agent_id] = action.item()
    log_probs[agent_id] = log_probability.item()


print("\nActions selected by MAPPO:")

for agent_id in env.get_agent_ids():
    print(
        f"{agent_id}: "
        f"action={actions[agent_id]}, "
        f"log_prob={log_probs[agent_id]:.4f}"
    )


# --------------------------------------------------
# 5. Send actions to CybORG
# --------------------------------------------------

next_observations, rewards, terminated, info = env.step(
    actions
)


print("\nCybORG step completed")


# --------------------------------------------------
# 6. Display results
# --------------------------------------------------

print("\nRewards:")

for agent_id in env.get_agent_ids():
    print(
        f"{agent_id}: {rewards[agent_id]}"
    )


print("\nTerminated:")

for agent_id in env.get_agent_ids():
    print(
        f"{agent_id}: {terminated[agent_id]}"
    )


print("\nNext observation shapes:")

for agent_id in env.get_agent_ids():
    print(
        f"{agent_id}: "
        f"{next_observations[agent_id].shape}"
    )


# --------------------------------------------------
# 7. Centralized critic test
# --------------------------------------------------

centralized_observations = torch.tensor(
    [
        observations[agent_id]
        for agent_id in env.get_agent_ids()
    ],
    dtype=torch.float32
).unsqueeze(0)


value = model.critic(
    centralized_observations
)


print("\nCentralized critic value:")
print(value.item())


print("\n===== INTEGRATION TEST PASSED =====")