from CybORG import CybORG
from CybORG.Simulator.Scenarios.EnterpriseScenarioGenerator import EnterpriseScenarioGenerator
from CybORG.Agents.SimpleAgents.RandomAgent import cc4BlueRandomAgent


print("=" * 60)
print("        CybORG CAGE Challenge 4 Demo")
print("=" * 60)

# 1. Create the Enterprise scenario
scenario = EnterpriseScenarioGenerator(
    blue_agent_class=cc4BlueRandomAgent,
    steps=10
)

print("\n[1] Enterprise scenario created")

# 2. Create CybORG
cyborg = CybORG(scenario)

print("[2] CybORG environment initialized")

# 3. Reset the environment
cyborg.reset()

print("[3] Environment reset successfully")

# 4. Display agents
blue_agents = [a for a in cyborg.agents if a.startswith("blue")]
red_agents = [a for a in cyborg.agents if a.startswith("red")]
green_agents = [a for a in cyborg.agents if a.startswith("green")]

print("\nAgents in simulation:")
print("  Blue :", len(blue_agents))
print("  Red  :", len(red_agents))
print("  Green:", len(green_agents))

# 5. Get Blue Agent 0's initial observation
agent = "blue_agent_0"
result = cyborg.reset(agent)

print("\n[4] Blue Agent 0 initialized")
print("    Observation type:", type(result.observation).__name__)
print("    Action space available:", result.action_space is not None)

# 6. Run the simulation
print("\n[5] Running simulation...")

for step in range(10):

    result = cyborg.step(agent=agent)

    print(
        f"    Step {step + 1}: "
        f"Action={result.action}, "
        f"Reward={result.reward}, "
        f"Done={result.done}"
    )

    if result.done:
        print("    Episode ended.")
        break

print("\n" + "=" * 60)
print("        CybORG simulation completed successfully")
print("=" * 60)