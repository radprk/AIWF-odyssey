import random
import matplotlib.pyplot as plt
import numpy as np

# Agent class
class Agent:
    def __init__(self, name, level, agent_type, resolve_prob, time_to_resolve):
        self.name = name
        self.level = level
        self.agent_type = agent_type  # 'digital' or 'human'
        self.resolve_prob = resolve_prob
        self.time_to_resolve = time_to_resolve
        self.queries_handled = 0
        self.total_time_spent = 0

    def attempt_resolve(self):
        self.queries_handled += 1
        self.total_time_spent += self.time_to_resolve
        return random.random() < self.resolve_prob

# Decider agent for triaging
class DeciderAgent:
    def __init__(self, time_to_decide=1.0):
        self.time_to_decide = time_to_decide
        self.total_time_spent = 0
        self.decisions_made = 0

    def decide(self):
        self.total_time_spent += self.time_to_decide
        self.decisions_made += 1

# Router to handle query escalation
class Router:
    def __init__(self, escalation_policy):
        self.escalation_policy = escalation_policy
        self.decider = DeciderAgent()

    def resolve_query(self):
        total_time = 0
        self.decider.decide()
        total_time += self.decider.time_to_decide

        for agent in self.escalation_policy:
            resolved = agent.attempt_resolve()
            total_time += agent.time_to_resolve
            if resolved:
                return total_time
        return total_time  # unresolved after all levels

# Simulation function
def simulate_queries(num_days=30, queries_per_day=100):
    agents = [
        Agent("Digital L1", 1, "digital", 0.6, 2),
        Agent("Digital L2", 2, "digital", 0.4, 4),
        Agent("Digital L3", 3, "digital", 0.3, 6),
        Agent("Human L1", 1, "human", 0.2, 5),
        Agent("Human L2", 2, "human", 0.1, 8),
        Agent("Human L3", 3, "human", 0.05, 10),
        Agent("SME", 4, "human", 0.01, 15),
    ]

    router = Router(agents)
    avg_resolution_time_per_day = []

    for day in range(num_days):
        day_total_time = 0
        for _ in range(queries_per_day):
            time = router.resolve_query()
            day_total_time += time
        avg_time = day_total_time / queries_per_day
        avg_resolution_time_per_day.append(avg_time)
        print(f"Day {day+1}: Avg time = {avg_time:.2f}s")

    return avg_resolution_time_per_day, agents, router.decider

# Run simulation
avg_times, agents, decider = simulate_queries()

# Plotting
plt.figure(figsize=(10, 5))
plt.plot(avg_times, marker='o', label='Avg Resolution Time per Day')
plt.xlabel('Day')
plt.ylabel('Avg Resolution Time (s)')
plt.title('Customer Support Simulation')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# Summary Stats
print("\n--- AGENT STATS ---")
for agent in agents:
    print(f"{agent.name:12s} | Queries: {agent.queries_handled:4d} | Time Spent: {agent.total_time_spent:6.2f}s")

print(f"\n--- DECIDER ---\nDecisions made: {decider.decisions_made} | Time spent: {decider.total_time_spent:.2f}s")
