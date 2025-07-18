from mesa import Model
from mesa.time import RandomActivation
from mesa.datacollection import DataCollector
from agents import WorkerAgent
import json
import os

# === Constants ===
PER_AGENT_MONTHLY_COST = 860  # Monthly cost to employ a worker
UBI_COST_PER_AGENT = 500      # Universal Basic Income per agent/month

class CallCenterModel(Model):
    def __init__(self, size="medium", automation_pressure=1.0, enable_augmentation=True, ubi=False,job_guarantee=False, reskilling_subsidy=False, layoff_moratorium=False,             reskilling_rate=0.2,
             escalation_threshold=0.3):
        super().__init__()
        self.schedule = RandomActivation(self)
        self.automation_pressure = automation_pressure
        self.enable_augmentation = enable_augmentation
        self.ubi = ubi
        self.job_guarantee = job_guarantee
        self.reskilling_subsidy = reskilling_subsidy
        self.layoff_moratorium = layoff_moratorium
        self.reskilling_rate = reskilling_rate
        self.escalation_threshold = escalation_threshold
        self.reskilling_cost_per_agent = 100 if not reskilling_subsidy else 50
        self.step_count = 0

        # === Cost & ROI tracking ===
        self.total_cost = 0
        self.total_savings = 0
        self.total_reskilling_cost = 0
        self.roi_threshold = 5.0
        self.robot_tax_rate = 0.2
        self.robot_tax_paid = 0
        self.reskilling_cost_per_agent = 100

        # === Load structure and role-task weights ===
        BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

        with open(os.path.join(BASE_DIR, "data/processed/call_center_structures.json")) as f:
            center_structs = json.load(f)

        with open(os.path.join(BASE_DIR, "data/processed/role_weights.json")) as f:
            role_weights = json.load(f)

        self.num_agents = sum(center_structs[size].values())

        # === Initialize Agents ===
        uid = 0
        for role, count in center_structs[size].items():
            weights = role_weights[role]
            for _ in range(count):
                agent = WorkerAgent(uid, self, role, weights)
                self.schedule.add(agent)
                uid += 1

        # === Track all metrics ===
        self.datacollector = DataCollector(
            model_reporters={
                "Employed": lambda m: sum(1 for a in m.schedule.agents if a.status == "employed"),
                "Automated": lambda m: sum(1 for a in m.schedule.agents if a.status == "automated"),
                "Reskilled": lambda m: sum(1 for a in m.schedule.agents if a.status == "reskilled"),
                "Cost": lambda m: m.total_cost,
                "Savings": lambda m: m.total_savings,
                "ReskillCost": lambda m: m.total_reskilling_cost,
                "ROI": lambda m: (m.total_savings - m.total_reskilling_cost) / (m.total_cost + 1e-6),
                "RobotTax": lambda m: m.robot_tax_paid
            }
        )

    def step(self):
        # === Compute monthly workforce dynamics and cost ===
        for agent in self.schedule.agents:
            if agent.status == "automated":
                self.total_savings += PER_AGENT_MONTHLY_COST
            elif agent.status == "reskilled":
                self.total_reskilling_cost += self.reskilling_cost_per_agent
            elif agent.status == "employed":
                self.total_cost += PER_AGENT_MONTHLY_COST

            if self.ubi:
                self.total_cost += UBI_COST_PER_AGENT

        # === Apply robot tax based on ROI ===
        current_roi = (self.total_savings - self.total_reskilling_cost) / (self.total_cost + 1e-6)
        if current_roi > self.roi_threshold:
            taxable_savings = (current_roi - self.roi_threshold) * self.total_cost
            robot_tax = self.robot_tax_rate * taxable_savings
            self.robot_tax_paid += robot_tax
            self.total_savings -= robot_tax  # alternate: add to cost instead

        self.datacollector.collect(self)
        self.schedule.step()
        self.step_count += 1
