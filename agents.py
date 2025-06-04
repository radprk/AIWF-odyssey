 
from mesa import Agent
import random

class WorkerAgent(Agent):
    """Worker agent that decides on reskilling, adapting, or resisting automation."""
    
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.skill_level = random.uniform(0.3, 0.9)  # Initial skill level
        self.adaptability = random.uniform(0.2, 0.8)  # How willing to reskill
        self.employed = True
        self.well_being = 0.7
        
    def step(self):
        # Decision to reskill based on automation threat
        automation_threat = self.model.automation_level
        
        if automation_threat > self.skill_level:
            if random.random() < self.adaptability:
                # Reskill
                self.skill_level += 0.1
                self.well_being -= 0.05  # Short-term stress
            else:
                # Risk of unemployment
                if random.random() < automation_threat - self.skill_level:
                    self.employed = False
                    self.well_being -= 0.2
        
        # Update well-being based on employment
        if self.employed:
            self.well_being = min(1.0, self.well_being + 0.02)
        else:
            self.well_being = max(0.1, self.well_being - 0.05)

class CorporationAgent(Agent):
    """Corporation that chooses between automation, augmentation, or human-centric strategies."""
    
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.automation_investment = 0.5
        self.competitiveness = 0.7
        self.profit = 1.0
        
    def step(self):
        # Decide on strategy based on market conditions
        employed_workers = len([w for w in self.model.schedule.agents 
                              if isinstance(w, WorkerAgent) and w.employed])
        
        if employed_workers > self.model.num_workers * 0.7:
            # Invest in automation
            if self.profit > 0.8:
                self.automation_investment += 0.1
                self.profit -= 0.2
                self.model.automation_level += 0.02
        else:
            # Invest in augmentation
            self.automation_investment -= 0.05
            self.competitiveness += 0.02
        
        # Update profit based on competitiveness
        self.profit += self.competitiveness * 0.1 - self.automation_investment * 0.05

class GovernmentAgent(Agent):
    """Government implements policies such as taxation, reskilling subsidies, or regulation."""
    
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.policy_type = "balanced"  # balanced, pro-worker, pro-business
        
    def step(self):
        # Monitor unemployment
        employed_workers = len([w for w in self.model.schedule.agents 
                              if isinstance(w, WorkerAgent) and w.employed])
        employment_rate = employed_workers / self.model.num_workers
        
        # Implement policies based on conditions
        if employment_rate < 0.7:
            self.policy_type = "pro-worker"
            self.implement_reskilling_program()
        elif employment_rate > 0.9:
            self.policy_type = "pro-business"
        else:
            self.policy_type = "balanced"
    
    def implement_reskilling_program(self):
        for agent in self.model.schedule.agents:
            if isinstance(agent, WorkerAgent) and not agent.employed:
                agent.skill_level += 0.1
                agent.adaptability += 0.05