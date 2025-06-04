from mesa import Model, Agent
from mesa.time import RandomActivation
from mesa.datacollection import DataCollector
import random

class WorkerAgent(Agent):
    """Worker agent that decides on reskilling, adapting, or resisting automation."""
    
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.skill_level = random.uniform(0.3, 0.9)
        self.adaptability = random.uniform(0.2, 0.8)
        self.employed = True
        self.well_being = 0.7
        
    def step(self):
        automation_threat = self.model.automation_level
        
        if automation_threat > self.skill_level:
            if random.random() < self.adaptability:
                self.skill_level += 0.1
                self.well_being -= 0.05
            else:
                if random.random() < automation_threat - self.skill_level:
                    self.employed = False
                    self.well_being -= 0.2
        
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
        employed_workers = len([w for w in self.model.schedule.agents 
                              if isinstance(w, WorkerAgent) and w.employed])
        
        if employed_workers > self.model.num_workers * 0.7:
            if self.profit > 0.8:
                self.automation_investment += 0.1
                self.profit -= 0.2
                self.model.automation_level += 0.02
        else:
            self.automation_investment -= 0.05
            self.competitiveness += 0.02
        
        self.profit += self.competitiveness * 0.1 - self.automation_investment * 0.05

class GovernmentAgent(Agent):
    """Government implements policies such as taxation, reskilling subsidies, or regulation."""
    
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.policy_type = "balanced"
        
    def step(self):
        employed_workers = len([w for w in self.model.schedule.agents 
                              if isinstance(w, WorkerAgent) and w.employed])
        employment_rate = employed_workers / self.model.num_workers
        
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

class WorkFutureModel(Model):
    """Main simulation model for AI Work Force Odyssey."""
    
    def __init__(self, num_workers=100, num_corporations=10, automation_level=0.3):
        super().__init__()  # Add this line to properly initialize the Model parent class
        self.num_workers = num_workers
        self.num_corporations = num_corporations
        self.automation_level = automation_level
        self.schedule = RandomActivation(self)
        
        # Create agents
        for i in range(self.num_workers):
            worker = WorkerAgent(i, self)
            self.schedule.add(worker)
        
        for i in range(self.num_corporations):
            corp = CorporationAgent(i + self.num_workers, self)
            self.schedule.add(corp)
        
        # Add government
        gov = GovernmentAgent(self.num_workers + self.num_corporations, self)
        self.schedule.add(gov)
        
        # Data collector
        self.datacollector = DataCollector(
            model_reporters={
                "Employment": lambda m: sum(1 for a in m.schedule.agents 
                                          if isinstance(a, WorkerAgent) and a.employed) / m.num_workers,
                "Average_Skill": lambda m: sum(a.skill_level for a in m.schedule.agents 
                                             if isinstance(a, WorkerAgent)) / m.num_workers,
                "Worker_Wellbeing": lambda m: sum(a.well_being for a in m.schedule.agents 
                                                if isinstance(a, WorkerAgent)) / m.num_workers,
                "Corporate_Profit": lambda m: sum(a.profit for a in m.schedule.agents 
                                                if isinstance(a, CorporationAgent)) / m.num_corporations,
                "Automation_Level": lambda m: m.automation_level
            }
        )
    
    def step(self):
        self.datacollector.collect(self)
        self.schedule.step()