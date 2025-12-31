from mesa import Agent
import random

class WorkerAgent(Agent):
    def __init__(self, unique_id, model, role, automation_probs):
        super().__init__(unique_id, model)
        self.role = role
        self.automation_probs = automation_probs
        self.status = "employed"  # or "automated", "reskilled"
        self.adaptability = random.uniform(0.3, 0.9)  # personal reskill score
        self.happiness = 1.0  # proxy for well-being

    def step(self):
        if self.status != "employed":
            return

        if self.model.layoff_moratorium and self.model.step_count < 6:
            self.happiness = min(1.0, self.happiness + 0.01)
            return

        p_auto = self.automation_probs["automatable"] * self.model.automation_pressure
        if random.random() < p_auto:
            if self.adaptability > 0.6 and self.model.enable_augmentation:
                if random.random() < self.model.reskilling_rate:
                    self.status = "reskilled"
                    self.happiness -= 0.1
                else:
                    self.status = "automated"
                    self.happiness -= 0.5
            else:
                self.status = "automated"
                self.happiness -= 0.5
        else:
            self.happiness = min(1.0, self.happiness + 0.01)

