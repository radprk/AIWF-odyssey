# # enhanced_call_center.py - Complete simulation framework
# import random
# import time
# import json
# from datetime import datetime, timedelta
# from typing import Dict, List, Optional
# from dataclasses import dataclass
# from queue import Queue, PriorityQueue
# import numpy as np
# from collections import defaultdict

# # ONET Task Automation Mapping
# ONET_TASKS = {
#     "information_retrieval": {
#         "automation_score": 0.95,
#         "frequency": "daily",
#         "description": "Provide information about products/services",
#         "ai_capability": "excellent"
#     },
#     "record_keeping": {
#         "automation_score": 0.95,
#         "frequency": "daily", 
#         "description": "Keep records of customer interactions",
#         "ai_capability": "excellent"
#     },
#     "basic_problem_solving": {
#         "automation_score": 0.80,
#         "frequency": "daily",
#         "description": "Resolve routine service/billing issues",
#         "ai_capability": "good"
#     },
#     "billing_calculations": {
#         "automation_score": 0.90,
#         "frequency": "daily",
#         "description": "Calculate charges and process payments",
#         "ai_capability": "excellent"
#     },
#     "customer_communication": {
#         "automation_score": 0.75,
#         "frequency": "daily",
#         "description": "Communicate with customers via phone/email",
#         "ai_capability": "good"
#     },
#     "dispute_resolution": {
#         "automation_score": 0.50,
#         "frequency": "weekly",
#         "description": "Handle complex disputes and complaints",
#         "ai_capability": "limited"
#     },
#     "emotional_management": {
#         "automation_score": 0.35,
#         "frequency": "daily",
#         "description": "Handle angry/distressed customers",
#         "ai_capability": "poor"
#     },
#     "sales_solicitation": {
#         "automation_score": 0.60,
#         "frequency": "weekly",
#         "description": "Solicit sales of additional services",
#         "ai_capability": "moderate"
#     }
# }

# # Enhanced Agent Configuration
# AGENT_CONFIG = {
#     "L1": {
#         "model": "llama3.1:8b",
#         "cost_per_1k_tokens": 0.0001,
#         "tokens_per_interaction": 800,
#         "capacity": 10,
#         "hourly_wage": 25,
#         "automation_score": 0.85,
#         "task_capabilities": ["information_retrieval", "record_keeping", "basic_problem_solving", "billing_calculations"]
#     },
#     "L2": {
#         "model": "llama3.1:70b",
#         "cost_per_1k_tokens": 0.0008,
#         "tokens_per_interaction": 1200,
#         "capacity": 5,
#         "hourly_wage": 45,
#         "automation_score": 0.65,
#         "task_capabilities": ["customer_communication", "dispute_resolution", "sales_solicitation"]
#     },
#     "L3": {
#         "model": "claude-3-sonnet",
#         "cost_per_1k_tokens": 0.003,
#         "tokens_per_interaction": 2000,
#         "capacity": 2,
#         "hourly_wage": 80,
#         "automation_score": 0.35,
#         "task_capabilities": ["emotional_management", "complex_dispute_resolution", "escalation_handling"]
#     }
# }

# @dataclass
# class CustomerPersona:
#     persona_type: str
#     patience_level: str  # low, medium, high
#     technical_literacy: str  # basic, intermediate, advanced
#     emotional_state: str  # calm, frustrated, angry
#     issue_complexity: str  # simple, moderate, complex
    
#     def get_response_pattern(self):
#         """Generate response characteristics based on persona"""
#         if self.emotional_state == "angry" and self.patience_level == "low":
#             return {"response_length": "short", "escalation_probability": 0.7}
#         elif self.technical_literacy == "advanced":
#             return {"response_length": "detailed", "escalation_probability": 0.3}
#         else:
#             return {"response_length": "medium", "escalation_probability": 0.4}

# @dataclass
# class CallInteraction:
#     call_id: str
#     customer: CustomerPersona
#     query_type: str
#     start_time: datetime
#     end_time: Optional[datetime] = None
#     agent_tier: Optional[str] = None
#     resolution_status: str = "pending"
#     escalation_count: int = 0
#     cost: float = 0.0
#     customer_satisfaction: float = 0.0
#     onet_tasks_used: List[str] = None
    
#     def __post_init__(self):
#         if self.onet_tasks_used is None:
#             self.onet_tasks_used = []

# class CostCalculator:
#     def __init__(self):
#         self.total_ai_cost = 0.0
#         self.total_human_cost = 0.0
#         self.interaction_costs = []
    
#     def calculate_ai_cost(self, agent_tier: str, duration_minutes: float) -> float:
#         """Calculate AI model cost for interaction"""
#         config = AGENT_CONFIG[agent_tier]
#         tokens_used = config["tokens_per_interaction"]
#         cost = (tokens_used / 1000) * config["cost_per_1k_tokens"]
#         self.total_ai_cost += cost
#         return cost
    
#     def calculate_human_cost(self, agent_tier: str, duration_minutes: float) -> float:
#         """Calculate human agent cost for interaction"""
#         config = AGENT_CONFIG[agent_tier]
#         hourly_wage = config["hourly_wage"]
#         cost = (duration_minutes / 60) * hourly_wage
#         self.total_human_cost += cost
#         return cost
    
#     def get_cost_comparison(self) -> Dict:
#         return {
#             "ai_total": self.total_ai_cost,
#             "human_total": self.total_human_cost,
#             "savings": self.total_human_cost - self.total_ai_cost,
#             "savings_percentage": ((self.total_human_cost - self.total_ai_cost) / self.total_human_cost) * 100
#         }

# class MetricsTracker:
#     def __init__(self):
#         self.interactions = []
#         self.hourly_metrics = defaultdict(lambda: defaultdict(int))
#         self.onet_task_usage = defaultdict(int)
        
#     def track_interaction(self, interaction: CallInteraction):
#         self.interactions.append(interaction)
        
#         # Track hourly metrics
#         hour = interaction.start_time.hour
#         self.hourly_metrics[hour]["total_calls"] += 1
#         if interaction.escalation_count > 0:
#             self.hourly_metrics[hour]["escalations"] += 1
#         if interaction.resolution_status == "resolved":
#             self.hourly_metrics[hour]["resolved"] += 1
            
#         # Track ONET task usage
#         for task in interaction.onet_tasks_used:
#             self.onet_task_usage[task] += 1
    
#     def get_performance_metrics(self) -> Dict:
#         if not self.interactions:
#             return {}
            
#         total_calls = len(self.interactions)
#         resolved_calls = len([i for i in self.interactions if i.resolution_status == "resolved"])
#         escalated_calls = len([i for i in self.interactions if i.escalation_count > 0])
        
#         avg_duration = np.mean([
#             (i.end_time - i.start_time).total_seconds() / 60 
#             for i in self.interactions if i.end_time
#         ])
        
#         avg_satisfaction = np.mean([i.customer_satisfaction for i in self.interactions])
        
#         return {
#             "total_calls": total_calls,
#             "resolution_rate": resolved_calls / total_calls,
#             "escalation_rate": escalated_calls / total_calls,
#             "avg_handle_time_minutes": avg_duration,
#             "avg_customer_satisfaction": avg_satisfaction,
#             "onet_task_distribution": dict(self.onet_task_usage)
#         }

# class CallCenterSimulation:
#     def __init__(self):
#         self.agent_pools = self._initialize_agent_pools()
#         self.customer_queue = PriorityQueue()
#         self.metrics_tracker = MetricsTracker()
#         self.cost_calculator = CostCalculator()
#         self.current_time = datetime.now()
        
#     def _initialize_agent_pools(self) -> Dict:
#         pools = {}
#         for tier, config in AGENT_CONFIG.items():
#             pools[tier] = {
#                 "available": config["capacity"],
#                 "total": config["capacity"],
#                 "busy": 0
#             }
#         return pools
    
#     def generate_customer_persona(self) -> CustomerPersona:
#         """Generate realistic customer persona based on distributions"""
#         return CustomerPersona(
#             persona_type=random.choice(["routine", "technical", "upset", "new_customer"]),
#             patience_level=random.choices(["low", "medium", "high"], weights=[0.3, 0.5, 0.2])[0],
#             technical_literacy=random.choices(["basic", "intermediate", "advanced"], weights=[0.4, 0.4, 0.2])[0],
#             emotional_state=random.choices(["calm", "frustrated", "angry"], weights=[0.6, 0.3, 0.1])[0],
#             issue_complexity=random.choices(["simple", "moderate", "complex"], weights=[0.5, 0.3, 0.2])[0]
#         )
    
#     def determine_query_type(self, customer: CustomerPersona) -> str:
#         """Determine query type based on customer persona"""
#         if customer.issue_complexity == "simple":
#             return random.choice(["billing_inquiry", "account_info", "service_hours"])
#         elif customer.issue_complexity == "moderate":
#             return random.choice(["payment_dispute", "service_issue", "feature_request"])
#         else:
#             return random.choice(["complex_dispute", "technical_problem", "account_closure"])
    
#     def map_query_to_onet_tasks(self, query_type: str, agent_tier: str) -> List[str]:
#         """Map query to relevant ONET tasks"""
#         task_mapping = {
#             "billing_inquiry": ["information_retrieval", "record_keeping", "billing_calculations"],
#             "account_info": ["information_retrieval", "customer_communication"],
#             "payment_dispute": ["dispute_resolution", "record_keeping", "customer_communication"],
#             "service_issue": ["basic_problem_solving", "customer_communication"],
#             "complex_dispute": ["dispute_resolution", "emotional_management", "customer_communication"],
#             "technical_problem": ["basic_problem_solving", "information_retrieval"],
#             "account_closure": ["customer_communication", "record_keeping", "emotional_management"]
#         }
        
#         return task_mapping.get(query_type, ["customer_communication"])
    
#     def simulate_interaction(self, interaction: CallInteraction) -> CallInteraction:
#         """Simulate a customer service interaction"""
#         # Determine initial agent tier based on complexity
#         if interaction.customer.issue_complexity == "simple":
#             agent_tier = "L1"
#         elif interaction.customer.issue_complexity == "moderate":
#             agent_tier = "L2" if random.random() > 0.7 else "L1"
#         else:
#             agent_tier = "L3" if random.random() > 0.5 else "L2"
        
#         # Check agent availability
#         if self.agent_pools[agent_tier]["available"] == 0:
#             # Queue or route to available tier
#             available_tiers = [t for t, pool in self.agent_pools.items() if pool["available"] > 0]
#             if available_tiers:
#                 agent_tier = available_tiers[0]
#             else:
#                 # All agents busy - customer waits or abandons
#                 if interaction.customer.patience_level == "low":
#                     interaction.resolution_status = "abandoned"
#                     return interaction
        
#         # Assign agent
#         interaction.agent_tier = agent_tier
#         self.agent_pools[agent_tier]["available"] -= 1
#         self.agent_pools[agent_tier]["busy"] += 1
        
#         # Map to ONET tasks
#         interaction.onet_tasks_used = self.map_query_to_onet_tasks(
#             interaction.query_type, agent_tier
#         )
        
#         # Simulate interaction duration
#         base_duration = {"L1": 5, "L2": 8, "L3": 15}[agent_tier]
#         complexity_multiplier = {"simple": 0.8, "moderate": 1.0, "complex": 1.5}[interaction.customer.issue_complexity]
#         duration_minutes = base_duration * complexity_multiplier * random.uniform(0.7, 1.3)
        
#         # Calculate costs
#         ai_cost = self.cost_calculator.calculate_ai_cost(agent_tier, duration_minutes)
#         human_cost = self.cost_calculator.calculate_human_cost(agent_tier, duration_minutes)
#         interaction.cost = ai_cost  # Using AI cost for this simulation
        
#         # Determine resolution and satisfaction
#         agent_capability = AGENT_CONFIG[agent_tier]["automation_score"]
#         task_difficulty = sum(1 - ONET_TASKS[task]["automation_score"] for task in interaction.onet_tasks_used)
        
#         resolution_probability = max(0.1, agent_capability - (task_difficulty * 0.1))
        
#         if random.random() < resolution_probability:
#             interaction.resolution_status = "resolved"
#             interaction.customer_satisfaction = random.uniform(0.7, 1.0)
#         else:
#             # Escalation needed
#             interaction.escalation_count += 1
#             if agent_tier == "L1":
#                 interaction = self.simulate_interaction(interaction)  # Recursively escalate
#             else:
#                 interaction.resolution_status = "unresolved"
#                 interaction.customer_satisfaction = random.uniform(0.2, 0.5)
        
#         # Release agent
#         interaction.end_time = interaction.start_time + timedelta(minutes=duration_minutes)
#         self.agent_pools[agent_tier]["available"] += 1
#         self.agent_pools[agent_tier]["busy"] -= 1
        
#         return interaction
    
#     def run_simulation(self, duration_hours: int = 8, calls_per_hour: int = 50) -> Dict:
#         """Run complete call center simulation"""
#         print(f"🚀 Starting {duration_hours}-hour simulation with {calls_per_hour} calls/hour")
        
#         total_calls = duration_hours * calls_per_hour
        
#         for call_num in range(total_calls):
#             # Generate customer and interaction
#             customer = self.generate_customer_persona()
#             query_type = self.determine_query_type(customer)
            
#             interaction = CallInteraction(
#                 call_id=f"call_{call_num:04d}",
#                 customer=customer,
#                 query_type=query_type,
#                 start_time=self.current_time + timedelta(minutes=call_num * (60/calls_per_hour))
#             )
            
#             # Process interaction
#             completed_interaction = self.simulate_interaction(interaction)
#             self.metrics_tracker.track_interaction(completed_interaction)
            
#             # Progress indicator
#             if call_num % 50 == 0:
#                 print(f"📞 Processed {call_num}/{total_calls} calls")
        
#         # Generate final report
#         performance_metrics = self.metrics_tracker.get_performance_metrics()
#         cost_comparison = self.cost_calculator.get_cost_comparison()
        
#         return {
#             "performance": performance_metrics,
#             "costs": cost_comparison,
#             "onet_analysis": self._analyze_onet_automation()
#         }
    
#     def _analyze_onet_automation(self) -> Dict:
#         """Analyze ONET task automation effectiveness"""
#         task_analysis = {}
        
#         for task, usage_count in self.metrics_tracker.onet_task_usage.items():
#             task_info = ONET_TASKS[task]
#             task_analysis[task] = {
#                 "usage_count": usage_count,
#                 "automation_score": task_info["automation_score"],
#                 "frequency": task_info["frequency"],
#                 "ai_capability": task_info["ai_capability"],
#                 "automation_potential": "High" if task_info["automation_score"] > 0.8 else 
#                                       "Medium" if task_info["automation_score"] > 0.5 else "Low"
#             }
        
#         return task_analysis

# # Example usage and testing
# if __name__ == "__main__":
#     # Quick test run
#     simulator = CallCenterSimulation()
#     results = simulator.run_simulation(duration_hours=2, calls_per_hour=30)
    
#     print("\n📊 SIMULATION RESULTS")
#     print("=" * 50)
    
#     print("\n🎯 Performance Metrics:")
#     for metric, value in results["performance"].items():
#         if isinstance(value, float):
#             print(f"  {metric}: {value:.2f}")
#         else:
#             print(f"  {metric}: {value}")
    
#     print("\n💰 Cost Analysis:")
#     for metric, value in results["costs"].items():
#         if isinstance(value, float):
#             print(f"  {metric}: ${value:.2f}")
#         else:
#             print(f"  {metric}: {value}")
    
#     print("\n🔧 ONET Task Automation Analysis:")
#     for task, analysis in results["onet_analysis"].items():
#         print(f"  {task}:")
#         print(f"    Usage: {analysis['usage_count']} times")
#         print(f"    Automation Score: {analysis['automation_score']:.0%}")
#         print(f"    Potential: {analysis['automation_potential']}")