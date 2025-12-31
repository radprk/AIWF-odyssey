import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mesa import Agent, Model
from mesa.time import RandomActivation
from mesa.datacollection import DataCollector
import random
from enum import Enum
from dataclasses import dataclass
from typing import List, Dict, Optional
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Configure Streamlit
st.set_page_config(page_title="AI Workforce Policy Simulator", layout="wide")

# ==========================================
# Data Structures and Enums
# ==========================================

class QueryComplexity(Enum):
    LOW = "low"
    MEDIUM = "medium"  
    HIGH = "high"
    EXPERT = "expert"

class AgentType(Enum):
    ROUTER = "router"
    DIGITAL = "digital"
    HUMAN = "human"

class QueryStatus(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    RESOLVED = "resolved"
    ESCALATED = "escalated"
    FAILED = "failed"

@dataclass
class Query:
    id: int
    complexity: QueryComplexity
    creation_time: float
    total_time: float = 0.0
    cost: float = 0.0
    status: QueryStatus = QueryStatus.PENDING
    resolution_path: List[str] = None
    
    def __post_init__(self):
        if self.resolution_path is None:
            self.resolution_path = []

# ==========================================
# Agent Classes
# ==========================================

class RouterAgent(Agent):
    """Intelligent routing agent that assigns queries to appropriate agents"""
    
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.agent_type = AgentType.ROUTER
        self.queries_routed = 0
        self.routing_time = 0.5  # minutes per routing decision
        
    def route_query(self, query: Query) -> Optional[Agent]:
        """Route query to most appropriate agent based on complexity and availability"""
        self.queries_routed += 1
        query.total_time += self.routing_time
        query.cost += self.model.router_cost_per_minute * self.routing_time
        query.resolution_path.append(f"Router-{self.unique_id}")
        
        # Get available agents by complexity matching
        suitable_agents = self.get_suitable_agents(query.complexity)
        
        if not suitable_agents:
            return None
            
        # Prefer digital agents first (cost optimization), then by load
        suitable_agents.sort(key=lambda a: (
            0 if a.agent_type == AgentType.DIGITAL else 1,  # Digital first
            a.current_load,  # Then by load
            -a.skill_level   # Then by skill level (higher better)
        ))
        
        return suitable_agents[0] if suitable_agents else None
    
    def get_suitable_agents(self, complexity: QueryComplexity) -> List[Agent]:
        """Get agents suitable for handling given complexity"""
        suitable = []
        min_skill_needed = {
            QueryComplexity.LOW: 1,
            QueryComplexity.MEDIUM: 2, 
            QueryComplexity.HIGH: 3,
            QueryComplexity.EXPERT: 4
        }
        
        for agent in self.model.schedule.agents:
            if (agent.agent_type in [AgentType.DIGITAL, AgentType.HUMAN] and
                agent.skill_level >= min_skill_needed[complexity] and
                agent.is_available() and
                not agent.is_laid_off):
                suitable.append(agent)
                
        return suitable

class WorkerAgent(Agent):
    """Base class for digital and human agents"""
    
    def __init__(self, unique_id, model, agent_type: AgentType, skill_level: int, 
                 base_resolve_prob: float, avg_time_per_query: float, cost_per_minute: float):
        super().__init__(unique_id, model)
        self.agent_type = agent_type
        self.skill_level = skill_level
        self.base_resolve_prob = base_resolve_prob
        self.avg_time_per_query = avg_time_per_query
        self.cost_per_minute = cost_per_minute
        
        # State variables
        self.current_load = 0.0
        self.max_load = 100.0
        self.queries_handled = 0
        self.queries_resolved = 0
        self.total_time_worked = 0.0
        self.total_cost_incurred = 0.0
        self.is_laid_off = False
        self.skill_investment = 0.0  # Upskilling investment received
        
        # Performance modifiers
        self.fatigue_factor = 1.0
        self.skill_multiplier = 1.0
        
    def is_available(self) -> bool:
        """Check if agent is available to take new queries"""
        return self.current_load < self.max_load and not self.is_laid_off
    
    def attempt_resolve(self, query: Query) -> bool:
        """Attempt to resolve a query"""
        if self.is_laid_off:
            return False
            
        self.queries_handled += 1
        
        # Calculate actual resolution probability based on various factors
        complexity_modifier = self.get_complexity_modifier(query.complexity)
        load_modifier = max(0.5, 1.0 - (self.current_load / self.max_load) * 0.3)
        skill_modifier = self.skill_multiplier
        
        actual_resolve_prob = (self.base_resolve_prob * 
                             complexity_modifier * 
                             load_modifier * 
                             skill_modifier)
        
        # Time calculation with variation
        time_variation = np.random.normal(1.0, 0.2)
        actual_time = self.avg_time_per_query * time_variation * (2.0 - load_modifier)
        
        # Cost calculation (includes robot tax if applicable)
        base_cost = actual_time * self.cost_per_minute
        robot_tax = (base_cost * self.model.robot_tax_rate 
                    if self.agent_type == AgentType.DIGITAL else 0)
        total_cost = base_cost + robot_tax
        
        # Update query
        query.total_time += actual_time
        query.cost += total_cost
        query.resolution_path.append(f"{self.agent_type.value}-L{self.skill_level}-{self.unique_id}")
        
        # Update agent state
        self.current_load += actual_time * 0.5  # Load accumulation
        self.total_time_worked += actual_time
        self.total_cost_incurred += total_cost
        
        # Check resolution
        resolved = np.random.random() < actual_resolve_prob
        if resolved:
            self.queries_resolved += 1
            query.status = QueryStatus.RESOLVED
            
        return resolved
    
    def get_complexity_modifier(self, complexity: QueryComplexity) -> float:
        """Get performance modifier based on query complexity vs agent skill"""
        complexity_to_skill = {
            QueryComplexity.LOW: 1,
            QueryComplexity.MEDIUM: 2,
            QueryComplexity.HIGH: 3,
            QueryComplexity.EXPERT: 4
        }
        
        skill_gap = self.skill_level - complexity_to_skill[complexity]
        if skill_gap >= 0:
            return min(1.2, 1.0 + skill_gap * 0.1)  # Bonus for over-qualification
        else:
            return max(0.3, 1.0 + skill_gap * 0.2)  # Penalty for under-qualification
    
    def daily_recovery(self):
        """Daily load recovery and maintenance"""
        self.current_load = max(0, self.current_load - 20)  # Daily recovery
        self.fatigue_factor = max(0.8, 1.0 - (self.current_load / self.max_load) * 0.2)
    
    def receive_upskilling(self, investment: float):
        """Receive upskilling investment"""
        self.skill_investment += investment
        # Improve skill multiplier based on investment
        self.skill_multiplier = min(1.5, 1.0 + self.skill_investment / 1000)

class DigitalAgent(WorkerAgent):
    """AI/Digital agent with specific characteristics"""
    
    def __init__(self, unique_id, model, skill_level: int):
        # Digital agents are faster but less creative
        base_times = {1: 3, 2: 5, 3: 7, 4: 10}
        base_probs = {1: 0.7, 2: 0.6, 3: 0.5, 4: 0.4}
        base_costs = {1: 0.10, 2: 0.15, 3: 0.20, 4: 0.30}  # per minute
        
        super().__init__(
            unique_id, model, AgentType.DIGITAL, skill_level,
            base_probs[skill_level], base_times[skill_level], base_costs[skill_level]
        )
        self.max_load = 150.0  # Digital agents can handle more load

class HumanAgent(WorkerAgent):
    """Human agent with specific characteristics"""
    
    def __init__(self, unique_id, model, skill_level: int):
        # Human agents are slower but more creative/flexible
        base_times = {1: 10, 2: 15, 3: 20, 4: 25}
        base_probs = {1: 0.8, 2: 0.75, 3: 0.70, 4: 0.65}
        base_costs = {1: 0.50, 2: 0.75, 3: 1.00, 4: 1.50}  # per minute
        
        super().__init__(
            unique_id, model, AgentType.HUMAN, skill_level,
            base_probs[skill_level], base_times[skill_level], base_costs[skill_level]
        )
        self.max_load = 100.0  # Human agents have standard load capacity
        self.ubi_received = 0.0  # Track UBI received

# ==========================================
# Mesa Model
# ==========================================

class CustomerSupportModel(Model):
    """Enhanced Mesa model with holistic policy system"""
    
    def __init__(self, config: Dict):
        super().__init__()
        
        # Configuration
        self.config = config
        self.current_step = 0
        
        # Policy parameters
        self.robot_tax_rate = config.get('robot_tax_rate', 0.0)
        self.ubi_amount = config.get('ubi_amount', 0.0)
        self.upskilling_budget = config.get('upskilling_budget', 0.0)
        self.layoff_rate = config.get('layoff_rate', 0.0)
        
        # Economic parameters
        self.router_cost_per_minute = 0.05
        self.queries_per_day = config.get('queries_per_day', 100)
        self.revenue_per_resolved_query = config.get('revenue_per_resolved_query', 10.0)
        
        # NEW: Economic state tracking
        self.government_revenue = 0.0  # From robot taxes
        self.total_ubi_paid = 0.0
        self.total_upskilling_spent = 0.0
        self.automation_pressure = 0.0  # Economic pressure to automate
        self.worker_satisfaction = 1.0  # Affects productivity and retention
        self.market_competitiveness = 1.0  # Affects revenue
        
        # NEW: Policy effectiveness tracking
        self.layoff_resistance = 0.0  # From UBI and upskilling
        self.digital_capability_boost = 0.0  # From robot tax making humans more competitive
        self.human_skill_multiplier = 1.0  # From upskilling investments
        
        # Query complexity distribution
        self.complexity_distribution = config.get('complexity_distribution', {
            QueryComplexity.LOW: 0.4,
            QueryComplexity.MEDIUM: 0.35,
            QueryComplexity.HIGH: 0.20,
            QueryComplexity.EXPERT: 0.05
        })
        
        # Enhanced data collection
        self.schedule = RandomActivation(self)
        self.datacollector = DataCollector(
            model_reporters={
                # Core metrics
                "Total_Queries_Resolved": self.get_total_queries_resolved,
                "Total_Cost": self.get_total_cost,
                "Total_Revenue": self.get_total_revenue,
                "Average_Resolution_Time": self.get_average_resolution_time,
                "Digital_Agent_Count": self.get_digital_agent_count,
                "Human_Agent_Count": self.get_human_agent_count,
                "Unemployment_Rate": self.get_unemployment_rate,
                "Customer_Satisfaction": self.get_customer_satisfaction,
                
                # NEW: Policy impact metrics
                "Government_Revenue": lambda m: m.government_revenue,
                "Total_UBI_Paid": lambda m: m.total_ubi_paid,
                "Worker_Satisfaction": lambda m: m.worker_satisfaction,
                "Market_Competitiveness": lambda m: m.market_competitiveness,
                "Automation_Pressure": lambda m: m.automation_pressure,
                "Human_Skill_Level": lambda m: m.human_skill_multiplier,
                "Policy_Effectiveness": self.get_policy_effectiveness,
                "Social_Welfare_Index": self.get_social_welfare_index,
            }
        )
        
        # Initialize agents and queries
        self.queries = []
        self.daily_queries = []
        self.create_agents()
    
    def create_agents(self):
        """Create initial agent population"""
        agent_id = 0
        
        # Create router agents
        for _ in range(self.config.get('router_count', 2)):
            router = RouterAgent(agent_id, self)
            self.schedule.add(router)
            agent_id += 1
        
        # Create digital agents
        digital_config = self.config.get('digital_agents', {})
        for level in range(1, 5):
            count = digital_config.get(f'level_{level}_count', 0)
            for _ in range(count):
                agent = DigitalAgent(agent_id, self, level)
                self.schedule.add(agent)
                agent_id += 1
        
        # Create human agents  
        human_config = self.config.get('human_agents', {})
        for level in range(1, 5):
            count = human_config.get(f'level_{level}_count', 0)
            for _ in range(count):
                agent = HumanAgent(agent_id, self, level)
                self.schedule.add(agent)
                agent_id += 1
    
    def generate_daily_queries(self):
        """Generate queries with market dynamics"""
        # Adjust query volume based on market competitiveness
        adjusted_queries = int(self.queries_per_day * self.market_competitiveness)
        
        daily_queries = []
        for i in range(adjusted_queries):
            # Determine complexity based on distribution
            rand = np.random.random()
            cumsum = 0
            complexity = QueryComplexity.LOW
            for comp, prob in self.complexity_distribution.items():
                cumsum += prob
                if rand <= cumsum:
                    complexity = comp
                    break
            
            query = Query(
                id=len(self.queries) + i,
                complexity=complexity,
                creation_time=self.current_step
            )
            daily_queries.append(query)
        
        self.daily_queries = daily_queries
        self.queries.extend(daily_queries)
    
    def process_queries(self):
        """Process queries with enhanced worker performance"""
        router_agents = [a for a in self.schedule.agents if a.agent_type == AgentType.ROUTER]
        
        for query in self.daily_queries:
            if not router_agents:
                query.status = QueryStatus.FAILED
                continue
                
            # Assign to a router (round-robin)
            router = router_agents[query.id % len(router_agents)]
            
            # Attempt resolution through routing system
            max_escalations = 3
            escalations = 0
            
            while escalations < max_escalations and query.status == QueryStatus.PENDING:
                assigned_agent = router.route_query(query)
                
                if assigned_agent is None:
                    query.status = QueryStatus.FAILED
                    break
                
                if assigned_agent.attempt_resolve(query):
                    break  # Successfully resolved
                else:
                    escalations += 1
                    # Query remains pending for next escalation
            
            if query.status == QueryStatus.PENDING:
                query.status = QueryStatus.FAILED
    
    def apply_holistic_policies(self):
        """Apply all policy interventions holistically with realistic interactions"""
        
        # Get current workforce state
        human_agents = [a for a in self.schedule.agents 
                       if a.agent_type == AgentType.HUMAN and not getattr(a, 'is_laid_off', False)]
        digital_agents = [a for a in self.schedule.agents if a.agent_type == AgentType.DIGITAL]
        total_human_agents = [a for a in self.schedule.agents if a.agent_type == AgentType.HUMAN]
        
        print(f"\n=== Day {self.current_step} Policy Analysis ===")
        print(f"Active humans: {len(human_agents)}, Digital: {len(digital_agents)}")
        
        # 1. CALCULATE ECONOMIC PRESSURES
        self.calculate_economic_pressures(human_agents, digital_agents)
        
        # 2. APPLY ROBOT TAX (affects digital agent costs and government revenue)
        robot_tax_revenue = self.apply_robot_tax(digital_agents)
        
        # 3. APPLY UPSKILLING (improves human competitiveness)
        self.apply_upskilling_programs(human_agents)
        
        # 4. APPLY UBI (reduces layoff pressure and improves worker satisfaction)
        ubi_cost = self.apply_ubi_programs(total_human_agents)
        
        # 5. MAKE LAYOFF DECISIONS (considering all factors)
        self.make_layoff_decisions(human_agents)
        
        # 6. UPDATE MARKET DYNAMICS
        self.update_market_dynamics()
        
        # 7. GOVERNMENT BUDGET BALANCE
        self.balance_government_budget(robot_tax_revenue, ubi_cost)
        
        print(f"Government revenue: ${self.government_revenue:.0f}")
        print(f"Worker satisfaction: {self.worker_satisfaction:.2f}")
        print(f"Market competitiveness: {self.market_competitiveness:.2f}")
        print("=" * 50)
    
    def calculate_economic_pressures(self, human_agents, digital_agents):
        """Calculate economic pressures driving automation"""
        if not human_agents:
            self.automation_pressure = 1.0
            return
            
        # Cost pressure: digital vs human cost efficiency
        avg_human_cost = np.mean([a.cost_per_minute for a in human_agents])
        avg_digital_cost = np.mean([a.cost_per_minute for a in digital_agents]) if digital_agents else 0.1
        
        cost_pressure = min(2.0, avg_human_cost / (avg_digital_cost * (1 + self.robot_tax_rate)))
        
        # Performance pressure: digital vs human success rates
        human_performance = np.mean([a.queries_resolved / max(1, a.queries_handled) for a in human_agents])
        digital_performance = np.mean([a.queries_resolved / max(1, a.queries_handled) for a in digital_agents]) if digital_agents else 0.5
        
        performance_pressure = digital_performance / max(0.1, human_performance)
        
        # Market pressure: customer satisfaction and competitiveness
        market_pressure = 2.0 - self.get_customer_satisfaction()
        
        self.automation_pressure = (cost_pressure + performance_pressure + market_pressure) / 3.0
        
        print(f"Automation pressure: {self.automation_pressure:.2f} (cost: {cost_pressure:.2f}, perf: {performance_pressure:.2f}, market: {market_pressure:.2f})")
    
    def apply_robot_tax(self, digital_agents):
        """Apply robot tax and calculate government revenue"""
        daily_robot_tax = 0.0
        
        if self.robot_tax_rate > 0:
            for agent in digital_agents:
                # Tax based on agent's daily productivity
                daily_productivity_value = agent.queries_handled * self.revenue_per_resolved_query * 0.1
                tax_amount = daily_productivity_value * self.robot_tax_rate
                daily_robot_tax += tax_amount
            
            self.government_revenue += daily_robot_tax
            
            # Robot tax makes digital agents less attractive, reducing automation pressure
            tax_effect = self.robot_tax_rate * 2.0  # Amplify impact
            self.automation_pressure = max(0.1, self.automation_pressure - tax_effect)
            
            print(f"Robot tax collected: ${daily_robot_tax:.2f} (rate: {self.robot_tax_rate:.1%})")
        
        return daily_robot_tax
    
    def apply_upskilling_programs(self, human_agents):
        """Apply upskilling with realistic learning curves and effectiveness"""
        if self.upskilling_budget > 0 and human_agents:
            investment_per_agent = self.upskilling_budget / len(human_agents)
            self.total_upskilling_spent += self.upskilling_budget
            
            # Upskilling effectiveness diminishes with scale but improves over time
            learning_curve_bonus = min(0.5, self.current_step * 0.01)  # Improves over time
            effectiveness = (investment_per_agent / 100.0) * (1.0 + learning_curve_bonus)
            
            for agent in human_agents:
                agent.receive_upskilling(investment_per_agent)
                
                # Upskilling also improves job satisfaction and reduces layoff likelihood
                if hasattr(agent, 'job_security'):
                    agent.job_security += effectiveness * 0.1
                else:
                    agent.job_security = 1.0 + effectiveness * 0.1
            
            # Global skill improvement
            skill_improvement = effectiveness * 0.1
            self.human_skill_multiplier = min(2.0, self.human_skill_multiplier + skill_improvement)
            
            # Upskilling reduces automation pressure by making humans more competitive
            automation_reduction = effectiveness * 0.2
            self.automation_pressure = max(0.1, self.automation_pressure - automation_reduction)
            
            print(f"Upskilling investment: ${self.upskilling_budget:.0f} (${investment_per_agent:.2f}/agent)")
            print(f"Human skill multiplier: {self.human_skill_multiplier:.2f}")
    
    def apply_ubi_programs(self, total_human_agents):
        """Apply UBI with realistic social and economic effects"""
        total_ubi_cost = 0.0
        
        if self.ubi_amount > 0:
            for agent in total_human_agents:
                if not hasattr(agent, 'ubi_received'):
                    agent.ubi_received = 0.0
                
                # UBI amount may be higher for unemployed workers
                ubi_multiplier = 2.0 if getattr(agent, 'is_laid_off', False) else 1.0
                daily_ubi = self.ubi_amount * ubi_multiplier
                
                agent.ubi_received += daily_ubi
                total_ubi_cost += daily_ubi
                
                # UBI improves worker satisfaction and reduces layoff pressure
                if hasattr(agent, 'financial_security'):
                    agent.financial_security += daily_ubi * 0.01
                else:
                    agent.financial_security = 1.0 + daily_ubi * 0.01
            
            self.total_ubi_paid += total_ubi_cost
            
            # UBI effects on economy
            # Positive: Reduces social pressure, improves worker satisfaction
            ubi_satisfaction_boost = min(0.3, self.ubi_amount * 0.01)
            self.worker_satisfaction = min(1.5, self.worker_satisfaction + ubi_satisfaction_boost)
            
            # UBI reduces layoff pressure by providing safety net
            layoff_pressure_reduction = self.ubi_amount * 0.005
            self.layoff_resistance += layoff_pressure_reduction
            
            print(f"UBI distributed: ${total_ubi_cost:.0f} to {len(total_human_agents)} agents")
        
        return total_ubi_cost
    
    def make_layoff_decisions(self, human_agents):
        """Make layoff decisions considering all economic and policy factors"""
        if self.layoff_rate <= 0 or len(human_agents) <= 1:
            return
        
        # Calculate base layoff pressure from economic conditions
        base_layoff_pressure = self.automation_pressure * self.layoff_rate
        
        # Adjust for policy interventions
        # UBI and upskilling reduce layoff pressure
        policy_protection = self.layoff_resistance + (self.human_skill_multiplier - 1.0) * 0.5
        adjusted_layoff_pressure = max(0.01, base_layoff_pressure - policy_protection)
        
        # Convert to daily layoff probability
        daily_layoff_prob = adjusted_layoff_pressure / 30.0
        
        # Determine layoffs using cumulative approach for consistency
        total_days = self.current_step
        total_humans = len([a for a in self.schedule.agents if a.agent_type == AgentType.HUMAN])
        
        expected_total_layoffs = total_humans * (daily_layoff_prob * total_days)
        current_layoffs = len([a for a in self.schedule.agents 
                              if a.agent_type == AgentType.HUMAN and getattr(a, 'is_laid_off', False)])
        
        layoffs_needed = expected_total_layoffs - current_layoffs
        
        print(f"Layoff analysis:")
        print(f"  Base pressure: {base_layoff_pressure:.3f}, Policy protection: {policy_protection:.3f}")
        print(f"  Adjusted pressure: {adjusted_layoff_pressure:.3f}")
        print(f"  Expected layoffs: {expected_total_layoffs:.2f}, Current: {current_layoffs}, Needed: {layoffs_needed:.2f}")
        
        if layoffs_needed >= 1.0:
            layoff_count = int(layoffs_needed)
            layoff_count = min(layoff_count, len(human_agents) // 3)  # Don't lay off more than 1/3
            
            # Smart layoff selection considering multiple factors
            def layoff_priority(agent):
                performance = agent.queries_resolved / max(1, agent.queries_handled)
                job_security = getattr(agent, 'job_security', 1.0)
                financial_security = getattr(agent, 'financial_security', 1.0)
                
                # Lower score = higher layoff priority
                return performance * job_security * financial_security
            
            human_agents.sort(key=layoff_priority)
            
            # Lay off agents with lowest priority scores
            laid_off_count = 0
            for agent in human_agents[:layoff_count]:
                agent.is_laid_off = True
                laid_off_count += 1
                
                # Layoffs affect worker satisfaction
                self.worker_satisfaction = max(0.3, self.worker_satisfaction - 0.05)
                
                print(f"  Laid off agent {agent.unique_id} (priority score: {layoff_priority(agent):.3f})")
            
            print(f"  Total laid off today: {laid_off_count}")
    
    def update_market_dynamics(self):
        """Update market competitiveness based on service quality and costs"""
        # Customer satisfaction affects market position
        satisfaction_factor = self.get_customer_satisfaction()
        
        # Cost efficiency affects competitiveness
        if self.daily_queries:
            avg_cost_per_query = self.get_total_cost() / len(self.daily_queries)
            cost_efficiency = max(0.5, 2.0 - (avg_cost_per_query / 5.0))  # Normalize around $5/query
        else:
            cost_efficiency = 1.0
        
        # Worker satisfaction affects service quality
        service_quality = 0.7 + (self.worker_satisfaction * 0.3)
        
        # Update market competitiveness
        self.market_competitiveness = (satisfaction_factor + cost_efficiency + service_quality) / 3.0
        self.market_competitiveness = max(0.5, min(1.5, self.market_competitiveness))
        
        print(f"Market dynamics: satisfaction {satisfaction_factor:.2f}, cost efficiency {cost_efficiency:.2f}, service quality {service_quality:.2f}")
    
    def balance_government_budget(self, robot_tax_revenue, ubi_cost):
        """Track government fiscal balance"""
        daily_balance = robot_tax_revenue - ubi_cost - self.upskilling_budget
        
        print(f"Government daily balance: ${daily_balance:.0f} (revenue: ${robot_tax_revenue:.0f}, costs: ${ubi_cost + self.upskilling_budget:.0f})")
        
        # If government is running deficit, may need to adjust policies
        if daily_balance < -1000:  # Large deficit
            print("⚠️  Government running large deficit - policy adjustments may be needed")
    
    def step(self):
        """Execute one simulation step with holistic policy system"""
        self.current_step += 1
        
        # Generate and process daily queries
        self.generate_daily_queries()
        self.process_queries()
        
        # Agent daily maintenance
        for agent in self.schedule.agents:
            if hasattr(agent, 'daily_recovery'):
                agent.daily_recovery()
        
        # Apply holistic policy system
        self.apply_holistic_policies()
        
        # Collect data
        self.datacollector.collect(self)
    
    # Enhanced reporter methods
    def get_total_queries_resolved(self):
        return sum(1 for q in self.daily_queries if q.status == QueryStatus.RESOLVED)
    
    def get_total_cost(self):
        return sum(q.cost for q in self.daily_queries)
    
    def get_total_revenue(self):
        resolved_count = self.get_total_queries_resolved()
        return resolved_count * self.revenue_per_resolved_query * self.market_competitiveness
    
    def get_average_resolution_time(self):
        resolved_queries = [q for q in self.daily_queries if q.status == QueryStatus.RESOLVED]
        if not resolved_queries:
            return 0
        return np.mean([q.total_time for q in resolved_queries])
    
    def get_digital_agent_count(self):
        return len([a for a in self.schedule.agents 
                   if a.agent_type == AgentType.DIGITAL and not getattr(a, 'is_laid_off', False)])
    
    def get_human_agent_count(self):
        return len([a for a in self.schedule.agents 
                   if a.agent_type == AgentType.HUMAN and not getattr(a, 'is_laid_off', False)])
    
    def get_unemployment_rate(self):
        human_agents = [a for a in self.schedule.agents if a.agent_type == AgentType.HUMAN]
        if not human_agents:
            return 0
        laid_off = sum(1 for a in human_agents if getattr(a, 'is_laid_off', False))
        return laid_off / len(human_agents)
    
    def get_customer_satisfaction(self):
        if not self.daily_queries:
            return 0.8
        
        resolved_rate = self.get_total_queries_resolved() / len(self.daily_queries)
        avg_time = self.get_average_resolution_time()
        
        # Enhanced satisfaction model
        time_penalty = min(1.0, avg_time / 30.0)
        worker_satisfaction_bonus = (self.worker_satisfaction - 1.0) * 0.2
        skill_bonus = (self.human_skill_multiplier - 1.0) * 0.1
        
        satisfaction = resolved_rate * (1.0 - time_penalty * 0.3) + worker_satisfaction_bonus + skill_bonus
        return max(0.2, min(1.0, satisfaction))
    
    def get_policy_effectiveness(self):
        """Measure overall policy effectiveness"""
        unemployment_penalty = self.get_unemployment_rate() * 0.5
        satisfaction_bonus = self.get_customer_satisfaction() * 0.3
        market_bonus = self.market_competitiveness * 0.2
        
        return max(0.0, satisfaction_bonus + market_bonus - unemployment_penalty)
    
    def get_social_welfare_index(self):
        """Comprehensive social welfare measure"""
        employment_factor = 1.0 - self.get_unemployment_rate()
        worker_satisfaction_factor = self.worker_satisfaction
        economic_performance = self.market_competitiveness
        
        return (employment_factor + worker_satisfaction_factor + economic_performance) / 3.0
    
# ==========================================
# Streamlit Dashboard
# ==========================================
def test_holistic_policies():
    """Test the holistic policy system"""
    st.write("🔧 **Testing Holistic Policy System:**")
    
    # Test with realistic policy combinations
    test_scenarios = [
        {
            "name": "High Automation + Robot Tax + UBI",
            "config": {
                'queries_per_day': 100,
                'revenue_per_resolved_query': 15.0,
                'robot_tax_rate': 0.25,  # 25% robot tax
                'ubi_amount': 50.0,      # $50/day UBI
                'upskilling_budget': 500.0,  # $500/day upskilling
                'layoff_rate': 0.15,     # 15% monthly layoff rate
                'router_count': 2,
                'digital_agents': {'level_1_count': 15, 'level_2_count': 10, 'level_3_count': 5, 'level_4_count': 2},
                'human_agents': {'level_1_count': 10, 'level_2_count': 5, 'level_3_count': 3, 'level_4_count': 2},
                'complexity_distribution': {
                    QueryComplexity.LOW: 0.4, QueryComplexity.MEDIUM: 0.35,
                    QueryComplexity.HIGH: 0.20, QueryComplexity.EXPERT: 0.05,
                }
            }
        },
        {
            "name": "Balanced Approach",
            "config": {
                'queries_per_day': 100,
                'revenue_per_resolved_query': 15.0,
                'robot_tax_rate': 0.15,  # 15% robot tax
                'ubi_amount': 30.0,      # $30/day UBI
                'upskilling_budget': 800.0,  # $800/day upskilling
                'layoff_rate': 0.08,     # 8% monthly layoff rate
                'router_count': 2,
                'digital_agents': {'level_1_count': 8, 'level_2_count': 6, 'level_3_count': 4, 'level_4_count': 2},
                'human_agents': {'level_1_count': 15, 'level_2_count': 10, 'level_3_count': 5, 'level_4_count': 3},
                'complexity_distribution': {
                    QueryComplexity.LOW: 0.4, QueryComplexity.MEDIUM: 0.35,
                    QueryComplexity.HIGH: 0.20, QueryComplexity.EXPERT: 0.05,
                }
            }
        }
    ]
    
    results = []
    
    for scenario in test_scenarios:
        st.write(f"**Testing: {scenario['name']}**")
        
        with st.spinner(f"Running {scenario['name']}..."):
            model = CustomerSupportModel(scenario['config'])
            
            # Run for 60 days to see policy effects develop
            for i in range(60):
                model.step()
            
            # Collect final metrics
            final_humans = model.get_human_agent_count()
            unemployment_rate = model.get_unemployment_rate()
            customer_satisfaction = model.get_customer_satisfaction()
            worker_satisfaction = model.worker_satisfaction
            market_competitiveness = model.market_competitiveness
            policy_effectiveness = model.get_policy_effectiveness()
            social_welfare = model.get_social_welfare_index()
            
            # Government finances
            total_robot_tax = model.government_revenue
            total_ubi_cost = model.total_ubi_paid
            total_upskilling = model.total_upskilling_spent
            govt_balance = total_robot_tax - total_ubi_cost - total_upskilling
            
            result = {
                "Scenario": scenario['name'],
                "Final Human Agents": final_humans,
                "Unemployment Rate": f"{unemployment_rate:.1%}",
                "Customer Satisfaction": f"{customer_satisfaction:.2f}",
                "Worker Satisfaction": f"{worker_satisfaction:.2f}",
                "Market Competitiveness": f"{market_competitiveness:.2f}",
                "Policy Effectiveness": f"{policy_effectiveness:.2f}",
                "Social Welfare Index": f"{social_welfare:.2f}",
                "Govt Revenue (Tax)": f"${total_robot_tax:.0f}",
                "Govt Spending (UBI+Skills)": f"${total_ubi_cost + total_upskilling:.0f}",
                "Govt Balance": f"${govt_balance:.0f}",
                "Financially Sustainable": "✅" if govt_balance >= 0 else "❌"
            }
            results.append(result)
            
            # Show brief results
            st.write(f"- Final human agents: {final_humans}")
            st.write(f"- Unemployment: {unemployment_rate:.1%}")
            st.write(f"- Social welfare index: {social_welfare:.2f}")
            st.write(f"- Government balance: ${govt_balance:.0f}")
            
            if unemployment_rate < 0.3 and social_welfare > 0.7 and govt_balance >= -5000:
                st.success("✅ Policy combination successful!")
            else:
                st.warning("⚠️ Policy needs adjustment")
            
            st.write("---")
    
    # Display comparison table
    st.write("**📊 Policy Scenario Comparison:**")
    df_results = pd.DataFrame(results)
    st.dataframe(df_results, use_container_width=True)
    
    # Policy recommendations
    st.write("**🎯 Policy Insights:**")
    
    best_scenario = max(results, key=lambda x: float(x["Social Welfare Index"]))
    worst_scenario = min(results, key=lambda x: float(x["Social Welfare Index"]))
    
    st.write(f"**Best performing:** {best_scenario['Scenario']} (Social Welfare: {best_scenario['Social Welfare Index']})")
    st.write(f"**Needs improvement:** {worst_scenario['Scenario']} (Social Welfare: {worst_scenario['Social Welfare Index']})")
    
    # Financial sustainability analysis
    sustainable_scenarios = [r for r in results if r["Financially Sustainable"] == "✅"]
    if sustainable_scenarios:
        st.write(f"**Financially sustainable scenarios:** {len(sustainable_scenarios)}/{len(results)}")
    else:
        st.warning("⚠️ No scenarios are financially sustainable - consider adjusting policy levels")

def test_layoff_logic():
    """Quick test to see if layoffs work"""
    st.write("🔧 **Testing Layoff Logic:**")
    
    test_config = {
        'queries_per_day': 100,
        'revenue_per_resolved_query': 15.0,
        'robot_tax_rate': 0.0,
        'ubi_amount': 0.0,
        'upskilling_budget': 0.0,
        'layoff_rate': 0.20,  # 20% monthly = aggressive
        'router_count': 2,
        'digital_agents': {'level_1_count': 5, 'level_2_count': 3, 'level_3_count': 2, 'level_4_count': 1},
        'human_agents': {'level_1_count': 10, 'level_2_count': 5, 'level_3_count': 3, 'level_4_count': 2},
        'complexity_distribution': {
            QueryComplexity.LOW: 0.4, QueryComplexity.MEDIUM: 0.35,
            QueryComplexity.HIGH: 0.20, QueryComplexity.EXPERT: 0.05,
        }
    }
    
    with st.spinner("Running layoff test..."):
        model = CustomerSupportModel(test_config)
        
        initial_humans = model.get_human_agent_count()
        st.write(f"**Initial human agents:** {initial_humans}")
        
        for i in range(30):
            model.step()
        
        final_humans = model.get_human_agent_count()
        unemployment_rate = model.get_unemployment_rate()
        
        st.write(f"**Final human agents:** {final_humans}")
        st.write(f"**Unemployment rate:** {unemployment_rate:.1%}")
        st.write(f"**Agents lost:** {initial_humans - final_humans}")
        
        if unemployment_rate > 0:
            st.success("✅ Layoffs are working!")
        else:
            st.error("❌ Layoffs are NOT working - bug confirmed")

def main():
    st.title("🤖 AI Workforce Policy Impact Simulator")
    st.markdown("**Holistic policy simulation** with realistic economic modeling and policy interactions")
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("🔧 Simulation Configuration")
        
        # Debug Tools Section
        st.subheader("🔧 Debug & Testing")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Test Layoffs", help="Quick layoff mechanism test"):
                test_layoff_logic()
                st.stop()
        with col2:
            if st.button("Test Policies", help="Test holistic policy interactions"):
                test_holistic_policies()
                st.stop()
        
        st.markdown("---")
        
        # Simulation Parameters
        st.subheader("📊 Simulation Parameters")
        simulation_days = st.slider("Simulation Days", 30, 365, 120, help="Longer simulations show policy effects better")
        queries_per_day = st.slider("Queries per Day", 50, 1000, 200)
        revenue_per_query = st.slider("Revenue per Query ($)", 5.0, 50.0, 15.0)
        
        # Workforce Configuration
        st.subheader("👥 Initial Workforce")
        
        # Quick presets
        preset = st.selectbox("Quick Presets", [
            "Custom",
            "Human-Heavy (Traditional)",
            "Balanced (Hybrid)", 
            "AI-Heavy (Modern)",
            "Minimal (Startup)"
        ])
        
        if preset == "Human-Heavy (Traditional)":
            digital_l1, digital_l2, digital_l3, digital_l4 = 5, 3, 2, 1
            human_l1, human_l2, human_l3, human_l4 = 25, 15, 8, 4
            router_count = 2
        elif preset == "Balanced (Hybrid)":
            digital_l1, digital_l2, digital_l3, digital_l4 = 12, 8, 5, 2
            human_l1, human_l2, human_l3, human_l4 = 15, 10, 6, 3
            router_count = 3
        elif preset == "AI-Heavy (Modern)":
            digital_l1, digital_l2, digital_l3, digital_l4 = 20, 15, 8, 4
            human_l1, human_l2, human_l3, human_l4 = 8, 5, 3, 2
            router_count = 4
        elif preset == "Minimal (Startup)":
            digital_l1, digital_l2, digital_l3, digital_l4 = 3, 2, 1, 1
            human_l1, human_l2, human_l3, human_l4 = 5, 3, 2, 1
            router_count = 1
        else:  # Custom
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Digital Agents**")
                digital_l1 = st.number_input("Digital L1", 0, 50, 10, key="dig_l1")
                digital_l2 = st.number_input("Digital L2", 0, 50, 5, key="dig_l2")
                digital_l3 = st.number_input("Digital L3", 0, 20, 3, key="dig_l3")
                digital_l4 = st.number_input("Digital L4", 0, 10, 1, key="dig_l4")
            
            with col2:
                st.write("**Human Agents**")
                human_l1 = st.number_input("Human L1", 0, 100, 20, key="hum_l1")
                human_l2 = st.number_input("Human L2", 0, 50, 10, key="hum_l2")
                human_l3 = st.number_input("Human L3", 0, 20, 5, key="hum_l3")
                human_l4 = st.number_input("Human L4", 0, 10, 2, key="hum_l4")
            
            router_count = st.number_input("Router Agents", 1, 10, 3)
        
        # Show workforce summary
        total_digital = digital_l1 + digital_l2 + digital_l3 + digital_l4
        total_human = human_l1 + human_l2 + human_l3 + human_l4
        automation_ratio = total_digital / (total_digital + total_human) if (total_digital + total_human) > 0 else 0
        
        st.info(f"**Workforce:** {total_human} humans + {total_digital} digital = {automation_ratio:.1%} automation")
        
        # Policy Interventions
        st.subheader("🏛️ Policy Interventions")
        
        # Policy presets
        policy_preset = st.selectbox("Policy Package", [
            "Custom",
            "Free Market (No Intervention)",
            "Progressive (High Support)",
            "Moderate (Balanced)",
            "Experimental (High Tax + UBI)"
        ])
        
        if policy_preset == "Free Market (No Intervention)":
            robot_tax_rate, ubi_amount, upskilling_budget, layoff_rate = 0.0, 0.0, 0.0, 0.12
        elif policy_preset == "Progressive (High Support)":
            robot_tax_rate, ubi_amount, upskilling_budget, layoff_rate = 0.20, 60.0, 1000.0, 0.05
        elif policy_preset == "Moderate (Balanced)":
            robot_tax_rate, ubi_amount, upskilling_budget, layoff_rate = 0.12, 35.0, 600.0, 0.08
        elif policy_preset == "Experimental (High Tax + UBI)":
            robot_tax_rate, ubi_amount, upskilling_budget, layoff_rate = 0.35, 80.0, 1200.0, 0.03
        else:  # Custom
            robot_tax_rate = st.slider("Robot Tax Rate (%)", 0.0, 50.0, 15.0, 
                                     help="Tax on digital agent productivity") / 100
            ubi_amount = st.slider("UBI per Human per Day ($)", 0.0, 100.0, 40.0,
                                 help="Universal Basic Income for all humans")
            upskilling_budget = st.slider("Daily Upskilling Budget ($)", 0.0, 2000.0, 750.0,
                                        help="Investment in human skill development")
            layoff_rate = st.slider("Monthly Layoff Rate (%)", 0.0, 25.0, 10.0,
                                  help="Base rate before policy adjustments") / 100
        
        # Show policy impact preview
        daily_robot_tax_est = total_digital * 50 * robot_tax_rate if robot_tax_rate > 0 else 0
        daily_ubi_cost = total_human * ubi_amount
        daily_govt_balance = daily_robot_tax_est - daily_ubi_cost - upskilling_budget
        
        if daily_govt_balance >= 0:
            st.success(f"💰 Est. daily govt surplus: ${daily_govt_balance:.0f}")
        else:
            st.warning(f"⚠️ Est. daily govt deficit: ${abs(daily_govt_balance):.0f}")
        
        # Query Complexity
        st.subheader("📋 Query Complexity")
        complexity_low = st.slider("Low Complexity (%)", 0, 100, 40, key="comp_low") / 100
        complexity_medium = st.slider("Medium Complexity (%)", 0, 100, 35, key="comp_med") / 100
        complexity_high = st.slider("High Complexity (%)", 0, 100, 20, key="comp_high") / 100
        complexity_expert = max(0, 1.0 - complexity_low - complexity_medium - complexity_high)
        st.write(f"Expert Complexity: {complexity_expert:.1%}")
    
    # Build configuration
    config = {
        'queries_per_day': queries_per_day,
        'revenue_per_resolved_query': revenue_per_query,
        'robot_tax_rate': robot_tax_rate,
        'ubi_amount': ubi_amount,
        'upskilling_budget': upskilling_budget,
        'layoff_rate': layoff_rate,
        'router_count': router_count,
        'digital_agents': {
            'level_1_count': digital_l1,
            'level_2_count': digital_l2,
            'level_3_count': digital_l3,
            'level_4_count': digital_l4,
        },
        'human_agents': {
            'level_1_count': human_l1,
            'level_2_count': human_l2,
            'level_3_count': human_l3,
            'level_4_count': human_l4,
        },
        'complexity_distribution': {
            QueryComplexity.LOW: complexity_low,
            QueryComplexity.MEDIUM: complexity_medium,
            QueryComplexity.HIGH: complexity_high,
            QueryComplexity.EXPERT: complexity_expert,
        }
    }
    
    # Main content area
    st.subheader("🚀 Run Holistic Policy Simulation")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Workforce", f"{total_human + total_digital}", f"{automation_ratio:.1%} digital")
    with col2:
        st.metric("Est. Daily Robot Tax", f"${daily_robot_tax_est:.0f}", f"{robot_tax_rate:.1%} rate")
    with col3:
        st.metric("Est. Daily UBI Cost", f"${daily_ubi_cost:.0f}", f"{total_human} recipients")
    
    if st.button("🚀 Run Simulation", type="primary", use_container_width=True):
        with st.spinner("Running holistic policy simulation..."):
            # Create and run model
            model = CustomerSupportModel(config)
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i in range(simulation_days):
                model.step()
                progress_bar.progress((i + 1) / simulation_days)
                
                # Show periodic updates
                if i % 30 == 0:
                    status_text.text(f"Day {i+1}: {model.get_human_agent_count()} humans active, "
                                   f"satisfaction: {model.get_customer_satisfaction():.2f}")
            
            # Collect results
            model_data = model.datacollector.get_model_vars_dataframe()
            
        st.success("Simulation completed!")
        
        # Display enhanced results
        display_enhanced_results(model, model_data, simulation_days)

def display_enhanced_results(model, model_data, simulation_days):
    """Display enhanced results with policy impact analysis"""
    
    # Executive Summary
    st.header("📊 Executive Summary")
    
    total_queries = model_data['Total_Queries_Resolved'].sum()
    total_revenue = model_data['Total_Revenue'].sum()
    total_cost = model_data['Total_Cost'].sum()
    profit = total_revenue - total_cost
    final_unemployment = model_data['Unemployment_Rate'].iloc[-1]
    final_satisfaction = model_data['Customer_Satisfaction'].iloc[-1]
    final_social_welfare = model_data['Social_Welfare_Index'].iloc[-1]
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("Total Profit", f"${profit:,.0f}", 
                 f"{(profit/total_revenue)*100:.1f}% margin" if total_revenue > 0 else "0%")
    
    with col2:
        st.metric("Final Unemployment", f"{final_unemployment:.1%}",
                 delta="Lower is better", delta_color="inverse")
    
    with col3:
        st.metric("Customer Satisfaction", f"{final_satisfaction:.2f}",
                 delta="Target: >0.80")
    
    with col4:
        st.metric("Social Welfare Index", f"{final_social_welfare:.2f}",
                 delta="Target: >0.70")
    
    with col5:
        policy_score = model_data['Policy_Effectiveness'].iloc[-1]
        st.metric("Policy Effectiveness", f"{policy_score:.2f}",
                 delta="Higher is better")
    
    # Policy Impact Analysis
    st.header("🏛️ Policy Impact Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Government Finances")
        total_robot_tax = model.government_revenue
        total_ubi_paid = model.total_ubi_paid
        total_upskilling = model.total_upskilling_spent
        net_balance = total_robot_tax - total_ubi_paid - total_upskilling
        
        st.write(f"**Robot Tax Revenue:** ${total_robot_tax:,.0f}")
        st.write(f"**UBI Payments:** ${total_ubi_paid:,.0f}")
        st.write(f"**Upskilling Investment:** ${total_upskilling:,.0f}")
        st.write(f"**Net Government Balance:** ${net_balance:,.0f}")
        
        if net_balance >= 0:
            st.success("✅ Fiscally sustainable policy package")
        else:
            st.warning(f"⚠️ Government deficit of ${abs(net_balance):,.0f}")
    
    with col2:
        st.subheader("Social Outcomes")
        initial_humans = model_data['Human_Agent_Count'].iloc[0]
        final_humans = model_data['Human_Agent_Count'].iloc[-1]
        jobs_lost = initial_humans - final_humans
        
        st.write(f"**Initial Human Jobs:** {initial_humans}")
        st.write(f"**Final Human Jobs:** {final_humans}")
        st.write(f"**Net Jobs Lost:** {jobs_lost}")
        st.write(f"**Worker Satisfaction:** {model.worker_satisfaction:.2f}")
        st.write(f"**Market Competitiveness:** {model.market_competitiveness:.2f}")
    
    # Time Series Visualizations
    st.header("📈 Performance Over Time")
    
    # Create comprehensive dashboard
    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=('Workforce Evolution', 'Economic Performance', 
                       'Policy Effectiveness', 'Social Welfare',
                       'Government Finances', 'Market Dynamics'),
        specs=[[{"secondary_y": False}, {"secondary_y": True}],
               [{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Workforce evolution
    fig.add_trace(
        go.Scatter(x=model_data.index, y=model_data['Human_Agent_Count'],
                  name='Human Agents', line=dict(color='blue')),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=model_data.index, y=model_data['Digital_Agent_Count'],
                  name='Digital Agents', line=dict(color='purple')),
        row=1, col=1
    )
    
    # Economic performance
    fig.add_trace(
        go.Scatter(x=model_data.index, y=model_data['Total_Revenue'],
                  name='Revenue', line=dict(color='green')),
        row=1, col=2
    )
    fig.add_trace(
        go.Scatter(x=model_data.index, y=model_data['Total_Cost'],
                  name='Cost', line=dict(color='red')),
        row=1, col=2
    )
    
    # Policy effectiveness
    fig.add_trace(
        go.Scatter(x=model_data.index, y=model_data['Policy_Effectiveness'],
                  name='Policy Score', line=dict(color='orange')),
        row=2, col=1
    )
    
    # Social welfare
    fig.add_trace(
        go.Scatter(x=model_data.index, y=model_data['Social_Welfare_Index'],
                  name='Social Welfare', line=dict(color='teal')),
        row=2, col=2
    )
    
    # Government finances
    fig.add_trace(
        go.Scatter(x=model_data.index, y=model_data['Government_Revenue'],
                  name='Govt Revenue', line=dict(color='navy')),
        row=3, col=1
    )
    
    # Market dynamics
    fig.add_trace(
        go.Scatter(x=model_data.index, y=model_data['Market_Competitiveness'],
                  name='Competitiveness', line=dict(color='magenta')),
        row=3, col=2
    )
    
    fig.update_layout(height=900, showlegend=True, title="Holistic Policy Impact Dashboard")
    st.plotly_chart(fig, use_container_width=True)
    
    # Policy Recommendations
    st.header("🎯 Policy Recommendations")
    
    recommendations = []
    
    # Unemployment analysis
    if final_unemployment > 0.25:
        recommendations.append("🚨 **Critical unemployment** - Increase UBI and upskilling, reduce layoff rates")
    elif final_unemployment > 0.15:
        recommendations.append("⚠️ **High unemployment** - Consider policy adjustments to protect workers")
    else:
        recommendations.append("✅ **Manageable unemployment** - Current policies maintaining social stability")
    
    # Customer satisfaction analysis
    if final_satisfaction < 0.7:
        recommendations.append("📉 **Poor customer satisfaction** - Increase human agents or improve AI capabilities")
    elif final_satisfaction > 0.85:
        recommendations.append("🌟 **Excellent customer satisfaction** - Well-balanced workforce")
    
    # Fiscal sustainability
    if net_balance < -10000:
        recommendations.append("💸 **Unsustainable deficit** - Increase robot tax or reduce spending")
    elif net_balance > 20000:
        recommendations.append("💰 **Large surplus** - Consider expanding social programs")
    
    # Social welfare
    if final_social_welfare < 0.6:
        recommendations.append("🏚️ **Low social welfare** - Comprehensive policy reform needed")
    elif final_social_welfare > 0.8:
        recommendations.append("🏆 **High social welfare** - Excellent policy balance achieved")
    
    # Market competitiveness
    if model.market_competitiveness < 0.8:
        recommendations.append("📉 **Declining competitiveness** - Focus on efficiency and service quality")
    
    if not recommendations:
        recommendations.append("✅ **Well-balanced system** - Current policies showing good results")
    
    for rec in recommendations:
        st.write(rec)
    
    # Detailed Data Export
    st.header("📥 Export Detailed Results")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📊 Download Full Data"):
            csv = model_data.to_csv(index=True)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=f"holistic_simulation_{simulation_days}days.csv",
                mime="text/csv"
            )
    
    with col2:
        # Create summary report
        summary_data = {
            "Metric": [
                "Simulation Days", "Total Queries Processed", "Final Human Agents",
                "Final Unemployment Rate", "Customer Satisfaction", "Social Welfare Index",
                "Total Revenue", "Total Cost", "Net Profit", "Government Balance",
                "Policy Effectiveness", "Market Competitiveness"
            ],
            "Value": [
                simulation_days, total_queries, final_humans, f"{final_unemployment:.1%}",
                f"{final_satisfaction:.2f}", f"{final_social_welfare:.2f}",
                f"${total_revenue:,.0f}", f"${total_cost:,.0f}", f"${profit:,.0f}",
                f"${net_balance:,.0f}", f"{policy_score:.2f}", f"{model.market_competitiveness:.2f}"
            ]
        }
        
        summary_csv = pd.DataFrame(summary_data).to_csv(index=False)
        st.download_button(
            label="Download Summary",
            data=summary_csv,
            file_name=f"policy_summary_{simulation_days}days.csv",
            mime="text/csv"
        )
    
    with col3:
        if st.button("🔄 Run Another Scenario"):
            st.rerun()

if __name__ == "__main__":
    main()