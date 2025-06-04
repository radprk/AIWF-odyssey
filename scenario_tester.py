from model import WorkFutureModel, WorkerAgent, CorporationAgent, GovernmentAgent
import matplotlib.pyplot as plt
import pandas as pd
import json

def create_scenarios():
    """Define different test scenarios with descriptions."""
    scenarios = {
        # Worker-focused scenarios
        "Worker_HighAdapt": {
            "description": "Workers with high adaptability",
            "params": {"automation_level": 0.3},
            "agent_changes": {"worker_adaptability": 0.9}
        },
        "Worker_LowAdapt": {
            "description": "Workers with low adaptability",
            "params": {"automation_level": 0.3},
            "agent_changes": {"worker_adaptability": 0.1}
        },
        
        # Corporation-focused scenarios
        "Corp_AggressiveAuto": {
            "description": "Corporations aggressively automate",
            "params": {"automation_level": 0.5},
            "agent_changes": {"corp_automation_tendency": 0.9}
        },
        "Corp_HumanCentric": {
            "description": "Corporations prefer human workers",
            "params": {"automation_level": 0.2},
            "agent_changes": {"corp_automation_tendency": 0.3}
        },
        
        # Government-focused scenarios
        "Gov_ProWorker": {
            "description": "Government actively supports workers",
            "params": {"automation_level": 0.5},
            "agent_changes": {"gov_policy": "pro_worker", "reskilling_investment": 0.8}
        },
        "Gov_LaissezFaire": {
            "description": "Government minimal intervention",
            "params": {"automation_level": 0.5},
            "agent_changes": {"gov_policy": "laissez_faire", "reskilling_investment": 0.1}
        }
    }
    return scenarios

def run_scenario(name, params, agent_changes):
    """Run a single scenario with specified parameters."""
    print(f"Running scenario: {name}")
    
    # Create a new model instance for each scenario
    model = WorkFutureModel(**params)
    
    # Apply agent changes after model creation
    if "worker_adaptability" in agent_changes:
        for agent in model.schedule.agents:
            if isinstance(agent, WorkerAgent):
                agent.adaptability = agent_changes["worker_adaptability"]
    
    if "corp_automation_tendency" in agent_changes:
        for agent in model.schedule.agents:
            if isinstance(agent, CorporationAgent):
                agent.automation_tendency = agent_changes["corp_automation_tendency"]
    
    if "gov_policy" in agent_changes:
        for agent in model.schedule.agents:
            if isinstance(agent, GovernmentAgent):
                agent.policy_type = agent_changes["gov_policy"]
    
    # Run simulation
    for _ in range(50):
        model.step()
    
    return model.datacollector.get_model_vars_dataframe()

def compare_scenarios(scenarios_data):
    """Create comparison plots for scenarios."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle("Scenario Comparisons", fontsize=16)
    
    metrics = ["Employment", "Average_Skill", "Worker_Wellbeing", "Corporate_Profit"]
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx // 2, idx % 2]
        for name, data in scenarios_data.items():
            ax.plot(data.index, data[metric], label=name)
        ax.set_title(metric)
        ax.set_xlabel("Steps")
        ax.set_ylabel(metric)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.savefig("scenario_comparison.png", dpi=300, bbox_inches='tight')
    plt.show()

def detailed_analysis(scenario_name, data):
    """Provide detailed analysis for a specific scenario."""
    print(f"\n=== Detailed Analysis: {scenario_name} ===")
    
    # Calculate key metrics
    final_values = data.iloc[-1]
    employment_change = data['Employment'].iloc[-1] - data['Employment'].iloc[0]
    skill_growth = data['Average_Skill'].iloc[-1] - data['Average_Skill'].iloc[0]
    
    print(f"Final Employment: {final_values['Employment']:.2%}")
    print(f"Employment Change: {employment_change:.2%}")
    print(f"Skill Growth: {skill_growth:.2f}")
    print(f"Final Worker Well-being: {final_values['Worker_Wellbeing']:.2f}")
    print(f"Final Corporate Profit: {final_values['Corporate_Profit']:.2f}")
    
    # Find critical points
    if 'Employment' in data.columns:
        steepest_decline = data['Employment'].diff().min()
        if steepest_decline < 0:  # Only if there's a decline
            steepest_decline_step = data['Employment'].diff().idxmin()
            print(f"Steepest Employment Decline: {steepest_decline:.2%} at step {steepest_decline_step}")
    
    return {
        "final_values": final_values.to_dict(),
        "employment_change": employment_change,
        "skill_growth": skill_growth
    }

if __name__ == "__main__":
    scenarios = create_scenarios()
    scenario_results = {}
    analysis_results = {}
    
    # Run all scenarios
    for name, config in scenarios.items():
        print(f"\n{config['description']}")
        data = run_scenario(name, config['params'], config['agent_changes'])
        scenario_results[name] = data
        analysis_results[name] = detailed_analysis(name, data)
    
    # Compare scenarios
    compare_scenarios(scenario_results)
    
    # Save analysis results
    with open("scenario_analysis.json", "w") as f:
        json.dump(analysis_results, f, indent=4)
    
    # Create a summary visualization
    plt.figure(figsize=(12, 8))
    scenarios_list = list(analysis_results.keys())
    employment_values = [analysis_results[s]['final_values']['Employment'] for s in scenarios_list]
    wellbeing_values = [analysis_results[s]['final_values']['Worker_Wellbeing'] for s in scenarios_list]
    profit_values = [analysis_results[s]['final_values']['Corporate_Profit'] for s in scenarios_list]
    
    x = range(len(scenarios_list))
    width = 0.25
    
    plt.bar([i - width for i in x], employment_values, width, label='Employment')
    plt.bar(x, wellbeing_values, width, label='Worker Wellbeing')
    plt.bar([i + width for i in x], profit_values, width, label='Corporate Profit')
    
    plt.xlabel('Scenarios')
    plt.ylabel('Values')
    plt.title('Final Outcomes by Scenario')
    plt.xticks(x, scenarios_list, rotation=45, ha='right')
    plt.legend()
    plt.tight_layout()
    plt.savefig("final_outcomes_comparison.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\nScenario testing complete! Check scenario_comparison.png and scenario_analysis.json")