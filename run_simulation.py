from model import WorkFutureModel
import matplotlib.pyplot as plt
import pandas as pd

def run_scenarios():
    """Run different scenarios and collect results."""
    
    scenarios = {
        "Base": {"automation_level": 0.3},
        "High_Automation": {"automation_level": 0.7},
        "Low_Automation": {"automation_level": 0.1}
    }
    
    results = {}
    
    for name, params in scenarios.items():
        print(f"Running scenario: {name}")
        model = WorkFutureModel(automation_level=params["automation_level"])
        
        for _ in range(50):  # Run for 50 steps
            model.step()
        
        results[name] = model.datacollector.get_model_vars_dataframe()
    
    return results

def plot_results(results):
    """Plot simulation results."""
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("AI Work Force Simulation Results", fontsize=16)
    
    metrics = ["Employment", "Average_Skill", "Worker_Wellbeing", "Corporate_Profit"]
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx // 2, idx % 2]
        for scenario_name, data in results.items():
            ax.plot(data.index, data[metric], label=scenario_name)
        ax.set_title(metric)
        ax.set_xlabel("Steps")
        ax.set_ylabel(metric)
        ax.legend()
    
    plt.tight_layout()
    plt.savefig("simulation_results.png")
    plt.show()

if __name__ == "__main__":
    results = run_scenarios()
    plot_results(results)
    
    # Save final values
    final_values = {}
    for scenario, data in results.items():
        final_values[scenario] = data.iloc[-1].to_dict()
    
    import json
    with open("final_results.json", "w") as f:
        json.dump(final_values, f, indent=4)
    
    print("Simulation complete! Check simulation_results.png and final_results.json")