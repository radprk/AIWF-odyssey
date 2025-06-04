 
# visualization.py
from mesa.visualization.modules import ChartModule
from mesa.visualization.ModularVisualization import ModularServer
from model import WorkFutureModel

def create_server():
    """Create a visualization server for the model."""
    
    chart_employment = ChartModule([
        {"Label": "Employment", "Color": "Blue"},
        {"Label": "Worker_Wellbeing", "Color": "Green"}
    ])
    
    chart_corporate = ChartModule([
        {"Label": "Corporate_Profit", "Color": "Red"},
        {"Label": "Automation_Level", "Color": "Orange"}
    ])
    
    server = ModularServer(
        WorkFutureModel,
        [chart_employment, chart_corporate],
        "AI Work Force Simulation",
        {"num_workers": 100, "num_corporations": 10, "automation_level": 0.3}
    )
    
    return server

if __name__ == "__main__":
    server = create_server()
    server.port = 8521
    server.launch()