# from mesa_model.model import CallCenterModel
# import matplotlib.pyplot as plt
# import pandas as pd

# # === Run a basic simulation ===
# model = CallCenterModel(
#     size="medium",
#     automation_pressure=1.0,  # Try 0.5 to 1.5
#     enable_augmentation=True,
#     ubi=False
# )

# for _ in range(30):
#     model.step()

# # === Plot the results ===
# results = model.datacollector.get_model_vars_dataframe()

# import os
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# # Create output directory if it doesn't exist
# output_dir = "outputs"
# os.makedirs(output_dir, exist_ok=True)

# # === Plot 1: Workforce status over time ===
# results[["Employed", "Automated", "Reskilled"]].plot(figsize=(10, 5))
# plt.title("Call Center Workforce Status Over Time")
# plt.xlabel("Time Step (months)")
# plt.ylabel("Number of Workers")
# plt.grid(True)
# plt.tight_layout()
# plt.savefig(os.path.join(output_dir, "workforce_status.png"))
# plt.close()

# # === Plot 2: Financials (Costs vs Savings) ===
# results[["Cost", "Savings", "ReskillCost", "RobotTax"]].plot(figsize=(10, 5))
# plt.title("Monthly Financial Impact of Automation")
# plt.xlabel("Time Step (months)")
# plt.ylabel("USD")
# plt.grid(True)
# plt.tight_layout()
# plt.savefig(os.path.join(output_dir, "financial_metrics.png"))
# plt.close()

# # === Plot 3: ROI Over Time ===
# results["ROI"].plot(figsize=(10, 5), color='purple')
# plt.title("Return on Investment (ROI) Over Time")
# plt.xlabel("Time Step (months)")
# plt.ylabel("ROI Ratio")
# plt.grid(True)
# plt.tight_layout()
# plt.savefig(os.path.join(output_dir, "roi_over_time.png"))
# plt.close()