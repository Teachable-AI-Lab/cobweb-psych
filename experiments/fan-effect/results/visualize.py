from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np


def plot_fan(df: pd.DataFrame, out_dir: Path, label: str):
	"""
	Plot fan effect results.
	
	Citation: Anderson, J. R. (1974). Retrieval of propositional information from 
	          long-term memory. Cognitive Psychology, 6(4), 451-474.
	
	Expected: Retrieval probability decreases with fan size (interference).
	"""
	# Use 'prob' column for retrieval probability
	if "prob" in df.columns:
		value_col = "prob"
	elif "mean_prob_true_fact" in df.columns:
		value_col = "mean_prob_true_fact"
	else:
		raise ValueError("No suitable probability column found in fan-effect CSV")

	# Learning curves by fan size
	mean_fan = df.groupby(["fan_size", "block"], as_index=False)[value_col].mean()
	fig, ax = plt.subplots(figsize=(8, 5))
	sns.lineplot(
		data=mean_fan, x="block", y=value_col, hue="fan_size", marker="o", 
		ax=ax, palette="rocket", linewidth=2
	)
	ax.set_title("Fan Effect: Retrieval Probability vs Training\\nAnderson (1974)", fontsize=12)
	ax.set_ylabel("Retrieval Probability (P(Correct Fact))", fontsize=11)
	ax.set_xlabel("Training Block", fontsize=11)
	ax.set_ylim(0, 1.05)
	ax.legend(title='Fan Size', bbox_to_anchor=(1.05, 1), loc='upper left')
	fig.tight_layout()
	fig.savefig(out_dir / f"fan_effect_blocks_{label}.png", dpi=200)

	# Final retrieval probability vs fan size
	end_fan = mean_fan.groupby("fan_size", as_index=False).agg(prob_end=(value_col, "last"))
	fig, ax = plt.subplots(figsize=(7, 4.5))
	sns.scatterplot(data=end_fan, x="fan_size", y="prob_end", s=150, ax=ax, color='darkred')
	
	# Add regression line
	if len(end_fan) > 1:
		z = np.polyfit(end_fan["fan_size"], end_fan["prob_end"], 1)
		p = np.poly1d(z)
		ax.plot(end_fan["fan_size"], p(end_fan["fan_size"]), "--", color='gray', 
		        linewidth=2, label=f'Linear fit (slope={z[0]:.3f})')
	
	ax.set_title("Fan Effect: Final Retrieval vs Fan Size\\nAnderson (1974)", fontsize=12)
	ax.set_ylabel("Final Retrieval Probability", fontsize=11)
	ax.set_xlabel("Fan Size (Number of Facts)", fontsize=11)
	ax.set_ylim(0, 1.05)
	
	# Add value annotations
	for i, row in end_fan.iterrows():
		ax.text(row["fan_size"], row["prob_end"] + 0.03, f'{row["prob_end"]:.2f}', 
		        ha='center', va='bottom', fontsize=9)
	
	ax.legend(loc='upper right')
	ax.grid(True, alpha=0.3)
	fig.tight_layout()
	fig.savefig(out_dir / f"fan_effect_final_{label}.png", dpi=200)


def main(
	discrete_csv=Path(__file__).resolve().parent.parent / "results" / "exp_fan_effect_discrete.csv",
	out_dir=Path(__file__).resolve().parent,
):
	sns.set_theme(style="whitegrid")
	if discrete_csv.exists():
		df = pd.read_csv(discrete_csv)
		
		# Anderson 1974: RT increases with Fan Size
		# Cobweb: Probability decreases with Fan Size
		# Transformation: RT ~ 1 / Probability (Surprisal)
		
		# Take final performance
		df_final = df[df["epoch"] == df["epoch"].max()]
		
		# Calculate simulated RT
		# Add small epsilon to avoid divide by zero if prob=0
		df_final["simulated_rt"] = 1.0 / (df_final["prob"] + 0.001)
		
		mean_rt = df_final.groupby("fan_size", as_index=False)["simulated_rt"].mean()
		
		fig, ax = plt.subplots(figsize=(7, 5))
		
		# Line plot: Fan Size vs RT
		sns.lineplot(data=mean_rt, x="fan_size", y="simulated_rt", marker="s", markersize=10, 
		             color="black", linestyle="-", linewidth=2, ax=ax)
		
		# Regression line to show linearity
		if len(mean_rt) > 1:
			z = np.polyfit(mean_rt["fan_size"], mean_rt["simulated_rt"], 1)
			p = np.poly1d(z)
			ax.plot(mean_rt["fan_size"], p(mean_rt["fan_size"]), "--", color='red', alpha=0.6, 
			        label=f"Linear Fit")
			
		ax.set_title("Fan Effect\nAnderson (1974)", fontsize=12)
		ax.set_ylabel("Simulated Reaction Time (1/Prob)", fontsize=11)
		ax.set_xlabel("Fan Size (Facts per Concept)", fontsize=11)
		ax.set_xticks([1, 2, 4, 8])
		ax.legend()
		
		fig.tight_layout()
		fig.savefig(out_dir / "fan_effect_rt_curve_discrete.png", dpi=200)
		
	else:
		print("CSV not found.")


if __name__ == "__main__":
	main()
