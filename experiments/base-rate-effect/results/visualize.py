from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_inverse_base_rate(df: pd.DataFrame, out_dir: Path, label: str):
	"""
	Plot inverse base-rate effect (Medin & Edelson, 1988).
	
	Citation: Medin, D. L., & Edelson, S. M. (1988). Problem structure and the use of 
	          base-rate information from experience. JEP: General, 117(1), 68-85.
	
	Expected: On BC ambiguous trials, bias toward Rare category (inverse base-rate effect).
	"""
	
	# Filter for BC critical trials (ambiguous test)
	bc_trials = df[df["test_type"] == "BC_critical"].copy()
	
	if len(bc_trials) > 0:
		# Plot 1: IBRE rate (percent choosing Rare on BC trials) across training blocks
		ibre_by_block = bc_trials.groupby(["ratio", "block"], as_index=False)["shows_ibre"].mean()
		fig, ax = plt.subplots(figsize=(8, 5))
		sns.lineplot(
			data=ibre_by_block, x="block", y="shows_ibre", hue="ratio", 
			marker="o", ax=ax, palette="magma", linewidth=2
		)
		ax.set_title("Inverse Base-Rate Effect Emergence\\nMedin & Edelson (1988)", fontsize=12)
		ax.set_ylabel("P(Choose Rare on BC trials)", fontsize=11)
		ax.set_xlabel("Training Block", fontsize=11)
		ax.set_ylim(0, 1.05)
		ax.axhline(0.5, color='gray', linestyle='--', linewidth=0.8, alpha=0.5, label='Chance')
		ax.legend(title='Common:Rare Ratio', bbox_to_anchor=(1.05, 1), loc='upper left')
		fig.tight_layout()
		fig.savefig(out_dir / f"base_rate_ibre_blocks_{label}.png", dpi=200)
		
		# Plot 2: Final IBRE rate by ratio
		final_ibre = ibre_by_block.groupby("ratio", as_index=False).agg(
			ibre_final=("shows_ibre", "last")
		)
		fig, ax = plt.subplots(figsize=(6, 4.5))
		sns.barplot(data=final_ibre, x="ratio", y="ibre_final", ax=ax, palette="magma")
		ax.set_title("Final Inverse Base-Rate Effect\\nMedin & Edelson (1988)", fontsize=12)
		ax.set_ylabel("P(Choose Rare on BC)", fontsize=11)
		ax.set_xlabel("Common:Rare Training Ratio", fontsize=11)
		ax.set_ylim(0, 1.05)
		ax.axhline(0.5, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
		
		for i, row in final_ibre.iterrows():
			ax.text(i, row["ibre_final"] + 0.02, f'{row["ibre_final"]:.2f}', 
			        ha='center', va='bottom', fontsize=9)
		
		fig.tight_layout()
		fig.savefig(out_dir / f"base_rate_ibre_final_{label}.png", dpi=200)
	
	# Plot 3: Probabilities for different test types
	if "prob_rare" in df.columns:
		mean_probs = df.groupby(["test_type", "ratio"], as_index=False)[["prob_common", "prob_rare"]].mean()
		
		fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
		
		# Common probability
		sns.barplot(data=mean_probs, x="test_type", y="prob_common", hue="ratio", 
		            ax=axes[0], palette="magma")
		axes[0].set_title("P(Common) by Test Type", fontsize=11)
		axes[0].set_ylabel("P(Common)", fontsize=10)
		axes[0].set_xlabel("Test Pattern", fontsize=10)
		axes[0].set_ylim(0, 1.05)
		axes[0].legend(title='Ratio', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
		axes[0].tick_params(axis='x', rotation=45)
		
		# Rare probability
		sns.barplot(data=mean_probs, x="test_type", y="prob_rare", hue="ratio", 
		            ax=axes[1], palette="magma")
		axes[1].set_title("P(Rare) by Test Type", fontsize=11)
		axes[1].set_ylabel("P(Rare)", fontsize=10)
		axes[1].set_xlabel("Test Pattern", fontsize=10)
		axes[1].set_ylim(0, 1.05)
		axes[1].legend(title='Ratio', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
		axes[1].tick_params(axis='x', rotation=45)
		
		fig.suptitle("Medin & Edelson (1988): Response Probabilities", fontsize=13, y=1.02)
		fig.tight_layout()
		fig.savefig(out_dir / f"base_rate_probabilities_{label}.png", dpi=200, bbox_inches='tight')


def main(
	discrete_csv=Path(__file__).resolve().parent.parent / "results" / "exp_base_rate_discrete.csv",
	out_dir=Path(__file__).resolve().parent,
):
	sns.set_theme(style="whitegrid")
	if discrete_csv.exists():
		df = pd.read_csv(discrete_csv)
		
		# Medin & Edelson Graph: Bar chart of choice proportions
		# X: Stimulus (AB Common, AC Rare, BC Ambiguous)
		# Y: Proportion Choosing Common vs Rare
		# Bars: Grouped
		
		# Filter final epoch
		df_final = df[df["epoch"] == df["epoch"].max()]
		
		# We want Proportion of responses for each category
		# My discrete exp has 'pred_label' ("Common_Disease", "Rare_Disease")
		
		test_types = ["AB_train", "AC_train", "BC_critical"]
		display_names = {"AB_train": "Common\n(AB)", "AC_train": "Rare\n(AC)", "BC_critical": "Ambiguous\n(BC)"}
		
		df_tests = df_final[df_final["test_type"].isin(test_types)].copy()
		
		stats = []
		for t_type in test_types:
			sub = df_tests[df_tests["test_type"] == t_type]
			total = len(sub)
			if total == 0: continue
			p_common = len(sub[sub["pred_label"].str.contains("Common")]) / total
			p_rare = len(sub[sub["pred_label"].str.contains("Rare")]) / total
			
			stats.append({"Stimulus": display_names[t_type], "Response": "Common Disease", "Proportion": p_common})
			stats.append({"Stimulus": display_names[t_type], "Response": "Rare Disease", "Proportion": p_rare})
			
		df_stats = pd.DataFrame(stats)
		
		fig, ax = plt.subplots(figsize=(7, 5))
		
		sns.barplot(data=df_stats, x="Stimulus", y="Proportion", hue="Response", 
		            palette={"Common Disease": "blue", "Rare Disease": "red"}, ax=ax)
		
		ax.set_title("Inverse Base-Rate Effect\nMedin & Edelson (1988)", fontsize=12)
		ax.set_ylabel("Proportion of Responses", fontsize=11)
		ax.set_ylim(0, 1.05)
		ax.legend(title="Response")
		
		# Annotate critical effect: Rare > Common for BC
		ax.text(2, 0.5, "IBRE?", ha='center', va='center', alpha=0.3, fontsize=15, rotation=45)
		
		fig.tight_layout()
		fig.savefig(out_dir / "base_rate_ibre_bar_discrete.png", dpi=200)
		
	else:
		print("CSV not found.")


if __name__ == "__main__":
	main()
