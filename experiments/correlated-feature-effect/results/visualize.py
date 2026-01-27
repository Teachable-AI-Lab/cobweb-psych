from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_correlated(df: pd.DataFrame, out_dir: Path, label: str):
	"""
	Plot correlated-feature / XOR difficulty effect.
	
	Citation: Medin, D. L., Altom, M. W., Edelson, S. M., & Freko, D. (1982). 
	          Correlated symptoms and simulated medical classification. 
	          JEP: Learning, Memory, and Cognition, 8(1), 37-50.
	
	Expected: XOR (configural) structure shows slower learning than separable structure.
	"""
	# Learning curves: accuracy vs block for XOR vs separable
	mean_block = df.groupby(["structure", "block"], as_index=False)["accuracy"].mean()
	fig, ax = plt.subplots(figsize=(8, 5))
	sns.lineplot(
		data=mean_block, x="block", y="accuracy", hue="structure", marker="o", 
		ax=ax, palette="Set2", linewidth=2
	)
	ax.set_title("XOR Difficulty: Configural vs Separable Learning\\nMedin et al. (1982)", fontsize=12)
	ax.set_ylabel("Classification Accuracy", fontsize=11)
	ax.set_xlabel("Training Block", fontsize=11)
	ax.set_ylim(0, 1.05)
	ax.axhline(0.5, color='gray', linestyle='--', linewidth=0.8, alpha=0.5, label='Chance')
	ax.legend(title='Structure', loc='lower right')
	fig.tight_layout()
	fig.savefig(out_dir / f"correlated_feature_blocks_{label}.png", dpi=200)

	# Final accuracy bar plot
	final_block = mean_block.groupby("structure", as_index=False).agg(
		accuracy_end=("accuracy", "last")
	)
	fig, ax = plt.subplots(figsize=(6, 4.5))
	sns.barplot(data=final_block, x="structure", y="accuracy_end", ax=ax, palette="Set2")
	ax.set_title("Final Accuracy: XOR vs Separable\\nMedin et al. (1982)", fontsize=12)
	ax.set_ylabel("Final Accuracy", fontsize=11)
	ax.set_xlabel("Category Structure", fontsize=11)
	ax.set_ylim(0, 1.05)
	ax.axhline(0.5, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
	
	# Add value annotations
	for i, row in final_block.iterrows():
		ax.text(i, row["accuracy_end"] + 0.02, f'{row["accuracy_end"]:.2f}', 
		        ha='center', va='bottom', fontsize=9)
	
	fig.tight_layout()
	fig.savefig(out_dir / f"correlated_feature_final_{label}.png", dpi=200)


def main(
	discrete_csv=Path(__file__).resolve().parent.parent / "results" / "exp_correlated_feature_discrete.csv",
	out_dir=Path(__file__).resolve().parent,
):
	sns.set_theme(style="whitegrid")
	if discrete_csv.exists():
		df = pd.read_csv(discrete_csv)
		
		# Graph: Bar chart comparing Accuracy for Correlated vs Uncorrelated
		# X: Condition (Structure)
		# Y: Accuracy
		
		# Filter final block/epoch
		df_final = df[(df["epoch"] == df["epoch"].max()) & (df["block"] == df["block"].max())]
		
		mean_acc = df_final.groupby("structure", as_index=False)["accuracy"].mean()
		
		fig, ax = plt.subplots(figsize=(6, 5))
		
		sns.barplot(data=mean_acc, x="structure", y="accuracy", palette="Set2", ax=ax)
		
		ax.set_title("Correlated Feature Effect\nMedin et al (1982)", fontsize=12)
		ax.set_ylabel("Classification Accuracy", fontsize=11)
		ax.set_xlabel("Structure Type", fontsize=11)
		ax.set_ylim(0, 1.05)
		
		fig.tight_layout()
		fig.savefig(out_dir / "correlated_feature_bar_discrete.png", dpi=200)
		
	else:
		print("CSV not found.")


if __name__ == "__main__":
	main()
