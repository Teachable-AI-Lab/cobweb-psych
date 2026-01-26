from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_label_feedback(df: pd.DataFrame, out_dir: Path, label: str):
	"""
	Plot category-label / feedback effect results.
	
	Citation: Posner, M. I., & Keele, S. W. (1968). On the genesis of abstract ideas.
	          Journal of Experimental Psychology, 77(3, Pt.1), 353–363.
	
	Expected: Higher label rates (feedback) improve accuracy, especially for high distortions.
	"""
	# Learning curves: accuracy vs block for different label rates
	mean_acc = df.groupby(["label_rate", "block"], as_index=False)["accuracy"].mean()
	fig, ax = plt.subplots(figsize=(8, 5))
	sns.lineplot(
		data=mean_acc, x="block", y="accuracy", hue="label_rate", marker="o", 
		ax=ax, palette="viridis", linewidth=2
	)
	ax.set_title("Impact of Category Labels on Learning\nPosner & Keele (1968)", fontsize=12)
	ax.set_ylabel("Classification Accuracy", fontsize=11)
	ax.set_xlabel("Training Block", fontsize=11)
	ax.set_ylim(0, 1.05)
	ax.axhline(0.5, color='gray', linestyle='--', linewidth=0.8, alpha=0.5, label='Chance')
	ax.legend(title='Label Rate', bbox_to_anchor=(1.05, 1), loc='upper left')
	fig.tight_layout()
	fig.savefig(out_dir / f"category_label_feedback_blocks_{label}.png", dpi=200)

	# Final accuracy bar plot
	end_acc = mean_acc.groupby("label_rate", as_index=False).agg(
		accuracy_end=("accuracy", "last")
	)
	fig, ax = plt.subplots(figsize=(6, 4.5))
	sns.barplot(data=end_acc, x="label_rate", y="accuracy_end", ax=ax, palette="viridis")
	ax.set_title("End-State Accuracy vs Label Rate\nPosner & Keele (1968)", fontsize=12)
	ax.set_ylabel("Final Accuracy", fontsize=11)
	ax.set_xlabel("Label Rate (Feedback Probability)", fontsize=11)
	ax.set_ylim(0, 1.05)
	ax.axhline(0.5, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
	
	# Add value annotations
	for i, row in end_acc.iterrows():
		ax.text(i, row["accuracy_end"] + 0.02, f'{row["accuracy_end"]:.2f}', 
		        ha='center', va='bottom', fontsize=9)
	
	fig.tight_layout()
	fig.savefig(out_dir / f"category_label_feedback_final_{label}.png", dpi=200)
	
	# Distortion level analysis
	if "distortion" in df.columns:
		mean_dist = df.groupby(["label_rate", "distortion"], as_index=False)["accuracy"].mean()
		fig, ax = plt.subplots(figsize=(8, 5))
		sns.lineplot(
			data=mean_dist, x="distortion", y="accuracy", hue="label_rate", 
			marker="o", ax=ax, palette="viridis", linewidth=2
		)
		ax.set_title("Generalization Gradient: Accuracy vs Distortion\nPosner & Keele (1968)", fontsize=12)
		ax.set_ylabel("Classification Accuracy", fontsize=11)
		ax.set_xlabel("Distortion Level", fontsize=11)
		ax.set_ylim(0, 1.05)
		ax.legend(title='Label Rate', bbox_to_anchor=(1.05, 1), loc='upper left')
		fig.tight_layout()
		fig.savefig(out_dir / f"category_label_feedback_distortion_{label}.png", dpi=200)


def main(
	discrete_csv=Path(__file__).resolve().parent / "exp_category_label_feedback_discrete.csv",
	out_dir=Path(__file__).resolve().parent,
):
	sns.set_theme(style="whitegrid")
	generated = False

	if discrete_csv.exists():
		df_disc = pd.read_csv(discrete_csv)
		plot_label_feedback(df_disc, out_dir, label="discrete")
		generated = True

	if not generated:
		raise FileNotFoundError("No category-label feedback CSVs found; run experiments first.")


if __name__ == "__main__":
	main()
