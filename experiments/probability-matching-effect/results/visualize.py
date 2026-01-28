from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def main(
	csv_path=Path(__file__).resolve().parent / "exp_probability_matching.csv",
	out_dir=Path(__file__).resolve().parent,
):
	sns.set_theme(style="whitegrid")
	if not csv_path.exists():
		print("CSV not found.")
		return
		
	df = pd.read_csv(csv_path)

	# Gluck & Bower Figure Style: P(A) response over blocks
	# Hue = Assigned Base Rate Condition
	
	mean_share = df.groupby(["condition_base_rate", "block"], as_index=False)["response_prob_A"].mean()
	
	fig, ax = plt.subplots(figsize=(8, 5))
	sns.lineplot(
		data=mean_share, x="block", y="response_prob_A", hue="condition_base_rate", 
		marker="o", palette="coolwarm", linewidth=2, ax=ax
	)
	
	# Add reference horizontal lines for the true base rates
	colors = sns.color_palette("coolwarm", n_colors=len(mean_share["condition_base_rate"].unique()))
	unique_brs = sorted(mean_share["condition_base_rate"].unique())
	
	for i, br in enumerate(unique_brs):
		ax.axhline(br, color=colors[i], linestyle="--", alpha=0.5, label=f"True {br}")
		
	ax.set_title("Probability Matching: Model Response vs Condition Base Rate\n(Gluck & Bower, 1988)", fontsize=12)
	ax.set_ylabel("Probability of Choosing Category A", fontsize=11)
	ax.set_xlabel("Training Block (25 trials/block)", fontsize=11)
	ax.set_ylim(0, 1.05)
	
	# Clean legend
	handles, labels = ax.get_legend_handles_labels()
	# remove duplicates if any or just title
	ax.legend(title="True Base Rate P(A)")
	
	fig.tight_layout()
	fig.savefig(out_dir / "probability_matching_blocks.png", dpi=200)

	# End-state matching scatter
	# Plot actual base rate condition vs responding base rate
	end_block = mean_share["block"].max()
	end_share = mean_share[mean_share["block"] == end_block]
	
	fig, ax = plt.subplots(figsize=(6, 6))
	sns.scatterplot(
		data=end_share, x="condition_base_rate", y="response_prob_A", 
		s=100, color="black", ax=ax, label="Model Prediction"
	)
	
	# Identity line (Perfect Matching)
	min_val = min(unique_brs) - 0.1
	max_val = max(unique_brs) + 0.1
	ax.plot([min_val, max_val], [min_val, max_val], linestyle="--", color="gray", label="Perfect Matching")
	
	# Maximization Line? 
	# If > 0.5, predict 1.0. If < 0.5, predict 0.0.
	# We can draw piecewise line for maximization strategy comparison
	# x < 0.5 -> y=0. x > 0.5 -> y=1.
	ax.plot([min_val, 0.5, 0.5, max_val], [0, 0, 1, 1], linestyle=":", color="red", alpha=0.5, label="Maximization")
	
	ax.set_xlim(0, 1)
	ax.set_ylim(0, 1)
	ax.set_xlabel("True Probability P(A)", fontsize=11)
	ax.set_ylabel("Observed Response Probability P(A)", fontsize=11)
	ax.legend()
	ax.set_title("End-State: Matching vs Maximization")
	
	fig.tight_layout()
	fig.savefig(out_dir / "probability_matching_final.png", dpi=200)


if __name__ == "__main__":
	main()
