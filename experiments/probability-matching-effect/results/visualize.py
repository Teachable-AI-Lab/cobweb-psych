from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def main(
	csv_path=Path(__file__).resolve().parent / "exp_probability_matching.csv",
	out_dir=Path(__file__).resolve().parent,
):
	sns.set_theme(style="whitegrid")
	df = pd.read_csv(csv_path)

	mean_share = df.groupby(["base_rate", "block"], as_index=False)["pred_share_B"].mean()
	fig, ax = plt.subplots(figsize=(7, 4))
	sns.lineplot(
		data=mean_share, x="block", y="pred_share_B", hue="base_rate", marker="o", ax=ax
	)
	ax.set_title("Probability matching over blocks")
	ax.set_ylim(0, 1)
	fig.tight_layout()
	fig.savefig(out_dir / "probability_matching_blocks.png", dpi=200)

	end_share = mean_share.groupby("base_rate", as_index=False).agg(
		pred_share_B=("pred_share_B", "last")
	)
	fig, ax = plt.subplots(figsize=(5, 4))
	sns.scatterplot(data=end_share, x="base_rate", y="pred_share_B", s=80, ax=ax)
	ax.plot([0.4, 0.85], [0.4, 0.85], linestyle="--", color="gray", label="match line")
	ax.set_xlim(0.45, 0.85)
	ax.set_ylim(0.45, 0.85)
	ax.legend()
	ax.set_title("End-state matching")
	fig.tight_layout()
	fig.savefig(out_dir / "probability_matching_final.png", dpi=200)


if __name__ == "__main__":
	main()
