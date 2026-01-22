from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def main(
	csv_path=Path(__file__).resolve().parents[1] / "exp_correlated_feature.csv",
	out_dir=Path(__file__).parent,
):
	sns.set_theme(style="whitegrid")
	df = pd.read_csv(csv_path)

	mean_block = df.groupby(["structure", "block"], as_index=False)["accuracy"].mean()
	fig, ax = plt.subplots(figsize=(7, 4))
	sns.lineplot(
		data=mean_block, x="block", y="accuracy", hue="structure", marker="o", ax=ax
	)
	ax.set_title("Correlated feature (XOR) difficulty")
	ax.set_ylim(0, 1)
	fig.tight_layout()
	fig.savefig(out_dir / "correlated_feature_blocks.png", dpi=200)

	final_block = mean_block.groupby("structure", as_index=False).agg(
		accuracy_end=("accuracy", "last")
	)
	fig, ax = plt.subplots(figsize=(5, 4))
	sns.barplot(data=final_block, x="structure", y="accuracy_end", ax=ax)
	ax.set_title("End-state accuracy")
	ax.set_ylim(0, 1)
	fig.tight_layout()
	fig.savefig(out_dir / "correlated_feature_final.png", dpi=200)


if __name__ == "__main__":
	main()
