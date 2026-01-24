from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def main(
	csv_path=Path(__file__).resolve().parent / "exp_basic_level.csv",
	out_dir=Path(__file__).resolve().parent,
):
	sns.set_theme(style="whitegrid")
	df = pd.read_csv(csv_path)

	mean_acc = df.groupby(["level", "block"], as_index=False)["accuracy"].mean()
	fig, ax = plt.subplots(figsize=(7, 4))
	sns.lineplot(
		data=mean_acc, x="block", y="accuracy", hue="level", marker="o", ax=ax
	)
	ax.set_title("Basic-level advantage over blocks")
	ax.set_ylim(0, 1)
	fig.tight_layout()
	fig.savefig(out_dir / "basic_level_blocks.png", dpi=200)

	end_acc = mean_acc.groupby("level", as_index=False).agg(
		accuracy_end=("accuracy", "last")
	)
	fig, ax = plt.subplots(figsize=(5, 4))
	sns.barplot(data=end_acc, x="level", y="accuracy_end", ax=ax)
	ax.set_title("End-state by level")
	ax.set_ylim(0, 1)
	fig.tight_layout()
	fig.savefig(out_dir / "basic_level_final.png", dpi=200)


if __name__ == "__main__":
	main()
