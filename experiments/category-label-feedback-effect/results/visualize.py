from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def main(
	csv_path=Path(__file__).resolve().parents[1] / "exp_category_label_feedback.csv",
	out_dir=Path(__file__).parent,
):
	sns.set_theme(style="whitegrid")
	df = pd.read_csv(csv_path)

	mean_acc = df.groupby(["label_rate", "block"], as_index=False)["accuracy"].mean()
	fig, ax = plt.subplots(figsize=(7, 4))
	sns.lineplot(
		data=mean_acc, x="block", y="accuracy", hue="label_rate", marker="o", ax=ax
	)
	ax.set_title("Impact of category labels on learning")
	ax.set_ylim(0, 1)
	fig.tight_layout()
	fig.savefig(out_dir / "category_label_feedback_blocks.png", dpi=200)

	end_acc = mean_acc.groupby("label_rate", as_index=False).agg(
		accuracy_end=("accuracy", "last")
	)
	fig, ax = plt.subplots(figsize=(5, 4))
	sns.barplot(data=end_acc, x="label_rate", y="accuracy_end", ax=ax)
	ax.set_title("End-state accuracy vs label rate")
	ax.set_ylim(0, 1)
	fig.tight_layout()
	fig.savefig(out_dir / "category_label_feedback_final.png", dpi=200)


if __name__ == "__main__":
	main()
