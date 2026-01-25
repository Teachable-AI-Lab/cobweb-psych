from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_label_feedback(df: pd.DataFrame, out_dir: Path, label: str):
	mean_acc = df.groupby(["label_rate", "block"], as_index=False)["accuracy"].mean()
	fig, ax = plt.subplots(figsize=(7, 4))
	sns.lineplot(
		data=mean_acc, x="block", y="accuracy", hue="label_rate", marker="o", ax=ax, palette="colorblind"
	)
	ax.set_title(f"Impact of category labels on learning ({label})")
	ax.set_ylim(0, 1)
	fig.tight_layout()
	fig.savefig(out_dir / f"category_label_feedback_blocks_{label}.png", dpi=200)

	end_acc = mean_acc.groupby("label_rate", as_index=False).agg(
		accuracy_end=("accuracy", "last")
	)
	fig, ax = plt.subplots(figsize=(5, 4))
	sns.barplot(data=end_acc, x="label_rate", y="accuracy_end", ax=ax, palette="colorblind")
	ax.set_title(f"End-state accuracy vs label rate ({label})")
	ax.set_ylim(0, 1)
	fig.tight_layout()
	fig.savefig(out_dir / f"category_label_feedback_final_{label}.png", dpi=200)


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
