from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def main(
	csv_path=Path(__file__).resolve().parents[1] / "exp_base_rate.csv",
	out_dir=Path(__file__).parent,
):
	sns.set_theme(style="whitegrid")
	df = pd.read_csv(csv_path)

	mean_acc = df.groupby(["base_rate", "block"], as_index=False)["accuracy"].mean()
	fig, ax = plt.subplots(figsize=(7, 4))
	sns.lineplot(
		data=mean_acc, x="block", y="accuracy", hue="base_rate", marker="o", ax=ax
	)
	ax.set_title("Accuracy under unequal base rates")
	ax.set_ylim(0, 1)
	fig.tight_layout()
	fig.savefig(out_dir / "base_rate_accuracy.png", dpi=200)

	mean_share = df.groupby(["base_rate", "block"], as_index=False)["pred_share_B"].mean()
	fig, ax = plt.subplots(figsize=(7, 4))
	sns.lineplot(
		data=mean_share, x="block", y="pred_share_B", hue="base_rate", marker="o", ax=ax
	)
	ax.axhline(0.5, color="gray", linestyle="--", linewidth=1)
	ax.set_title("Predicted share for category B")
	ax.set_ylim(0, 1)
	fig.tight_layout()
	fig.savefig(out_dir / "base_rate_share.png", dpi=200)


if __name__ == "__main__":
	main()
