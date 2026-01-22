from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def main(
	csv_path=Path(__file__).resolve().parents[1] / "exp_fan_effect.csv",
	out_dir=Path(__file__).parent,
):
	sns.set_theme(style="whitegrid")
	df = pd.read_csv(csv_path)

	mean_fan = df.groupby(["fan_size", "block"], as_index=False)["prob"].mean()
	fig, ax = plt.subplots(figsize=(7, 4))
	sns.lineplot(
		data=mean_fan, x="block", y="prob", hue="fan_size", marker="o", ax=ax
	)
	ax.set_title("Fan effect: fact retrieval probability")
	ax.set_ylim(0, 1)
	fig.tight_layout()
	fig.savefig(out_dir / "fan_effect_blocks.png", dpi=200)

	end_fan = mean_fan.groupby("fan_size", as_index=False).agg(prob_end=("prob", "last"))
	fig, ax = plt.subplots(figsize=(5, 4))
	sns.scatterplot(data=end_fan, x="fan_size", y="prob_end", s=80, ax=ax)
	ax.set_title("End-state by fan size")
	ax.set_ylim(0, 1)
	fig.tight_layout()
	fig.savefig(out_dir / "fan_effect_final.png", dpi=200)


if __name__ == "__main__":
	main()
