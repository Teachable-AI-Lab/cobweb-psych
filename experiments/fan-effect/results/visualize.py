from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_fan(df: pd.DataFrame, out_dir: Path, label: str):
	# prefer 'prob' column if present, else try 'mean_prob_true_fact' or 'prob' fallback
	if "prob" in df.columns:
		value_col = "prob"
	elif "mean_prob_true_fact" in df.columns:
		value_col = "mean_prob_true_fact"
	else:
		raise ValueError("No suitable probability column found in fan-effect CSV")

	mean_fan = df.groupby(["fan_size", "block"], as_index=False)[value_col].mean()
	fig, ax = plt.subplots(figsize=(7, 4))
	sns.lineplot(
		data=mean_fan, x="block", y=value_col, hue="fan_size", marker="o", ax=ax, palette="colorblind"
	)
	ax.set_title(f"Fan effect: fact retrieval probability ({label})")
	ax.set_ylim(0, 1)
	fig.tight_layout()
	fig.savefig(out_dir / f"fan_effect_blocks_{label}.png", dpi=200)

	end_fan = mean_fan.groupby("fan_size", as_index=False).agg(prob_end=(value_col, "last"))
	fig, ax = plt.subplots(figsize=(5, 4))
	sns.scatterplot(data=end_fan, x="fan_size", y="prob_end", s=80, ax=ax, palette="colorblind")
	ax.set_title(f"End-state by fan size ({label})")
	ax.set_ylim(0, 1)
	fig.tight_layout()
	fig.savefig(out_dir / f"fan_effect_final_{label}.png", dpi=200)


def main(
	continuous_csv=Path(__file__).resolve().parent / "exp_fan_effect_continuous.csv",
	discrete_csv=Path(__file__).resolve().parent / "exp_fan_effect_discrete.csv",
	out_dir=Path(__file__).resolve().parent,
):
	sns.set_theme(style="whitegrid")
	generated = False

	if continuous_csv.exists():
		df_cont = pd.read_csv(continuous_csv)
		plot_fan(df_cont, out_dir, label="continuous")
		generated = True

	if discrete_csv.exists():
		df_disc = pd.read_csv(discrete_csv)
		plot_fan(df_disc, out_dir, label="discrete")
		generated = True

	if not generated:
		raise FileNotFoundError("No fan-effect CSVs found; run experiments first.")


if __name__ == "__main__":
	main()
