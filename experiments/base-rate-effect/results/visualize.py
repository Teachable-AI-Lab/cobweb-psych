from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_base_rate(df: pd.DataFrame, out_dir: Path, label: str):
	"""Plot accuracy and B-share trajectories for a single dataset."""
	mean_acc = df.groupby(["base_rate", "block"], as_index=False)["accuracy"].mean()
	fig, ax = plt.subplots(figsize=(7, 4))
	sns.lineplot(
		data=mean_acc, x="block", y="accuracy", hue="base_rate", marker="o", ax=ax, palette="colorblind"
	)
	ax.set_title(f"Accuracy under unequal base rates ({label})")
	ax.set_ylim(0, 1)
	fig.tight_layout()
	fig.savefig(out_dir / f"base_rate_accuracy_{label}.png", dpi=200)

	mean_share = df.groupby(["base_rate", "block"], as_index=False)["pred_share_B"].mean()
	fig, ax = plt.subplots(figsize=(7, 4))
	sns.lineplot(
		data=mean_share, x="block", y="pred_share_B", hue="base_rate", marker="o", ax=ax, palette="colorblind"
	)
	ax.axhline(0.5, color="gray", linestyle="--", linewidth=1)
	ax.set_title(f"Predicted share for category B ({label})")
	ax.set_ylim(0, 1)
	fig.tight_layout()
	fig.savefig(out_dir / f"base_rate_share_{label}.png", dpi=200)


def main(
	continuous_csv=Path(__file__).resolve().parent / "exp_base_rate.csv",
	discrete_csv=Path(__file__).resolve().parent / "exp_base_rate_discrete.csv",
	out_dir=Path(__file__).resolve().parent,
):
	sns.set_theme(style="whitegrid")
	generated = False

	if continuous_csv.exists():
		df_cont = pd.read_csv(continuous_csv)
		plot_base_rate(df_cont, out_dir, label="continuous")
		generated = True

	if discrete_csv.exists():
		df_disc = pd.read_csv(discrete_csv)
		plot_base_rate(df_disc, out_dir, label="discrete")
		generated = True

	if not generated:
		raise FileNotFoundError("No base-rate CSVs found; run experiments first.")


if __name__ == "__main__":
	main()
