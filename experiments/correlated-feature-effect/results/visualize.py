from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_correlated(df: pd.DataFrame, out_dir: Path, label: str):
	mean_block = df.groupby(["structure", "block"], as_index=False)["accuracy"].mean()
	fig, ax = plt.subplots(figsize=(7, 4))
	sns.lineplot(
		data=mean_block, x="block", y="accuracy", hue="structure", marker="o", ax=ax, palette="colorblind"
	)
	ax.set_title(f"Correlated feature (XOR) difficulty ({label})")
	ax.set_ylim(0, 1)
	fig.tight_layout()
	fig.savefig(out_dir / f"correlated_feature_blocks_{label}.png", dpi=200)

	final_block = mean_block.groupby("structure", as_index=False).agg(
		accuracy_end=("accuracy", "last")
	)
	fig, ax = plt.subplots(figsize=(5, 4))
	sns.barplot(data=final_block, x="structure", y="accuracy_end", ax=ax, palette="colorblind")
	ax.set_title(f"End-state accuracy ({label})")
	ax.set_ylim(0, 1)
	fig.tight_layout()
	fig.savefig(out_dir / f"correlated_feature_final_{label}.png", dpi=200)


def main(
	discrete_csv=Path(__file__).resolve().parent / "exp_correlated_feature_discrete.csv",
	out_dir=Path(__file__).resolve().parent,
):
	sns.set_theme(style="whitegrid")

	if discrete_csv.exists():
		df_disc = pd.read_csv(discrete_csv)
		plot_correlated(df_disc, out_dir, label="discrete")
	else:
		raise FileNotFoundError("No correlated-feature discrete CSV found; run experiments first.")


if __name__ == "__main__":
	main()
