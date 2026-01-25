from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_continuous(df: pd.DataFrame, out_dir: Path, label: str):
	target = {
		"special": "A",
		"special_perturbed": "A",
		"typical_A": "A",
		"typical_B": "B",
		"boundary": "B",
	}
	df["correct_prob"] = df.apply(
		lambda r: r.pred_A if target[r.stimulus] == "A" else r.pred_B, axis=1
	)

	mean_stim = df.groupby("stimulus", as_index=False)["correct_prob"].mean()
	fig, ax = plt.subplots(figsize=(6, 4))
	sns.barplot(data=mean_stim, x="stimulus", y="correct_prob", ax=ax, palette="colorblind")
	ax.set_title(f"Specific-instance leverage ({label})")
	ax.set_ylim(0, 1)
	fig.tight_layout()
	fig.savefig(out_dir / f"specific_instance_bar_{label}.png", dpi=200)

	mean_block = df.groupby(["block", "stimulus"], as_index=False)["correct_prob"].mean()
	fig, ax = plt.subplots(figsize=(7, 4))
	sns.lineplot(
		data=mean_block, x="block", y="correct_prob", hue="stimulus", marker="o", ax=ax, palette="colorblind"
	)
	ax.set_title(f"Trajectory of exemplar influence ({label})")
	ax.set_ylim(0, 1)
	fig.tight_layout()
	fig.savefig(out_dir / f"specific_instance_blocks_{label}.png", dpi=200)


def plot_discrete(df: pd.DataFrame, out_dir: Path, label: str):
	# For discrete CSV, we have prob_A / prob_B; treat frequent A color as target A
	df = df.copy()
	target = {"freq_A": "A", "rare_A": "A", "rare_B": "B"}
	if "prob_A" in df.columns and "prob_B" in df.columns:
		df["correct_prob"] = df.apply(
			lambda r: r.prob_A if target.get(r.stimulus, "A") == "A" else r.prob_B, axis=1
		)

		mean_stim = df.groupby("stimulus", as_index=False)["correct_prob"].mean()
		fig, ax = plt.subplots(figsize=(6, 4))
		sns.barplot(data=mean_stim, x="stimulus", y="correct_prob", ax=ax, palette="colorblind")
		ax.set_title(f"Specific-instance leverage ({label})")
		ax.set_ylim(0, 1)
		fig.tight_layout()
		fig.savefig(out_dir / f"specific_instance_bar_{label}.png", dpi=200)

		mean_block = df.groupby(["block", "stimulus"], as_index=False)["correct_prob"].mean()
		fig, ax = plt.subplots(figsize=(7, 4))
		sns.lineplot(
			data=mean_block, x="block", y="correct_prob", hue="stimulus", marker="o", ax=ax, palette="colorblind"
		)
		ax.set_title(f"Trajectory of exemplar influence ({label})")
		ax.set_ylim(0, 1)
		fig.tight_layout()
		fig.savefig(out_dir / f"specific_instance_blocks_{label}.png", dpi=200)


def main(
	continuous_csv=Path(__file__).resolve().parent / "exp_specific_instance_continuous.csv",
	discrete_csv=Path(__file__).resolve().parent / "exp_specific_instance_discrete.csv",
	out_dir=Path(__file__).resolve().parent,
):
	sns.set_theme(style="whitegrid")
	generated = False

	if continuous_csv.exists():
		df_cont = pd.read_csv(continuous_csv)
		plot_continuous(df_cont, out_dir, label="continuous")
		generated = True

	if discrete_csv.exists():
		df_disc = pd.read_csv(discrete_csv)
		plot_discrete(df_disc, out_dir, label="discrete")
		generated = True

	if not generated:
		raise FileNotFoundError("No specific-instance CSVs found; run experiments first.")


if __name__ == "__main__":
	main()
