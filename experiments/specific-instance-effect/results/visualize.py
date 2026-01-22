from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def main(
	csv_path=Path(__file__).resolve().parents[1] / "exp_specific_instance.csv",
	out_dir=Path(__file__).parent,
):
	sns.set_theme(style="whitegrid")
	df = pd.read_csv(csv_path)
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
	sns.barplot(data=mean_stim, x="stimulus", y="correct_prob", ax=ax)
	ax.set_title("Specific-instance leverage")
	ax.set_ylim(0, 1)
	fig.tight_layout()
	fig.savefig(out_dir / "specific_instance_bar.png", dpi=200)

	mean_block = df.groupby(["block", "stimulus"], as_index=False)["correct_prob"].mean()
	fig, ax = plt.subplots(figsize=(7, 4))
	sns.lineplot(
		data=mean_block, x="block", y="correct_prob", hue="stimulus", marker="o", ax=ax
	)
	ax.set_title("Trajectory of exemplar influence")
	ax.set_ylim(0, 1)
	fig.tight_layout()
	fig.savefig(out_dir / "specific_instance_blocks.png", dpi=200)


if __name__ == "__main__":
	main()
