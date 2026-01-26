from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_continuous(df: pd.DataFrame, out_dir: Path, label: str):
	"""
	Plot specific-instance effect for continuous version (Nosofsky, 1988).
	
	Original graph: Typicality/Probability vs Similarity to Exemplar.
	Lines: High-Frequency vs Low-Frequency.
	
	Our Approximation:
	X-axis: Distance to Frequent Exemplar (A1).
	Y-axis: P(Category A).
	"""
	# Reconstruct coordinates to calculate distances
	# Stimuli definitions from exp_specific_instance_continuous.py
	coords = {
		"A1": (1.0, 2.0), # Frequent
		"A2": (2.0, 1.0),
		"A3": (2.0, 2.0),
		"A4": (2.0, 3.0),
		"A5": (3.0, 2.0),
		"A6": (3.0, 3.0),
		# Category B (far away, likely p(A)~0)
		"B1": (5.0, 6.0),
	}
	
	def get_dist(stim):
		if stim not in coords: return 99.0
		val = coords[stim]
		# Distance to A1 (1.0, 2.0)
		return ((val[0]-1.0)**2 + (val[1]-2.0)**2)**0.5

	df["dist_to_freq"] = df["stimulus"].apply(get_dist)
	
	# Filter to Category A items only for the main curve
	df_A = df[df["category"] == "A"].copy()
	
	# Mean P(A) per stimulus across blocks (or final block)
	# Nosofsky usually shows performance after learning. Let's take the last epoch/block.
	df_final = df_A[df_A["epoch"] == df_A["epoch"].max()].copy()
	
	mean_probs = df_final.groupby(["stimulus", "dist_to_freq"], as_index=False)["prob_A"].mean()
	# Sort by distance
	mean_probs = mean_probs.sort_values("dist_to_freq")

	fig, ax = plt.subplots(figsize=(6, 5))
	
	# Plot curve: Distance vs P(A)
	sns.lineplot(data=mean_probs, x="dist_to_freq", y="prob_A", marker="o", ax=ax, color='blue', label="Exemplars")
	
	# Highlight Frequent Exemplar
	freq = mean_probs[mean_probs["stimulus"] == "A1"]
	ax.scatter(freq.dist_to_freq, freq.prob_A, s=150, color='red', zorder=5, label="High Freq (A1)")
	
	ax.set_title("Specific-Instance Effect\n(Nosofsky, 1988)", fontsize=12)
	ax.set_ylabel("Probability of Category A", fontsize=11)
	ax.set_xlabel("Distance from Frequent Exemplar (A1)", fontsize=11)
	ax.set_ylim(0, 1.05)
	ax.legend()
	
	fig.tight_layout()
	fig.savefig(out_dir / f"specific_instance_curve_{label}.png", dpi=200)

def main(
	continuous_csv=Path(__file__).resolve().parent / "exp_specific_instance_continuous.csv",
	out_dir=Path(__file__).resolve().parent,
):
	sns.set_theme(style="whitegrid")
	if continuous_csv.exists():
		df_cont = pd.read_csv(continuous_csv)
		plot_continuous(df_cont, out_dir, label="continuous")
	else:
		print(f"Continuous CSV not found at {continuous_csv}")

if __name__ == "__main__":
	main()
