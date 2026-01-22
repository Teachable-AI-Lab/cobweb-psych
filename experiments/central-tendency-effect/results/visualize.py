from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def main(
    csv_path=Path(__file__).resolve().parents[1] / "exp_central_tendency_continuous.csv",
    out_dir=Path(__file__).parent,
):
    sns.set_theme(style="whitegrid")
    df = pd.read_csv(csv_path)
    df["correct_prob"] = df.apply(
        lambda r: r.pred_A if r.category == "A" else r.pred_B, axis=1
    )

    mean_dist = df.groupby("distance", as_index=False)["correct_prob"].mean()
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.lineplot(data=mean_dist, x="distance", y="correct_prob", marker="o", ax=ax)
    ax.set_title("Central Tendency: Mean correct prob vs distance")
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(out_dir / "central_tendency_distance.png", dpi=200)

    mean_block = (
        df.groupby(["block", "distance"], as_index=False)["correct_prob"].mean()
    )
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.lineplot(
        data=mean_block, x="block", y="correct_prob", hue="distance", marker="o", ax=ax
    )
    ax.set_title("Trajectory by prototype distance")
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(out_dir / "central_tendency_blocks.png", dpi=200)


if __name__ == "__main__":
    main()