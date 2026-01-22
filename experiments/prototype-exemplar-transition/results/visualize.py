from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def main(
    csv_path=Path(__file__).resolve().parent / "exp_smith-minda_blocks10_nseeds5_epoch5.csv",
    out_dir=Path(__file__).parent,
):
    sns.set_theme(style="whitegrid")
    df = pd.read_csv(csv_path)
    df["correct_prob"] = df.apply(
        lambda r: r.pred_A if r.category == "A" else r.pred_B, axis=1
    )
    df["stimulus"] = df["stimulus"].astype(int)

    mean_block = df.groupby(["stimulus", "block"], as_index=False)["correct_prob"].mean()
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.lineplot(
        data=mean_block, x="block", y="correct_prob", hue="stimulus", marker="o", ax=ax
    )
    ax.set_title("Prototype–exemplar transition (per stimulus)")
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(out_dir / "smith_minda_by_stimulus.png", dpi=200)

    proto = [1, 8]
    near = [2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13]
    far = [14]
    bucket = []
    for stim in df["stimulus"].unique():
        if stim in proto:
            bucket.append((stim, "prototype"))
        elif stim in far:
            bucket.append((stim, "exception"))
        else:
            bucket.append((stim, "near"))
    bucket_map = dict(bucket)
    df["group"] = df["stimulus"].map(bucket_map)

    mean_group = df.groupby(["group", "block"], as_index=False)["correct_prob"].mean()
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.lineplot(
        data=mean_group, x="block", y="correct_prob", hue="group", marker="o", ax=ax
    )
    ax.set_title("Group trajectories: prototype vs near vs exception")
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(out_dir / "smith_minda_groups.png", dpi=200)


if __name__ == "__main__":
    main()