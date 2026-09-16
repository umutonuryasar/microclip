# Export finished seed-matrix curves from W&B and plot validation loss.
# Run where wandb is authenticated (Colab after wandb.login, or local).
import re, wandb, pandas as pd, matplotlib.pyplot as plt

PROJECT = "umutonuryasar-independent/microclip"
api = wandb.Api()

records = []
for r in api.runs(PROJECT):
    if r.state != "finished":
        continue
    m = re.match(
        r"(sigmoid|softmax)_b(\d+)_s(\d+)$", r.name
    )  # seed matrix only (skip anchor)
    if not m:
        continue
    loss, batch, seed = m.group(1), int(m.group(2)), int(m.group(3))
    try:
        h = r.history(keys=["epoch", "val/loss"], pandas=True)
    except Exception as e:
        print(f"skip {r.name}: {e}")
        continue
    h = h.dropna(subset=["val/loss", "epoch"])
    for _, row in h.iterrows():
        records.append(
            {
                "loss": loss,
                "batch": batch,
                "seed": seed,
                "epoch": int(row["epoch"]),
                "val_loss": float(row["val/loss"]),
            }
        )

df = pd.DataFrame(records)
df.to_csv("results/wandb_curves.csv", index=False)
print("seeds per group:\n", df.groupby(["loss", "batch"]).seed.nunique(), sep="")

fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharex=True)
colors = {512: "#1f77b4", 128: "#d62728"}
for ax, loss in zip(axes, ["sigmoid", "softmax"]):
    for batch in (512, 128):
        g = df[(df.loss == loss) & (df.batch == batch)]
        if g.empty:
            continue
        piv = g.pivot_table(index="epoch", columns="seed", values="val_loss")
        mean, std = piv.mean(axis=1), piv.std(axis=1, ddof=1)
        ax.plot(mean.index, mean.values, color=colors[batch], label=f"b{batch}")
        ax.fill_between(
            mean.index, mean - std, mean + std, color=colors[batch], alpha=0.2
        )
    ax.set_title(f"{loss} — validation loss")
    ax.set_xlabel("epoch")
    ax.legend()
axes[0].set_ylabel("val loss")
fig.suptitle(
    "Validation loss over training (3 seeds, mean +/- std). "
    "Losses differ across panels and are NOT comparable — see the retrieval table.",
    fontsize=9,
)
fig.tight_layout()
fig.savefig("results/training_curves.png", dpi=150, bbox_inches="tight")
print("wrote results/training_curves.png + results/wandb_curves.csv")
