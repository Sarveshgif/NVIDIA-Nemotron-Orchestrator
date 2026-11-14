"""Presentation figures from real outputs (data/results/fig_*.png). Palette validated with the dataviz checker
(silage #008f86, grain #b7801a: lightness, chroma, CVD and normal-vision separation all pass on #fcfcfb)."""
import json, glob, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from common import ROOT, YEAR

from multiyear import RESULTS as RES
FEAT = ROOT / "features"
SIL, GRA, INK, MUTED, GRID, BG = "#008f86", "#b7801a", "#1b2620", "#6b746d", "#e7e3d7", "#fcfcfb"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG})
M = json.load(open(RES / "metrics.json"))
lab = pd.read_parquet(FEAT / "labels.parquet")[["field_id", "label"]]
spl = pd.read_csv(FEAT / "splits.csv")

# 1. Seasonal NDVI by declared class (all eligible fields, half-month medians of observed values only)
f = pd.read_parquet(FEAT / "features.parquet").merge(lab)
cols = [c for c in f.columns if c.startswith("ndvi_hm")]
x = np.arange(len(cols))
fig, ax = plt.subplots(figsize=(8, 3.6))
for cls, col in (("silage", SIL), ("grain", GRA)):
    g = f[f.label == cls][cols]
    q1, med, q3 = g.quantile(0.25), g.median(), g.quantile(0.75)
    ax.fill_between(x, q1, q3, color=col, alpha=0.15, linewidth=0)
    ax.plot(x, med, color=col, lw=2)
    ax.text(x[-1] + 0.3, med.iloc[-1], f"{cls} (n={len(g)})", color=INK, va="center", fontsize=9)
    ax.plot([], [], color=col, lw=2, label=cls)
ax.set_xticks(x[::2], [pd.Timestamp(f"{YEAR}-{c[7:9]}-01").strftime("%b") for c in cols[::2]])
ax.set_ylabel("NDVI (field median)"); ax.set_ylim(0, 1); ax.grid(axis="y", color=GRID, lw=0.8); ax.set_xlim(0, len(cols) + 3)
ax.legend(frameon=False, loc="upper left")
ax.set_title(f"Netherlands {YEAR}: declared silage vs grain maize, median and IQR of clear looks", loc="left", fontsize=10, color=INK)
plt.tight_layout(); plt.savefig(RES / "fig_ndvi_by_class.png", dpi=160); plt.close()

# 2. Model comparison on identical test fields (balanced accuracy; grain recall in a separate panel - no dual axis)
names = [("majority_silage", "Always silage"), ("rules_original_weak", "Rules v1 (Tulare)"), ("rules_adapted_weak", "Rules, adapted"),
         ("rf_s1_only", "RF · S1 only"), ("rf_s2_summary", "RF · S2"), ("rf_s2_summary_patch", "RF · S2 + zones"), ("rf_s2_s1", "RF · S2 + zones + S1")]
deep = {}
for p in glob.glob(str(RES / "deep_*_seed*.json")):
    d = json.load(open(p)); deep.setdefault(d["variant"], []).append(d["test"])
dl = {"s2": "CNN→Transformer · S2", "s2s1": "CNN→Transformer · S2+S1", "s2s1_nocnn": "Transformer · S2+S1 (no CNN)"}
rows = [(lbl, M[k]["balanced_accuracy"], M[k]["recall_grain"], M[k]["coverage"], None) for k, lbl in names if k in M]
rows += [(dl.get(v, v), np.mean([t["balanced_accuracy"] for t in ts]), np.mean([t["recall_grain"] for t in ts]), 1.0,
          np.std([t["balanced_accuracy"] for t in ts])) for v, ts in deep.items()]
fig, axs = plt.subplots(1, 2, figsize=(10, 0.42 * len(rows) + 1.2), sharey=True)
y = np.arange(len(rows))[::-1]
for ax, j, title in ((axs[0], 1, "Balanced accuracy"), (axs[1], 2, "Grain recall")):
    v = [r[j] if np.isfinite(r[j]) else 0 for r in rows]
    ax.barh(y, v, color=INK, height=0.55)
    for yi, vi, r in zip(y, v, rows):
        txt = f"{vi:.2f}" + (f" ±{r[4]:.2f}" if (j == 1 and r[4]) else "") + (f"  (cov {r[3]:.0%})" if r[3] < 0.999 else "")
        ax.text(vi + 0.01, yi, txt, va="center", fontsize=8, color=INK)
    ax.set_xlim(0, 1.15); ax.set_title(title, loc="left", fontsize=10, color=INK); ax.grid(axis="x", color=GRID, lw=0.8)
    if j == 1: ax.axvline(0.5, color=MUTED, lw=1, ls=":")
axs[0].set_yticks(y, [r[0] for r in rows])
fig.suptitle(f"Held-out test regions ({M['rf_s2_s1']['n']} fields, {M['rf_s2_s1']['n_grain']} grain). Rules: on fields they label.", x=0.01, ha="left", fontsize=10)
plt.tight_layout(); plt.savefig(RES / "fig_model_comparison.png", dpi=160); plt.close()

# 3. Review-band trade-off for the console model (test fields)
pr = pd.read_parquet(RES / "predictions_all.parquet"); pr = pr[pr.year == YEAR].drop(columns="year").merge(lab).merge(spl)
pr = pr[pr.split == "test"]
yv = (pr.label == "silage").values; p = pr.p_silage.values
t = M[M["selected_model"] + "_config"]["threshold"]
conf = np.abs(p - t)
cov, acc = [], []
for q in np.linspace(0, 0.9, 19):
    keep = conf >= np.quantile(conf, q)
    pred = p[keep] >= t
    rs = (pred & yv[keep]).sum() / max(1, yv[keep].sum()); rg = (~pred & ~yv[keep]).sum() / max(1, (~yv[keep]).sum())
    cov.append(keep.mean()); acc.append((rs + rg) / 2)
fig, ax = plt.subplots(figsize=(5.5, 3.4))
ax.plot(np.array(cov) * 100, acc, color=INK, lw=2, marker="o", ms=4)
ax.set_xlabel("Fields decided automatically (%)"); ax.set_ylabel("Balanced accuracy on those fields")
ax.invert_xaxis(); ax.grid(color=GRID, lw=0.8)
ax.set_title("Routing the least certain fields to review (test)", loc="left", fontsize=10, color=INK)
plt.tight_layout(); plt.savefig(RES / "fig_review_tradeoff.png", dpi=160); plt.close()
pd.DataFrame({"auto_decided_share": cov, "balanced_accuracy": acc}).to_csv(RES / "review_tradeoff_test.csv", index=False)
print("figures written")
