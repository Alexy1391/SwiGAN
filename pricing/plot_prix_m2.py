import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

s = pd.read_csv("pricing/output/maisons_dep08.csv", usecols=["dvf_prix_m2"])["dvf_prix_m2"].dropna()
p99 = s.quantile(0.99)
shown = s[s <= p99]

plt.rcParams.update({"font.size": 10, "axes.edgecolor": "#c9c8c4"})
fig, ax = plt.subplots(figsize=(9, 5), facecolor="#fcfcfb")
ax.set_facecolor("#fcfcfb")
ax.hist(shown, bins=60, color="#2a78d6", edgecolor="#fcfcfb", linewidth=1)
med = s.median()
ax.axvline(med, color="#52514e", lw=1.2, ls="--")
ax.text(med, ax.get_ylim()[1] * 0.97, f"  médiane {med:,.0f} €/m²", color="#0b0b0b", va="top")
ax.set_title(f"Distribution du prix au m² (DVF) – maisons, Ardennes (08)\n"
             f"{len(s):,} maisons renseignées sur 98 773 ; {len(s)-len(shown)} valeurs > p99 ({p99:,.0f} €/m²) masquées",
             loc="left", color="#0b0b0b", fontsize=11)
ax.set_xlabel("dvf_prix_m2 (€/m²)", color="#52514e")
ax.set_ylabel("Nombre de maisons", color="#52514e")
ax.grid(axis="y", color="#e6e5e1", lw=0.8); ax.set_axisbelow(True)
for side in ("top", "right"): ax.spines[side].set_visible(False)
ax.tick_params(colors="#52514e")
fig.tight_layout()
fig.savefig("pricing/output/dist_prix_m2_dep08.png", dpi=150)
print(s.describe(percentiles=[.05, .25, .5, .75, .95, .99]).round(0))
