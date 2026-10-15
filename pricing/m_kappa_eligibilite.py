#!/usr/bin/env python3
"""m et κ du prior bêta, estimés sur l'éligibilité climatique seule (circulaire 2024).

La demande du maire est abandonnée : une commune est éligible CatNat si et
seulement si son SWI remplit le critère météorologique de la circulaire
IOME2322937C (section 6.1 de l'article, brochure Météo-France). X_it vaut 1 si
la commune i est éligible l'année t, et

    θ_i ~ Bêta(mκ, (1−m)κ),   X_it | θ_i ~ Bernoulli(θ_i)   indépendants en t,

si bien que S_i = Σ_t X_it suit une bêta-binomiale :

    E[S_i]   = T m
    Var(S_i) = T m(1−m) (1 + (T−1)/(κ+1))

m est estimé par la part moyenne de communes éligibles, κ par la méthode des
moments sur la variance de S_i entre communes :

    r = Var(S_i) / (T m(1−m)),     κ = (T−1)/(r−1) − 1        (r > 1 requis)

LE PIÈGE : L'EFFET ANNÉE
------------------------
Le modèle tient les années pour interchangeables. Or l'éligibilité est d'abord
un événement climatique régional : en 2018 presque toutes les communes du Grand
Est le sont, en 2021 aucune. Avec m_t proche de 0 ou de 1, S_i varie à peine
d'une commune à l'autre, Var(S_i) tombe sous T m(1−m), r < 1, et la formule
rend un κ négatif — aucune bêta-binomiale ne produit de sous-dispersion.

Le script rend donc aussi la généralisation à m_t variable. Avec
v_t = m_t(1 − m_t) et ρ = 1/(κ+1) la corrélation intra-commune entre deux
années :

    Var(S_i) = Σ_t v_t  +  ρ Σ_{s≠t} √(v_s v_t)

qui redonne exactement la formule ci-dessus quand m_t = m pour tout t. Σ_t v_t
est la variance qu'aurait S_i sans aucune hétérogénéité communale ; ρ mesure ce
qui la dépasse.

LES DEUX JEUX DE DONNÉES
------------------------
calculé   seuils 1 et 2 de la brochure sur le SWI uniforme observé, par
          `confronter_eligibilite_catnat.py` (output/eligibilite_c12_vs_
          decisions_grand_est.csv). Le seuil 3 (communes limitrophes) n'est pas
          implémenté : c'est une borne inférieure de l'éligibilité.
reconnu   reconnaissances publiées au Journal officiel (decisions_catnat_<an>.csv),
          rapprochées des codes INSEE 2026 par le même script. Elles mêlent
          critère, demande du maire et critère géotechnique ; jusqu'en 2023
          elles relèvent en outre de la circulaire de 2019.

Sortie : output/m_kappa_grand_est.csv (une ligne par année)

Usage : python3 m_kappa_eligibilite.py [--annees 2018 2025]
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "output"
ELIGIBILITE = OUT / "eligibilite_c12_vs_decisions_grand_est.csv"
SORTIE = OUT / "m_kappa_grand_est.csv"


def matrice(tab: pd.DataFrame, colonne: str) -> pd.DataFrame:
    """Communes × années, 1 si la colonne est vraie (valeur manquante = 0)."""
    x = tab.assign(v=tab[colonne].eq(True).astype(float))
    return x.pivot(index="code_commune_insee", columns="annee", values="v").fillna(0.0)


def moments(x: pd.DataFrame) -> dict:
    """m, Var(S_i) et κ sous la formule à m constant, puis à m_t variable."""
    T = x.shape[1]
    s = x.sum(axis=1)
    m = s.mean() / T
    var_s = s.var(ddof=1)
    r = var_s / (T * m * (1 - m))
    kappa = (T - 1) / (r - 1) - 1 if r > 1 else np.nan

    v = (x.mean(axis=0) * (1 - x.mean(axis=0))).to_numpy()
    croise = np.sqrt(np.outer(v, v)).sum() - v.sum()
    rho = (var_s - v.sum()) / croise
    return {"T": T, "m": m, "var_s": var_s, "var_binom": T * m * (1 - m), "r": r,
            "kappa": kappa, "var_sans_heterogeneite": v.sum(), "rho": rho,
            "kappa_annee": 1 / rho - 1 if rho > 0 else np.nan}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--annees", type=int, nargs=2, default=[2018, 2025],
                   metavar=("DEBUT", "FIN"))
    p.add_argument("--csv", type=Path, default=SORTIE)
    a = p.parse_args()

    if not ELIGIBILITE.exists():
        raise SystemExit(f"{ELIGIBILITE} introuvable : lancer d'abord "
                         f"confronter_eligibilite_catnat.py")
    tab = pd.read_csv(ELIGIBILITE, dtype={"code_commune_insee": str})
    tab = tab[tab["annee"].between(*a.annees)]

    # --- m_t année par année -------------------------------------------------
    an = (tab.groupby("annee")
             .agg(communes=("eligible_c12", "size"),
                  eligibles=("eligible_c12", "sum"),
                  seuil1=("seuil1", "sum"), seuil2=("seuil2", "sum"),
                  demandes=("demande", "sum"),
                  reconnues=("reconnue", lambda s: s.eq(True).sum())))
    an["m_calcule"] = an["eligibles"] / an["communes"]
    an["m_reel"] = an["reconnues"] / an["communes"]
    an["taux_demande"] = an["demandes"] / an["communes"]

    # Parmi les seules communes qui ont demandé, le filtre de la demande tombe :
    # reste l'écart entre le critère calculé et la décision.
    dem = tab[tab["demande"]]
    an["eligibles_parmi_demandes"] = dem.groupby("annee")["eligible_c12"].mean()
    an["reconnues_parmi_demandes"] = dem.groupby("annee")["reconnue"].apply(
        lambda s: s.eq(True).mean())

    pd.set_option("display.width", 160)
    print(f"Grand Est, {an['communes'].iloc[0]:,} communes, "
          f"{a.annees[0]}-{a.annees[1]}\n")
    print("m_t = part des communes éligibles l'année t")
    print(an[["communes", "eligibles", "seuil1", "seuil2", "m_calcule",
              "reconnues", "m_reel", "demandes", "taux_demande"]]
          .to_string(float_format=lambda v: f"{v:.4f}"))
    print("\nParmi les communes ayant demandé (le filtre de la demande retiré)")
    print(an[["demandes", "eligibles_parmi_demandes", "reconnues_parmi_demandes"]]
          .to_string(float_format=lambda v: f"{v:.3f}"))

    # --- κ par la variance de S_i --------------------------------------------
    print("\nκ par la variance de S_i = Σ_t X_it entre communes")
    lignes = {}
    for nom, col in [("calculé", "eligible_c12"), ("reconnu", "reconnue")]:
        x = matrice(tab, col)
        mo = moments(x)
        lignes[nom] = mo
        dist = x.sum(axis=1).value_counts().sort_index()
        print(f"\n  {nom} : T = {mo['T']}, m = {mo['m']:.4f}")
        print(f"    S_i            : " + ", ".join(f"{int(k)}→{v}" for k, v in dist.items()))
        print(f"    Var(S_i)       = {mo['var_s']:.4f}")
        print(f"    T m(1−m)       = {mo['var_binom']:.4f}   r = {mo['r']:.3f}")
        if mo["r"] > 1:
            print(f"    κ (m constant) = {mo['kappa']:.3f}")
        else:
            print(f"    κ (m constant) : aucun — r < 1, sous-dispersion "
                  f"(la formule donnerait {(mo['T'] - 1) / (mo['r'] - 1) - 1:.2f})")
        print(f"    Σ_t m_t(1−m_t) = {mo['var_sans_heterogeneite']:.4f}   "
              f"ρ = {mo['rho']:.4f}")
        print(f"    κ (m_t variable) = "
              + (f"{mo['kappa_annee']:.2f}" if mo["rho"] > 0 else "aucun (ρ ≤ 0)"))

    an.to_csv(a.csv)
    pd.DataFrame(lignes).T.to_csv(a.csv.with_name(a.csv.stem + "_kappa.csv"))
    print(f"\nécrit dans {a.csv} et {a.csv.with_name(a.csv.stem + '_kappa.csv')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
