#!/usr/bin/env python3
"""Pouvoir prédictif des variables BDNB sur la valeur foncière (DVF).

Objectif : savoir si les variables disponibles pour *toutes* les maisons
suffisent à imputer une valeur aux maisons sans transaction, et mesurer la part
qui revient à la localisation communale.

Point de méthode — les colonnes `dvf_*` ne sont pas des variables explicatives :

    dvf_prix_m2      = dvf_valeur_fonciere / surface  -> fuite directe de la cible
    dvf_surface_bati = surface du bien muté           -> connue seulement si vente

Leur absence sur les maisons non transactées n'est donc pas un handicap : elles
n'auraient de toute façon pas pu servir. Le jeu de variables légitime se limite
au bâti, au foncier et à la localisation, tous connus pour l'ensemble du parc.

Deux régimes d'évaluation :

  aléatoire   les maisons de test sont dans des communes vues à l'entraînement
              -> le cas de déploiement réel (toutes les communes du 08 ont au
                 moins une transaction)
  par commune des communes entières sont retirées de l'entraînement
              -> le cas pessimiste, borne basse de généralisation spatiale
"""

import argparse
import warnings
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.model_selection import GroupKFold, KFold

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent
RNG = 0

NUM_STRUCT = ["nb_niveau", "annee_construction", "surface_sol_m2",
              "s_sol_constructions_m2", "nb_constructions", "hauteur_max_m",
              "nb_logements", "s_parcelle", "nb_pdl", "distance_monument_historique"]
CAT_STRUCT = ["materiau_mur", "materiau_toit", "alea_argile"]


def load(dep: str) -> pd.DataFrame:
    con = duckdb.connect()
    df = con.execute(f"""
        SELECT * FROM read_csv('{ROOT}/output/features_dep{dep}.csv', header=true)
        WHERE dvf_mutation_simple
          AND dvf_valeur_fonciere BETWEEN 5000 AND 1000000
    """).df()
    df["y"] = np.log(df["dvf_valeur_fonciere"])
    # Ratio d'emprise au sol sur parcelle : densité d'occupation du terrain,
    # plus informative que les deux surfaces prises séparément.
    df["ratio_emprise_parcelle"] = df["surface_sol_m2"] / df["s_parcelle"].replace(0, np.nan)
    df["annee_mutation"] = pd.to_datetime(df["dvf_date_mutation"], errors="coerce").dt.year
    return df


def target_encode(tr_key, tr_y, te_key, prior, m=10.0):
    """Encodage par la moyenne de la cible, lissé vers la moyenne globale.

    Le lissage (m pseudo-observations) évite qu'une commune à 2 transactions
    reçoive une estimation aussi confiante qu'une commune à 400.
    """
    g = pd.DataFrame({"k": tr_key, "y": tr_y}).groupby("k")["y"].agg(["mean", "count"])
    enc = (g["mean"] * g["count"] + prior * m) / (g["count"] + m)
    return te_key.map(enc).fillna(prior).values


def build_matrix(tr, te, feats_num, feats_cat, geo_cols):
    """Assemble X_train / X_test ; les colonnes géographiques sont encodées
    hors échantillon (calculées sur le seul pli d'entraînement)."""
    prior = tr["y"].mean()
    Xtr = tr[feats_num].copy()
    Xte = te[feats_num].copy()
    for c in feats_cat:
        cats = tr[c].astype("category").cat.categories
        Xtr[c] = pd.Categorical(tr[c], categories=cats).codes.astype(float)
        Xte[c] = pd.Categorical(te[c], categories=cats).codes.astype(float)
        Xtr.loc[Xtr[c] < 0, c] = np.nan
        Xte.loc[Xte[c] < 0, c] = np.nan
    for c in geo_cols:
        # encodage cross-fitté à l'intérieur du pli d'entraînement, pour que le
        # modèle ne voie jamais un encodage construit sur ses propres cibles
        oof = np.full(len(tr), np.nan)
        for a, b in KFold(5, shuffle=True, random_state=RNG).split(tr):
            oof[b] = target_encode(tr[c].iloc[a], tr["y"].iloc[a], tr[c].iloc[b], prior)
        Xtr[f"te_{c}"] = oof
        Xte[f"te_{c}"] = target_encode(tr[c], tr["y"], te[c], prior)
    return Xtr, Xte


def evaluate(df, feats_num, feats_cat, geo_cols, splitter, groups=None, label=""):
    r2s, medape, preds = [], [], np.full(len(df), np.nan)
    it = splitter.split(df, groups=groups) if groups is not None else splitter.split(df)
    for a, b in it:
        tr, te = df.iloc[a], df.iloc[b]
        Xtr, Xte = build_matrix(tr, te, feats_num, feats_cat, geo_cols)
        mdl = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.07,
                                            max_depth=6, min_samples_leaf=25,
                                            l2_regularization=1.0, random_state=RNG)
        mdl.fit(Xtr, tr["y"])
        p = mdl.predict(Xte)
        preds[b] = p
        r2s.append(1 - ((te["y"] - p) ** 2).sum() / ((te["y"] - tr["y"].mean()) ** 2).sum())
        medape.append(np.median(np.abs(np.exp(p) - te["dvf_valeur_fonciere"])
                                / te["dvf_valeur_fonciere"]) * 100)
    print(f"  {label:44s} R²(log) {np.mean(r2s):.3f}   erreur médiane {np.mean(medape):5.1f} %")
    return np.mean(r2s), preds


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dep", default="08")
    a = ap.parse_args()
    df = load(a.dep).reset_index(drop=True)
    print(f"\ndép {a.dep} — {len(df):,} maisons avec transaction simple exploitable "
          f"({df['code_commune_insee'].nunique()} communes)\n")

    extra = ["ratio_emprise_parcelle", "annee_mutation"]
    sets = [
        ("structure seule",              NUM_STRUCT + extra, CAT_STRUCT, []),
        ("structure + commune",          NUM_STRUCT + extra, CAT_STRUCT, ["code_commune_insee"]),
        ("structure + commune + IRIS",   NUM_STRUCT + extra, CAT_STRUCT,
                                         ["code_commune_insee", "code_iris"]),
        ("localisation seule",           ["annee_mutation"], [],
                                         ["code_commune_insee", "code_iris"]),
    ]
    kf = KFold(5, shuffle=True, random_state=RNG)
    gkf = GroupKFold(n_splits=5)

    print("RÉGIME ALÉATOIRE — communes vues à l'entraînement (cas de déploiement)")
    res = {}
    for name, fn, fc, geo in sets:
        res[name], _ = evaluate(df, fn, fc, geo, kf, label=name)

    print("\nRÉGIME PAR COMMUNE — communes entières retirées (borne basse)")
    for name, fn, fc, geo in sets:
        evaluate(df, fn, fc, geo, gkf, groups=df["code_commune_insee"], label=name)

    print("\nAPPORT MARGINAL (régime aléatoire, R² log)")
    base = res["structure seule"]
    print(f"  structure seule                              {base:.3f}")
    print(f"  + commune                                    {res['structure + commune']:.3f}"
          f"   ({res['structure + commune']-base:+.3f})")
    print(f"  + IRIS                                       {res['structure + commune + IRIS']:.3f}"
          f"   ({res['structure + commune + IRIS']-res['structure + commune']:+.3f})")
    print(f"  localisation seule (sans aucune structure)   {res['localisation seule']:.3f}")

    # importance par permutation sur le modèle complet
    print("\nIMPORTANCE PAR PERMUTATION (modèle structure + commune + IRIS)")
    fn, fc, geo = NUM_STRUCT + extra, CAT_STRUCT, ["code_commune_insee", "code_iris"]
    tr = df.sample(frac=0.75, random_state=RNG)
    te = df.drop(tr.index)
    Xtr, Xte = build_matrix(tr, te, fn, fc, geo)
    mdl = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.07, max_depth=6,
                                        min_samples_leaf=25, l2_regularization=1.0,
                                        random_state=RNG).fit(Xtr, tr["y"])
    imp = permutation_importance(mdl, Xte, te["y"], n_repeats=8,
                                 random_state=RNG, scoring="r2")
    order = np.argsort(imp.importances_mean)[::-1]
    for i in order:
        if imp.importances_mean[i] > 0.0005:
            print(f"    {Xte.columns[i]:28s} {imp.importances_mean[i]:.4f}"
                  f"  ± {imp.importances_std[i]:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
