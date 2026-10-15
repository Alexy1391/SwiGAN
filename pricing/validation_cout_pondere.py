#!/usr/bin/env python3
"""Protocole A-E : l'erreur de valeur foncière tient-elle une fois pondérée
par l'aléa et la vulnérabilité ?

    A  masquer le prix des maisons qui ont un DVF
    B  les traiter comme si elles n'avaient jamais été vendues
    C  prédire leur valeur avec structure + surface de parcelle + commune
    D  comparer, par commune, somme(V_predit) à somme(V_DVF)
    E  refaire la comparaison sur le coût de dommage q_i * v_i * V_i

L'étape B est plus contraignante qu'il n'y paraît : « jamais vendue » interdit
toute variable dont la disponibilité dépend de la vente. Sont donc écartées, en
plus des colonnes `dvf_*` :

    annee_mutation              n'existe que s'il y a eu mutation
    surface_habitable_logement  DPE obligatoire à la vente (59,8 % de couverture
    classe_bilan_dpe            sur les maisons vendues, 18,2 % sur les autres)

`modele_valeur.py` conservait `annee_mutation` ; le présent protocole ne le peut
pas, et c'est le seul écart de spécification entre les deux modèles.

Point de méthode sur l'étape E : q_i et v_i sont connus pour tout le parc
(l'aléa BRGM est porté par le bâtiment, la vulnérabilité est une fonction de
l'emprise et de l'année). Ils ne sont pas prédits. L'erreur sur le coût communal
provient donc entièrement de l'erreur sur V, mais *repondérée* :

    erreur_pondérée = ( Σ w_i V̂_i − Σ w_i V_i ) / Σ w_i V_i,  w_i = q_i v_i

Elle ne peut différer de l'erreur non pondérée que par deux canaux :
  1. biais — les résidus du modèle sont corrélés au poids w ;
  2. variance — le poids concentre la masse sur moins de maisons, donc les
     erreurs individuelles se compensent moins bien (taille effective de Kish).
Le script mesure les deux séparément.

Conséquence utile : seule la *variation relative* de q et v entre bâtiments
compte ; leur échelle absolue se simplifie dans le rapport. Le résultat ne
dépend donc pas de la calibration d'une prime, seulement de la forme du
gradient — ce que les variantes ci-dessous vérifient.

Usage : python3 validation_cout_pondere.py [--dep 08]
"""

import argparse
import warnings
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import GroupKFold, KFold

from modele_valeur import build_matrix

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent
RNG = 0

# Étape C — structure + foncier + commune. Aucune variable conditionnée par la
# vente (cf. docstring). L'aléa n'entre pas non plus : il sert de pondération à
# l'étape E, l'y mettre aussi comme régresseur mélangerait les deux rôles ; une
# variante le teste en fin de script.
NUM = ["nb_niveau", "annee_construction", "surface_sol_m2", "s_sol_constructions_m2",
       "nb_constructions", "hauteur_max_m", "nb_logements", "s_parcelle", "nb_pdl",
       "distance_monument_historique", "ratio_emprise_parcelle"]
CAT = ["materiau_mur", "materiau_toit"]
GEO = ["code_commune_insee"]

MODELE = dict(max_iter=300, learning_rate=0.07, max_depth=6, min_samples_leaf=25,
              l2_regularization=1.0, random_state=RNG)


def load(dep: str) -> pd.DataFrame:
    """Parc complet, aléa manquant imputé par la classe majoritaire communale
    (même convention que analyse_exposition_argile.py), puis restriction aux
    maisons dont on connaît le prix — ce sont elles dont on masque la valeur."""
    con = duckdb.connect()
    df = con.execute(f"""
        WITH src AS (SELECT * FROM read_csv('{ROOT}/output/features_dep{dep}.csv',
                                            header=true)),
        majo AS (
            SELECT code_commune_insee, alea_argile,
                   row_number() OVER (PARTITION BY code_commune_insee
                                      ORDER BY count(*) DESC, alea_argile) AS rk
            FROM src WHERE alea_argile IS NOT NULL GROUP BY 1, 2
        )
        SELECT s.*, coalesce(s.alea_argile, j.alea_argile) AS alea
        FROM src s
        LEFT JOIN (SELECT code_commune_insee, alea_argile FROM majo WHERE rk = 1) j
               USING (code_commune_insee)
        WHERE s.dvf_mutation_simple
          AND s.dvf_valeur_fonciere BETWEEN 5000 AND 1000000
        ORDER BY s.batiment_groupe_id          -- ordre stable : DuckDB parallélise
                                               -- le scan, sans quoi les plis de
                                               -- validation changent d'un run à l'autre
    """).df()
    df["y"] = np.log(df["dvf_valeur_fonciere"])
    df["ratio_emprise_parcelle"] = df["surface_sol_m2"] / df["s_parcelle"].replace(0, np.nan)
    df["alea"] = df["alea"].fillna("Faible")
    return df.reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Étapes A-C : prédiction hors échantillon, prix masqué                        #
# --------------------------------------------------------------------------- #

def duan(tr, feats_num, feats_cat, geo):
    """Facteur de smearing estimé sur des résidus honnêtes.

    Le calculer sur les résidus d'apprentissage le sous-estimerait : un modèle
    boosté ajuste ses propres points bien mieux qu'un point neuf, l'exponentielle
    des résidus y est donc trop proche de 1. On passe par une validation croisée
    interne au pli d'entraînement.
    """
    resid = np.full(len(tr), np.nan)
    for a, b in KFold(5, shuffle=True, random_state=RNG).split(tr):
        ia, ib = tr.iloc[a], tr.iloc[b]
        Xa, Xb = build_matrix(ia, ib, feats_num, feats_cat, geo)
        m = HistGradientBoostingRegressor(**MODELE).fit(Xa, ia["y"])
        resid[b] = ib["y"].values - m.predict(Xb)
    return float(np.mean(np.exp(resid)))


def predire_hors_echantillon(df, feats_num=NUM, feats_cat=CAT, geo=GEO,
                             par_commune=False):
    """Étapes A + B + C.

    Chaque maison est prédite par un modèle qui n'a jamais vu son prix : son pli
    est retiré de l'entraînement, de l'encodage communal et du facteur de Duan.
    Retourne la valeur prédite en euros.
    """
    pred = np.full(len(df), np.nan)
    if par_commune:
        split = GroupKFold(n_splits=5).split(df, groups=df["code_commune_insee"])
    else:
        split = KFold(5, shuffle=True, random_state=RNG).split(df)
    facteurs = []
    for a, b in split:
        tr, te = df.iloc[a], df.iloc[b]
        Xtr, Xte = build_matrix(tr, te, feats_num, feats_cat, geo)
        m = HistGradientBoostingRegressor(**MODELE).fit(Xtr, tr["y"])
        s = duan(tr, feats_num, feats_cat, geo)
        facteurs.append(s)
        pred[b] = np.exp(m.predict(Xte)) * s
    return pred, float(np.mean(facteurs))


def recalibrer(pred, obs):
    """Recalibrage isotone en euros, ajusté hors échantillon.

    Le modèle est calibré en log (pente log-log 1,006) mais ne l'est plus après
    exponentiation : le facteur de Duan est un multiplicateur unique alors que la
    dispersion des résidus varie avec le niveau de prix. Il en résulte une pente
    résiduelle — les valeurs basses sont sous-estimées d'environ 10 %, les hautes
    surestimées de 5 à 7 %. L'isotonie corrige cette pente sans imposer de forme.
    """
    cal = np.full(len(pred), np.nan)
    for a, b in KFold(5, shuffle=True, random_state=1).split(pred):
        cal[b] = (IsotonicRegression(out_of_bounds="clip")
                  .fit(pred[a], obs[a]).predict(pred[b]))
    return cal


# --------------------------------------------------------------------------- #
# Pondérations de l'étape E                                                    #
# --------------------------------------------------------------------------- #

def poids(df, q_map, expo_surface, facteur_ancien):
    """w_i = q_i * v_i.

    q_i  aléa retrait-gonflement porté par le bâtiment (BRGM), converti en poids
         relatif ; l'échelle absolue est sans effet sur l'erreur relative.
    v_i  vulnérabilité : croissante avec l'emprise (une semelle filante plus
         longue subit un tassement différentiel plus marqué) et majorée pour le
         bâti antérieur aux règles de fondation de 1976.
    """
    q = df["alea"].map(q_map).astype(float).values
    s = df["surface_sol_m2"].clip(lower=20).values
    v = (s / 110.0) ** expo_surface
    v = v * np.where(df["annee_construction"].fillna(1950).values < 1976,
                     facteur_ancien, 1.0)
    return q * v


CALIBRATIONS = [
    # nom, q_map, exposant surface, majoration bâti < 1976
    ("D  référence, poids uniforme (w=1)",
     {"Faible": 1.0, "Moyen": 1.0, "Fort": 1.0}, 0.0, 1.0),
    ("E1 aléa seul, sans vulnérabilité",
     {"Faible": 0.05, "Moyen": 0.50, "Fort": 1.0}, 0.0, 1.0),
    ("E2 vulnérabilité seule, sans aléa",
     {"Faible": 1.0, "Moyen": 1.0, "Fort": 1.0}, 0.5, 1.3),
    ("E3 base   q.v complet",
     {"Faible": 0.05, "Moyen": 0.50, "Fort": 1.0}, 0.5, 1.3),
    ("E4 aléa contrasté (Faible quasi nul)",
     {"Faible": 0.01, "Moyen": 0.50, "Fort": 1.0}, 0.5, 1.3),
    ("E5 binaire ELAN (Moyen/Fort seuls)",
     {"Faible": 0.0, "Moyen": 1.0, "Fort": 1.0}, 0.5, 1.3),
    ("E6 vulnérabilité forte (surface^1, 1976 x1,6)",
     {"Faible": 0.05, "Moyen": 0.50, "Fort": 1.0}, 1.0, 1.6),
]


# --------------------------------------------------------------------------- #
# Étapes D et E : agrégation communale                                         #
# --------------------------------------------------------------------------- #

def agreger(df, pred, w):
    """Somme par commune du coût observé et du coût prédit."""
    g = pd.DataFrame({
        "cc": df["code_commune_insee"].values,
        "n": 1,
        "obs": w * df["dvf_valeur_fonciere"].values,
        "est": w * pred,
        "w": w,
        "w2": w ** 2,
    }).groupby("cc").sum()
    g = g[g["obs"] > 0]
    g["err"] = g["est"] / g["obs"] - 1
    g["n_eff"] = g["w"] ** 2 / g["w2"]          # taille effective de Kish
    return g


BUCKETS = [(1, 4), (5, 19), (20, 99), (100, 10 ** 9)]


def resume(g, label, quiet=False):
    tot = g["est"].sum() / g["obs"].sum() - 1
    med = np.median(np.abs(g["err"])) * 100
    if not quiet:
        print(f"  {label:44s} total {tot*100:+6.2f} %   erreur médiane commune "
              f"{med:5.1f} %   n_eff médian {np.median(g['n_eff']):5.1f}")
    return tot * 100, med


def par_taille(g, label):
    cells = []
    for lo, hi in BUCKETS:
        s = g[(g["n"] >= lo) & (g["n"] <= hi)]
        cells.append(f"{np.median(np.abs(s['err']))*100:5.1f} %" if len(s) else "    — ")
    print(f"  {label:44s} " + "  ".join(cells))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dep", default="08")
    a = ap.parse_args()

    df = load(a.dep)
    print(f"\ndép {a.dep} — protocole A-E sur {len(df):,} maisons dont le prix DVF "
          f"est masqué, {df['code_commune_insee'].nunique()} communes")
    print(f"variables : structure + s_parcelle + commune "
          f"(sans dvf_*, sans annee_mutation, sans DPE)\n")

    # ---- A, B, C -----------------------------------------------------------
    pred, s = predire_hors_echantillon(df)
    df["pred"] = pred
    mape = np.median(np.abs(pred - df["dvf_valeur_fonciere"]) / df["dvf_valeur_fonciere"]) * 100
    r2 = 1 - ((np.log(pred) - df["y"]) ** 2).sum() / ((df["y"] - df["y"].mean()) ** 2).sum()
    print(f"ÉTAPES A-C — prédiction hors échantillon (facteur de Duan {s:.3f})")
    print(f"  R² (log)  {r2:.3f}    erreur médiane sur la maison  {mape:.1f} %\n")

    # ---- D et E ------------------------------------------------------------
    print("ÉTAPES D et E — erreur sur la somme communale")
    res = {}
    for nom, qm, es, fa in CALIBRATIONS:
        w = poids(df, qm, es, fa)
        g = agreger(df, pred, w)
        res[nom] = resume(g, nom)

    print("\n  erreur médiane par nombre de transactions dans la commune")
    print(f"  {'':44s} " + "  ".join(f"{lo}-{hi if hi < 10**9 else '+':>4}"
                                     for lo, hi in BUCKETS))
    for nom, qm, es, fa in CALIBRATIONS:
        par_taille(agreger(df, pred, poids(df, qm, es, fa)), nom)

    # ---- pourquoi : biais vs variance --------------------------------------
    print("\nCANAL 1 — biais : le modèle se trompe-t-il davantage là où l'aléa est fort ?")
    print(df.assign(ratio=df["pred"] / df["dvf_valeur_fonciere"])
            .groupby("alea")
            .apply(lambda x: pd.Series({
                "maisons": len(x),
                "valeur obs. moy.": x["dvf_valeur_fonciere"].mean(),
                "valeur préd. moy.": x["pred"].mean(),
                "biais total %": (x["pred"].sum() / x["dvf_valeur_fonciere"].sum() - 1) * 100,
                "erreur médiane %": np.median(np.abs(x["ratio"] - 1)) * 100,
            }))
            .reindex(["Faible", "Moyen", "Fort"])
            .to_string(float_format=lambda v: f"{v:,.3f}" if abs(v) < 100 else f"{v:,.0f}"))

    # le biais par classe est-il significatif, et est-il interne aux communes ?
    boot = {}
    rb = np.random.default_rng(RNG)
    for cls in ["Faible", "Moyen", "Fort"]:
        x = df[df["alea"] == cls]
        o, e = x["dvf_valeur_fonciere"].values, x["pred"].values
        t = [(e[i].sum() / o[i].sum() - 1) * 100
             for i in (rb.integers(0, len(x), len(x)) for _ in range(2000))]
        boot[cls] = (np.percentile(t, 2.5), np.percentile(t, 97.5))
    print("\n  intervalle de confiance 95 % (bootstrap 2 000 tirages) sur le biais")
    for cls, (lo, hi) in boot.items():
        print(f"    {cls:8s} [{lo:+6.2f} % , {hi:+6.2f} %]")

    # Comparaison interne aux communes : seules les communes portant à la fois du
    # Faible et du Moyen/Fort, pour que l'effet ne puisse pas venir du fait que
    # l'argile se trouve dans des communes globalement moins chères — l'encodage
    # communal est déjà dans le modèle, mais il est homogène sur la commune.
    d = df.assign(mf=df["alea"].isin(["Moyen", "Fort"]))
    g = d.groupby(["code_commune_insee", "mf"]).agg(
        o=("dvf_valeur_fonciere", "sum"), e=("pred", "sum"), n=("pred", "size"))
    g = (g.assign(b=(g["e"] / g["o"] - 1) * 100).reset_index()
          .pivot(index="code_commune_insee", columns="mf", values=["b", "n"]))
    g.columns = ["b_faible", "b_mf", "n_faible", "n_mf"]
    g = g.dropna()
    g = g[(g["n_faible"] >= 5) & (g["n_mf"] >= 5)]
    ecart = g["b_mf"] - g["b_faible"]
    print(f"\n  comparaison interne aux communes mixtes ({len(g)} communes avec au "
          f"moins 5 transactions de chaque côté)")
    print(f"    biais médian sur les maisons Faible      {g['b_faible'].median():+6.2f} %")
    print(f"    biais médian sur les maisons Moyen/Fort  {g['b_mf'].median():+6.2f} %")
    print(f"    écart médian intra-commune               {ecart.median():+6.2f} pt   "
          f"({(ecart > 0).mean():.0%} des communes au-dessus de zéro)")

    print("\nCANAL 2 — variance : sur combien de maisons l'erreur se compense-t-elle ?")
    for nom, qm, es, fa in CALIBRATIONS:
        g = agreger(df, pred, poids(df, qm, es, fa))
        print(f"  {nom:44s} n_eff médian {np.median(g['n_eff']):6.1f}   "
              f"n_eff total {g['n_eff'].sum():8.0f} / {len(df)}")

    print("\n  PLACEBO — mêmes poids permutés à l'intérieur de chaque commune.")
    print("  La permutation conserve exactement n_eff et détruit le lien entre le")
    print("  poids et les caractéristiques du bâtiment : l'écart au réel isole donc")
    print("  la part de l'erreur qui tient à l'aléa, le reste étant de la variance.")
    rng = np.random.default_rng(RNG)
    for nom, qm, es, fa in CALIBRATIONS:
        w = poids(df, qm, es, fa)
        reel = np.median(np.abs(agreger(df, pred, w)["err"])) * 100
        tirs = []
        for _ in range(20):
            wp = (pd.Series(w).groupby(df["code_commune_insee"].values)
                  .transform(lambda x: rng.permutation(x.values)).values)
            tirs.append(np.median(np.abs(agreger(df, pred, wp)["err"])) * 100)
        print(f"  {nom:44s} réel {reel:5.1f} %   permuté {np.mean(tirs):5.1f} %   "
              f"écart {reel - np.mean(tirs):+5.1f} pt")

    # ---- correctifs --------------------------------------------------------
    print("\nCORRECTIFS — deux leviers, qui ne visent pas la même erreur")
    print("  l'aléa en régresseur vise le biais pondéré ; le recalibrage isotone")
    print("  vise la pente résiduelle en euros, donc l'erreur non pondérée.")
    obs = df["dvf_valeur_fonciere"].values
    pred_alea, _ = predire_hors_echantillon(df, feats_cat=CAT + ["alea"])
    variantes = {
        "brut": pred,
        "aléa en régresseur": pred_alea,
        "recalibrage isotone": recalibrer(pred, obs),
        "les deux": recalibrer(pred_alea, obs),
    }
    entetes = [n[:2] for n, _, _, _ in CALIBRATIONS if n.startswith(("D ", "E3", "E5"))]
    print(f"\n  {'':22s} {'maison':>7s}   " +
          "".join(f"{e + ' total / médiane':>24s}" for e in entetes))
    for lab, p in variantes.items():
        cells = ""
        for nom, qm, es, fa in CALIBRATIONS:
            if nom.startswith(("D ", "E3", "E5")):
                g = agreger(df, p, poids(df, qm, es, fa))
                cells += (f"{(g['est'].sum() / g['obs'].sum() - 1) * 100:+10.2f} % /"
                          f"{np.median(np.abs(g['err'])) * 100:5.1f} %")
        print(f"  {lab:22s} {np.median(np.abs(p - obs) / obs) * 100:6.1f} %   " + cells)

    # ---- borne basse : communes jamais vues --------------------------------
    print("\nBORNE BASSE — communes entières retirées de l'entraînement")
    pred3, s3 = predire_hors_echantillon(df, par_commune=True)
    for nom, qm, es, fa in CALIBRATIONS:
        if not nom.startswith(("D ", "E3", "E5")):
            continue
        resume(agreger(df, pred3, poids(df, qm, es, fa)), nom)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
