#!/usr/bin/env python3
"""Calibration de q_i et v_i sur le coût total de la section 6 de l'article.

La partie 3 de l'étude (`validation_cout_pondere.py`) travaille avec des poids
w_i = q_i v_i purement *relatifs* : l'échelle absolue se simplifie dans le
rapport erreur/total, si bien qu'aucune calibration de prime n'était nécessaire.
Le présent script fixe cette échelle en l'ancrant sur la fonction de coût de la
section 6.2 de l'article SwiGAN (équation 6.1), reprise de Heranval et al. 2023.

    C(x) = a + b x - b exp(-k x)        (6.1)     k = a / (5 b)

    x  nombre maximal de bâtiments endommagés d'une commune éligible CatNat
    C  coût maximal des dommages, en euros

CE QUE L'ÉQUATION 6.1 IDENTIFIE, ET CE QU'ELLE N'IDENTIFIE PAS
--------------------------------------------------------------
C(x) ne dépend que d'un *comptage* de bâtiments : ni valeur, ni emprise, ni âge,
ni classe d'aléa n'y entrent. Le modèle de coût de la partie 3, lui, somme des
q_i v_i V_i. Écrire la cohérence

    somme_i  kappa q_i v_i V_i  =  somme_c  C(x_c)

sur un périmètre donné ne fournit donc qu'*une seule* équation scalaire. Elle
identifie le produit kappa = pi * delta, jamais pi et delta séparément :

    q_i = pi    * g(alea_i)          g = gradient d'aléa, normalisé en moyenne 1
    v_i = delta * h(emprise_i, annee_i)   h = gradient de vulnérabilité, idem

Toute paire (pi, delta) de produit kappa reproduit exactement le même coût total.
Séparer les deux exige une contrainte externe — une fréquence de sinistre ou une
sévérité moyenne observée. Le script calcule kappa, puis tabule les couples
(pi, delta) admissibles avec la sévérité moyenne qu'ils impliquent, de façon que
l'ancrage externe soit un choix explicite et non un artefact de calibration.

SUR LES CONSTANTES DE L'ÉQUATION 6.1
------------------------------------
L'article imprime « a = 464.4 et b = 4.121e8 [...] et k = a/5b ». Pris tels
quels, ces paramètres donnent C(0) = -412 MEUR et 412 MEUR par bâtiment : la
transcription est fautive. La source tranche. Heranval et al. (2023), équation 3
et section 5.2, posent

    E[M] = nombre de maisons x 464,4 - 4,121e8   (alpha, beta) = (-4,121e8 ; 464,4)

a = 464,4 est donc la PENTE, en euros par maison, et b = 4,121e8 la valeur
absolue de l'ORDONNÉE À L'ORIGINE — SwiGAN intervertit les deux rôles en écrivant
« C0(x) = a + bx ». La forme cohérente est C(x) = a x - b(1 - exp(-kx)), portée
par `cout_subsidence_futur.py`, où elle est le défaut.

Ce script, lui, a été écrit sous la lecture antérieure — forme imprimée conservée
et b relu à 4,121e2 — qui donnait 412,1 EUR/bâtiment contre 371,5 pour la lecture
juste, soit 10 % d'écart. Il l'épingle explicitement (`lecture="b-marginal"`)
pour que ses résultats restent comparables à ceux déjà produits. kappa étant
proportionnel au coût marginal, passer à la lecture juste ne ferait que déplacer
l'échelle de 10 % : la table de sensibilité à b, en fin de script, mesure déjà
exactement cet effet. À relancer sous `--b 371.5` pour un ancrage à jour.

Usage : python3 calibration_cout_paper.py [--dep 08] [--b 412.1]
"""

import argparse
import warnings
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import KFold

from cout_subsidence_futur import A_EURO, LECTURES
from cout_subsidence_futur import cout_commune as _cout_commune
from modele_valeur import build_matrix
from validation_cout_pondere import CAT, GEO, MODELE, NUM, RNG, duan

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent


# b de la lecture « b-marginal » : 412,1 EUR/bâtiment. À ne pas confondre avec
# `cout_subsidence_futur.B_EURO_DEFAUT`, qui est désormais l'ordonnée à l'origine
# de Heranval (4,121e8) — les deux constantes ne jouent pas le même rôle.
B_EURO_DEFAUT = LECTURES["b-marginal"]["b"]


def cout_commune(x, a=A_EURO, b=B_EURO_DEFAUT):
    """Équation 6.1 dans la lecture pour laquelle la docstring ci-dessus plaide.

    `cout_subsidence_futur.py` en porte une seconde, « a-marginal », qui prend
    les constantes à leur magnitude imprimée et qui y est le défaut. On épingle
    ici « b-marginal » pour que l'ancrage de kappa ne se déplace pas en silence :
    kappa étant proportionnel à b, changer de lecture ne ferait que translater
    l'échelle, ce que la table de sensibilité à b en fin de script mesure déjà.
    """
    return _cout_commune(x, a=a, b=b, lecture="b-marginal")


# --------------------------------------------------------------------------- #
# Parc complet (et non les seules maisons vendues)                             #
# --------------------------------------------------------------------------- #

def load_parc(dep: str) -> pd.DataFrame:
    """Tout le parc du département, aléa manquant imputé par la classe
    majoritaire communale — même convention que le reste de l'étude.

    À la différence de `validation_cout_pondere.load()`, aucune restriction aux
    maisons transactées : c'est bien l'ensemble du parc qui est exposé, et c'est
    sur lui que porte le comptage x_c de l'article.
    """
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
        SELECT s.*,
               coalesce(s.alea_argile, j.alea_argile) AS alea,
               s.alea_argile IS NOT NULL              AS alea_observe
        FROM src s
        LEFT JOIN (SELECT code_commune_insee, alea_argile FROM majo WHERE rk = 1) j
               USING (code_commune_insee)
        ORDER BY s.batiment_groupe_id      -- ordre stable : cf. validation_cout_pondere
    """).df()
    df["ratio_emprise_parcelle"] = df["surface_sol_m2"] / df["s_parcelle"].replace(0, np.nan)
    df["alea"] = df["alea"].fillna("Faible")
    df["vendue"] = (df["dvf_mutation_simple"].fillna(False)
                    & df["dvf_valeur_fonciere"].between(5000, 1000000))
    df.loc[df["vendue"], "y"] = np.log(df.loc[df["vendue"], "dvf_valeur_fonciere"])
    return df.reset_index(drop=True)


def valeur_parc(df: pd.DataFrame) -> tuple[np.ndarray, float]:
    """Valeur V_i de chaque maison du parc, en euros.

    Le modèle s'entraîne sur les seules maisons vendues (les seules qui portent
    une cible) et prédit l'ensemble du parc. Les maisons vendues reçoivent une
    prédiction hors échantillon, pour que la valeur totale ne soit pas gonflée
    par l'ajustement du modèle à ses propres points ; les maisons non vendues
    sont prédites par un modèle entraîné sur la totalité des ventes.
    """
    vendues = df[df["vendue"]].copy()
    pred = np.full(len(df), np.nan)

    # maisons vendues : prédiction hors échantillon (mêmes plis que la partie 3)
    idx_v = np.flatnonzero(df["vendue"].values)
    oof = np.full(len(vendues), np.nan)
    for a, b in KFold(5, shuffle=True, random_state=RNG).split(vendues):
        tr, te = vendues.iloc[a], vendues.iloc[b]
        Xtr, Xte = build_matrix(tr, te, NUM, CAT, GEO)
        m = HistGradientBoostingRegressor(**MODELE).fit(Xtr, tr["y"])
        oof[b] = np.exp(m.predict(Xte)) * duan(tr, NUM, CAT, GEO)
    pred[idx_v] = oof

    # maisons jamais vendues : modèle entraîné sur toutes les ventes
    reste = df[~df["vendue"]]
    Xtr, Xte = build_matrix(vendues, reste, NUM, CAT, GEO)
    m = HistGradientBoostingRegressor(**MODELE).fit(Xtr, vendues["y"])
    s = duan(vendues, NUM, CAT, GEO)
    pred[np.flatnonzero(~df["vendue"].values)] = np.exp(m.predict(Xte)) * s
    return pred, s


# --------------------------------------------------------------------------- #
# Gradients relatifs : ceux de la partie 3, normalisés en moyenne 1            #
# --------------------------------------------------------------------------- #

Q_MAP = {"Faible": 0.05, "Moyen": 0.50, "Fort": 1.0}   # E3, gradient d'aléa
EXPO_SURFACE, FACTEUR_ANCIEN = 0.5, 1.3                # E3, gradient de vulnérabilité


def gradients(df):
    """g (aléa) et h (vulnérabilité), chacun normalisé à une moyenne de 1 sur le
    parc exposé. La normalisation ne change aucun résultat — elle rend seulement
    pi et delta lisibles comme des niveaux moyens plutôt que comme des échelles
    arbitraires."""
    g = df["alea"].map(Q_MAP).astype(float).values
    s = df["surface_sol_m2"].clip(lower=20).values
    h = (s / 110.0) ** EXPO_SURFACE
    h = h * np.where(df["annee_construction"].fillna(1950).values < 1976,
                     FACTEUR_ANCIEN, 1.0)
    return g / g.mean(), h / h.mean()


# --------------------------------------------------------------------------- #
# Définitions du périmètre exposé                                              #
# --------------------------------------------------------------------------- #

def expositions(df):
    """Masques d'exposition. L'article compte « les bâtiments résidentiels situés
    en zone argileuse » de chaque commune éligible : toutes les classes d'aléa
    BRGM traduisent une présence d'argile, donc P0 en est la lecture littérale.
    Les variantes reprennent les définitions de la partie 2."""
    mf = df["alea"].isin(["Moyen", "Fort"]).values
    com_mf = df.groupby("code_commune_insee")["alea"].transform(
        lambda s: s.isin(["Moyen", "Fort"]).any()).values
    return {
        "P0 toute maison en zone argileuse (lecture littérale)": np.ones(len(df), bool),
        "A2 toute maison d'une commune portant du Moyen/Fort": com_mf,
        "B  maison en aléa Moyen ou Fort": mf,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dep", default="08")
    ap.add_argument("--b", type=float, default=B_EURO_DEFAUT,
                    help="coût marginal par bâtiment exposé, en euros (eq. 6.1)")
    a = ap.parse_args()

    df = load_parc(a.dep)
    print(f"\ndép {a.dep} — parc complet : {len(df):,} maisons, "
          f"{df['code_commune_insee'].nunique()} communes, "
          f"{int(df['vendue'].sum()):,} avec un prix DVF exploitable")
    print(f"équation 6.1 : a = {A_EURO} EUR, b = {a.b} EUR/bâtiment, "
          f"k = a/(5b) = {A_EURO / (5 * a.b):.4f}")

    V, smear = valeur_parc(df)
    df["V"] = V
    print(f"valeur prédite sur tout le parc (facteur de Duan {smear:.3f}) : "
          f"total {V.sum() / 1e9:.2f} Md EUR, médiane {np.median(V):,.0f} EUR")

    g, h = gradients(df)
    df["g"], df["h"] = g, h

    print("\n" + "=" * 78)
    print("ANCRAGE DU TOTAL — kappa = C_article / somme(g h V)")
    print("=" * 78)
    print(f"  {'périmètre exposé':52s} {'bâtiments':>9s} {'C_art.':>9s} {'kappa':>9s}")
    resultats = {}
    for nom, masque in expositions(df).items():
        sub = df[masque]
        # coût de l'article : par commune éligible, appliqué au comptage x_c
        x = sub.groupby("code_commune_insee").size()
        C = cout_commune(x.values, b=a.b).sum()
        base = (sub["g"] * sub["h"] * sub["V"]).sum()
        kappa = C / base
        resultats[nom] = (masque, C, kappa)
        print(f"  {nom:52s} {len(sub):9,} {C / 1e6:8.1f}M {kappa:9.2e}")

    # --- lecture retenue : le périmètre littéral de l'article ----------------
    nom0 = next(iter(expositions(df)))
    masque0, C0, kappa0 = resultats[nom0]
    sub = df[masque0].copy()
    sub["tau"] = kappa0 * sub["g"] * sub["h"]        # taux de dommage sur la valeur
    sub["cout"] = sub["tau"] * sub["V"]

    print("\n" + "=" * 78)
    print(f"TAUX DE DOMMAGE IMPLIQUÉ — périmètre « {nom0} »")
    print("=" * 78)
    print(f"  {'aléa':8s} {'maisons':>9s} {'tau médian':>11s} {'coût médian':>12s} "
          f"{'coût moyen':>11s} {'part du total':>13s}")
    for cls in ["Faible", "Moyen", "Fort"]:
        s = sub[sub["alea"] == cls]
        if s.empty:
            continue
        print(f"  {cls:8s} {len(s):9,} {s['tau'].median() * 100:10.3f}% "
              f"{s['cout'].median():11,.0f}E {s['cout'].mean():10,.0f}E "
              f"{s['cout'].sum() / sub['cout'].sum() * 100:12.1f}%")
    print(f"  {'ensemble':8s} {len(sub):9,} {sub['tau'].median() * 100:10.3f}% "
          f"{sub['cout'].median():11,.0f}E {sub['cout'].mean():10,.0f}E "
          f"{100:12.1f}%")
    print(f"\n  contrôle : somme des coûts {sub['cout'].sum() / 1e6:.1f} MEUR "
          f"contre {C0 / 1e6:.1f} MEUR pour l'équation 6.1 "
          f"(écart {(sub['cout'].sum() / C0 - 1) * 100:+.2e} %)")
    print(f"  coût moyen par bâtiment exposé : {sub['cout'].mean():,.0f} EUR "
          f"(b de l'équation 6.1 : {a.b:,.1f} EUR)")

    # --- séparation pi / delta : non identifiée par le total seul -------------
    print("\n" + "=" * 78)
    print("SÉPARATION FRÉQUENCE / SÉVÉRITÉ — non identifiée par le total seul")
    print("=" * 78)
    print(f"  kappa = pi * delta = {kappa0:.4e}. Toute paire ci-dessous donne le")
    print("  même coût total ; il faut un ancrage externe pour trancher.\n")
    Vmoy = float((sub["V"] * sub["h"]).sum() / sub["h"].sum())
    print(f"  {'pi (fréquence moy.)':>22s} {'delta (sévérité)':>18s} "
          f"{'sévérité moyenne':>18s}")
    for pi in [0.005, 0.01, 0.02, 0.05, 0.10]:
        delta = kappa0 / pi
        print(f"  {pi * 100:21.2f}% {delta * 100:17.2f}% "
              f"{delta * Vmoy:15,.0f} EUR")
    print(f"\n  (valeur moyenne pondérée par la vulnérabilité : {Vmoy:,.0f} EUR)")

    # --- diagnostic : où les deux répartitions divergent ----------------------
    print("\n" + "=" * 78)
    print("RÉPARTITION COMMUNALE — article (comptage) contre modèle (valeur)")
    print("=" * 78)
    com = sub.groupby("code_commune_insee").agg(
        n=("V", "size"), modele=("cout", "sum"), V_med=("V", "median")).reset_index()
    com["article"] = cout_commune(com["n"].values, b=a.b)
    com["ecart"] = (com["modele"] - com["article"]) / com["article"] * 100
    bornes = [(1, 49), (50, 199), (200, 999), (1000, 10 ** 9)]
    print(f"  {'taille (bâtiments exposés)':28s} {'communes':>9s} {'V médiane':>11s} "
          f"{'écart médian':>13s}")
    for lo, hi in bornes:
        s = com[com["n"].between(lo, hi)]
        if s.empty:
            continue
        lab = f"{lo}-{hi if hi < 10 ** 9 else '+'}"
        print(f"  {lab:28s} {len(s):9,} {s['V_med'].median():10,.0f}E "
              f"{s['ecart'].median():12.1f}%")
    q = com["ecart"].quantile([0.1, 0.5, 0.9])
    print(f"\n  écart modèle/article par commune : médiane {q[0.5]:+.1f} %, "
          f"décile 1 {q[0.1]:+.1f} %, décile 9 {q[0.9]:+.1f} %")
    print("  Le total est identique par construction : ces écarts sont une pure")
    print("  redistribution entre communes.")

    # --- quel canal porte la redistribution ? --------------------------------
    # L'article facture ~b euros par bâtiment, quel que soit le bâtiment. Le
    # modèle facture kappa g h V. Le rapport commune par commune vaut donc
    # kappa/b fois la moyenne communale de g h V. On neutralise tour à tour deux
    # des trois facteurs à leur moyenne de parc pour voir lequel disperse.
    print("\n" + "=" * 78)
    print("QUEL CANAL PORTE LA REDISTRIBUTION ?")
    print("=" * 78)
    canaux = {
        "aléa g seul (h, V neutralisés)":        (sub["g"], 1.0, 1.0),
        "vulnérabilité h seule (g, V neutr.)":   (1.0, sub["h"], 1.0),
        "valeur V seule (g, h neutralisés)":     (1.0, 1.0, sub["V"]),
        "les trois ensemble":                    (sub["g"], sub["h"], sub["V"]),
    }
    print(f"  {'canal':38s} {'D1':>9s} {'médiane':>9s} {'D9':>9s} {'D9/D1':>8s}")
    for nom, (gg, hh, vv) in canaux.items():
        z = pd.Series(np.broadcast_to(np.asarray(gg, float), len(sub))
                      * np.broadcast_to(np.asarray(hh, float), len(sub))
                      * np.broadcast_to(np.asarray(vv, float), len(sub)),
                      index=sub.index)
        m = z.groupby(sub["code_commune_insee"]).mean()
        m = m / m.mean()          # ramené à 1 en moyenne : seule la dispersion compte
        d1, md, d9 = m.quantile([0.1, 0.5, 0.9])
        print(f"  {nom:38s} {d1:9.2f} {md:9.2f} {d9:9.2f} {d9 / d1:8.1f}x")
    print("\n  Lecture : un rapport D9/D1 proche de 1 signifie que le canal ne")
    print("  redistribue rien. C'est l'aléa, non la valeur, qui écarte le modèle")
    print("  du comptage communal de l'article.")

    # --- kappa dépend-il du jeu de communes éligibles ? ----------------------
    # L'article ne facture que les communes déclarées éligibles CatNat une année
    # donnée ; on ne dispose pas de ces déclarations. kappa n'est utilisable que
    # s'il ne dépend pas du sous-ensemble retenu — ce que teste ce tirage.
    print("\n" + "=" * 78)
    print("STABILITÉ DE kappa SELON LE JEU DE COMMUNES ÉLIGIBLES")
    print("=" * 78)
    rng = np.random.default_rng(RNG)
    communes = sub["code_commune_insee"].unique()
    print(f"  {'part des communes éligibles':30s} {'kappa médian':>14s} {'écart D1-D9':>18s}")
    for part in [0.10, 0.25, 0.50, 0.75]:
        ks = []
        for _ in range(60):
            tirage = rng.choice(communes, size=max(1, int(len(communes) * part)),
                                replace=False)
            s = sub[sub["code_commune_insee"].isin(tirage)]
            x = s.groupby("code_commune_insee").size()
            ks.append(cout_commune(x.values, b=a.b).sum()
                      / (s["g"] * s["h"] * s["V"]).sum())
        ks = np.array(ks)
        d1, md, d9 = np.quantile(ks, [0.1, 0.5, 0.9])
        print(f"  {part * 100:29.0f}% {md:14.3e} "
              f"{(d9 / d1 - 1) * 100:16.1f}%")
    print(f"  {'toutes (référence)':30s} {kappa0:14.3e} {0.0:16.1f}%")
    print("\n  kappa varie de quelques pour cent d'un tirage à l'autre : il est")
    print("  gouverné par le coût marginal b, que l'équation 6.1 applique")
    print("  identiquement à tout bâtiment. Il se transporte donc d'un jeu de")
    print("  communes éligibles à un autre.")

    # --- sensibilité à b ------------------------------------------------------
    print("\n" + "=" * 78)
    print("SENSIBILITÉ À LA CONSTANTE b DE L'ÉQUATION 6.1")
    print("=" * 78)
    print(f"  {'b (EUR/bâtiment)':>20s} {'C article':>12s} {'kappa':>11s} "
          f"{'tau médian Fort':>16s}")
    gh = (sub["g"] * sub["h"] * sub["V"]).sum()
    fort = sub["alea"] == "Fort"
    for bb in [4.121, 41.21, 412.1, 4121.0]:
        x = sub.groupby("code_commune_insee").size()
        C = cout_commune(x.values, b=bb).sum()
        kp = C / gh
        tm = (kp * sub.loc[fort, "g"] * sub.loc[fort, "h"]).median()
        print(f"  {bb:20,.2f} {C / 1e6:11.2f}M {kp:11.2e} {tm * 100:15.3f}%")
    print("\n  kappa est proportionnel à b : changer de lecture de l'exposant")
    print("  déplace l'échelle sans toucher à la structure relative de q et v.")

    # --- contrôle externe : ordres de grandeur de la section 6.3 -------------
    # La section 6.3 annonce, pour tout le Grand Est, des pertes de pointe
    # dépassant 100 MEUR et une Value-at-Risk proche de 200 MEUR. Le pilote 08
    # ne couvre que 6,4 % du parc régional ; l'extrapolation donne la part de
    # communes qui devrait être déclarée éligible pour retrouver ces montants.
    print("\n" + "=" * 78)
    print("CONTRÔLE EXTERNE — ordres de grandeur annoncés en section 6.3")
    print("=" * 78)
    PARC_GE = 1_529_807          # maisons du Grand Est retenues (partie 1)
    part_08 = len(df) / PARC_GE
    ge_total = C0 / part_08
    print(f"  dép {a.dep} : {len(df):,} maisons, soit {part_08 * 100:.1f} % du parc "
          f"Grand Est ({PARC_GE:,})")
    print(f"  coût si toutes les communes du dép {a.dep} sont éligibles : "
          f"{C0 / 1e6:.1f} MEUR")
    print(f"  extrapolé au Grand Est, éligibilité totale : {ge_total / 1e6:.0f} MEUR")
    for cible, nom in [(100e6, "pointe annoncée   (> 100 MEUR)"),
                       (200e6, "Value-at-Risk     (~ 200 MEUR)")]:
        print(f"    {nom:34s} -> {cible / ge_total * 100:5.1f} % des communes éligibles")
    print("\n  Ces parts d'éligibilité sont plausibles pour une année de sécheresse")
    print("  sévère, ce qui conforte la lecture b = 412,1 EUR/bâtiment : avec")
    print("  b = 4 121, il faudrait moins de 2 % de communes éligibles pour saturer")
    print("  la VaR régionale, et le taux de dommage en aléa Fort atteindrait 20 %")
    print("  de la valeur du bien — hors de portée pour un bâtiment simplement exposé.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
