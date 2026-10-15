#!/usr/bin/env python3
"""Coût futur de la subsidence : la section 6.2 de l'article appliquée en avant.

La section 6.2 de l'article SwiGAN décrit une chaîne à trois maillons :

    trajectoire de SWI générée
      -> éligibilité CatNat de chaque commune, année par année        (section 6.1)
      -> x_c, comptage des bâtiments résidentiels exposés d'une commune éligible
      -> C(x_c) = a + b x_c - b exp(-k x_c)                           (équation 6.1)
      -> somme sur les communes éligibles, par trajectoire et par année

`calibration_cout_paper.py` implémente déjà l'équation, mais l'emploie *à
l'envers* : il s'en sert pour ancrer l'échelle kappa d'un modèle de coût par
bâtiment, sur un périmètre supposé entièrement éligible. Aucun script ne
parcourait la chaîne dans le sens de la projection — c'est l'objet de celui-ci.
L'équation 6.1 est écrite ici, et `calibration_cout_paper.py` l'importe : elle
n'a ainsi qu'une définition, et la projection ne dépend pas de la pile de
modélisation (scikit-learn) dont l'ancrage de kappa a besoin.

CE QUI EST ICI, ET LE MAILLON QUI MANQUE
----------------------------------------
Le maillon 1 (section 6.1) — agrégation mensuelle du SWI au mois le plus sec,
période de retour du minimum annuel par pixel SAFRAN, puis passage du pixel à la
commune — n'est pas implémenté dans le dépôt, et ne peut pas l'être en l'état :
le passage pixel -> commune demande les contours communaux, que les données
disponibles ne portent pas. Le SWI est sur une grille Lambert-93 de 8 km
(`Grand Est/swi_region.parquet`, 896 mailles, colonnes x/y en mètres) ; l'extrait
BDNB, lui, ne porte qu'un code INSEE, sans coordonnées. Seul le département 08
conserve son export brut, avec les géométries de `batiment_groupe`.

La table d'éligibilité est donc une **entrée** de ce script (`--eligibilite`), à
produire par ailleurs. En son absence, le script calcule l'enveloppe : coût à
éligibilité totale, puis balayage de la part de communes éligibles avec les
bornes atteignables — ce qui suffit à confronter les ordres de grandeur de la
section 6.3 (pointes > 100 MEUR, Value-at-Risk ~ 200 MEUR) au parc réel du Grand
Est, là où `calibration_cout_paper.py` devait les extrapoler depuis les Ardennes.

FORMAT DE LA TABLE D'ÉLIGIBILITÉ
--------------------------------
CSV ou Parquet, une ligne par (commune, année, trajectoire) éligible :

    code_commune_insee,annee,trajectoire[,eligible]
    08001,2031,0,true
    08013,2031,0,true

`trajectoire` est facultative (un scénario déterministe unique si elle manque),
`eligible` aussi (toute ligne présente est alors tenue pour éligible ; sinon les
lignes sont filtrées sur cette colonne). Les alias usuels des trois colonnes sont
reconnus, et les codes INSEE sont normalisés sur 5 caractères.

DÉFINITIONS D'EXPOSITION
------------------------
L'article compte « les bâtiments résidentiels situés en zone argileuse » des
communes éligibles. La partie 2 de l'étude a montré que cette phrase admet
plusieurs lectures, qui ne diffèrent pas seulement par leur total mais par la
commune à laquelle elles imputent l'exposition. Les quatre sont calculées ici,
P0 étant la lecture littérale et la valeur par défaut.

Réserve de couverture : `alea_argile` n'est renseigné que sur le département 08,
les neuf autres ayant été extraits avant l'ajout de `batiment_groupe_risques`.
Les définitions A1, A2 et B ne sont donc calculables que sur les Ardennes ; le
script le signale commune par commune plutôt que de compter zéro en silence.

CE QUE LE PROTOCOLE NE FAIT PAS
-------------------------------
Fidèlement à la section 6.2, et comme la section 6.3 le reconnaît : l'exposition
est figée (ni démographie, ni normes de construction, ni adaptation), et une
commune éligible plusieurs années est facturée en entier à chaque fois — aucune
maison n'est retirée du parc après sinistre.

UNE ANNÉE, PAS LE CUMUL 2025-2050
---------------------------------
Tout coût produit ici est un coût ANNUEL : celui d'une année où l'ensemble de
communes considéré est éligible. C'est la convention de l'article, que sa figure
13(a) rend explicite — une boîte à moustaches par année de 2025 à 2050, les 1 000
trajectoires en distribution, « Financial cost of damages » en ordonnée. La
médiane y est nulle sur la plupart des années et culmine vers 110 MEUR en 2038 et
2042 ; la section 6.3 lit ces mêmes chiffres comme « peak losses in the worst
years » et une Value-at-Risk proche de 200 MEUR. Le « maximum » de la section 6.2
porte sur l'exposition (tout bâtiment d'une commune éligible est compté comme
sinistré), pas sur le temps.

Le cumul de la période n'a donc de sens que par trajectoire, et `--eligibilite`
le donne à part (`cumul 2025-2050 par trajectoire`), à côté de la pire année.
L'enveloppe, elle, est une coupe annuelle : ses 572,5 MEUR sont le coût d'une
seule année où les 5 111 communes seraient éligibles à la fois — une borne
inatteignable, à ne pas confronter aux 100-200 MEUR de la section 6.3 sans passer
par la lecture inverse en part de communes éligibles.

L'ÉQUATION 6.1, RÉSOLUE PAR LA SOURCE
-------------------------------------
Prise à la lettre, l'équation 6.1 de SwiGAN donne C(0) = -412 MEUR et 412 MEUR
par bâtiment : elle est mal transcrite. La source la fixe. Heranval et al.
(2023), équation 3 et section 5.2, posent

    E[M] = nombre de maisons x 464,4 - 4,121e8    (alpha, beta) = (-4,121e8 ; 464,4)

donc a = 464,4 est la PENTE en euros par maison et b = 4,121e8 la valeur absolue
de l'ORDONNÉE À L'ORIGINE. SwiGAN intervertit les deux rôles en écrivant
« C0(x) = a + bx ». La reconstruction cohérente est

    C(x) = a x - b (1 - exp(-k x))        k = a/(5b)

vérifiée sur les résultats publiés de Heranval (leur tableau 4 reproduit à
0,07 % près). Elle est le défaut ; `--lecture b-marginal` conserve la lecture
antérieure du dépôt, aujourd'hui réfutée, et `--k-negatif` teste le signe imprimé.

UNE FOIS AU TOTAL, OU COMMUNE PAR COMMUNE ? L'ARTICLE NE LE DIT PAS
--------------------------------------------------------------------
La section 6.2 compte les bâtiments « in each commune », puis applique 6.1 pour
obtenir le coût « in the Grand Est region ». Elle n'écrit aucune somme, et x y
est « the maximum number of damaged buildings », sans indice de commune. Les deux
lectures sont donc ouvertes :

    commune   somme_c C(x_c) sur les communes éligibles
    region    C(X) une seule fois, X = somme_c x_c

Elles coïncident au premier ordre, car 1/k = 5b/a vaut 4,4 millions de bâtiments
et tout X régional lui est très inférieur : chaque C y vaut 0,8 a x. L'écart
mesuré va de +0,2 % à 5 % de communes éligibles à +3,9 % à éligibilité totale du
Grand Est. Le choix est donc sans conséquence pratique ici — ce qui explique sans
doute que l'article ne le précise pas.

À ne pas confondre avec `heranval`, la troisième agrégation : C0 = a X - b, le
modèle source *non corrigé*, appliqué une fois. Heranval et al. l'ajustent sur 15
points annuels nationaux (1,7 million de maisons exposées en moyenne, 4,7
millions en 2003) et l'appliquent bien ainsi — c'est de cette façon, et seulement
ainsi, que leur tableau 4 se reproduit. Mais C0 s'annule à b/a = 887 382 maisons :
en dessous, hors de son domaine d'ajustement, elle est négative. Le Grand Est
entier n'en compte que 1,54 million, et une commune 100 à 30 000 : d'où le
« eliminate negative costs » de 6.1, et d'où l'écart 572,5 contre 303,6 MEUR à
éligibilité régionale totale.

Usage : python3 cout_subsidence_futur.py [--dep 08] [--eligibilite elig.csv]
                                         [--exposition P0] [--lecture a-marginal]
                                         [--agregation commune] [--b B]
                                         [--k-negatif] [--csv out.csv]
"""

import argparse
from functools import partial
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
RNG = 0
ANNEES_ARTICLE = (2025, 2050)

# Ordres de grandeur annoncés en section 6.3, pour le Grand Est entier.
CIBLES_6_3 = [(100e6, "pointe des mauvaises années (> 100 MEUR)"),
              (200e6, "Value-at-Risk                (~ 200 MEUR)")]

# --------------------------------------------------------------------------- #
# Équation 6.1, telle que Heranval et al. (2023) la fixe                       #
# --------------------------------------------------------------------------- #

A_EURO = 464.4          # coût marginal par maison exposée : le beta de Heranval
B_EURO_DEFAUT = 4.121e8  # |ordonnée à l'origine| de Heranval : son alpha, au signe près

# L'équation 5.1 de Heranval et al. (2023), « Application of machine learning
# methods to predict drought cost in France », European Actuarial Journal, est
#
#     E[M] = nombre de maisons x 464,4 - 4,121e8            (leur équation 3)
#     (alpha, beta) = (-4,121e8 ; 464,4)                    (leur section 5.2)
#
# soit, avec les noms de constantes de SwiGAN, C0(x) = a x - b : a est la PENTE
# (euros par maison), b la valeur absolue de l'ORDONNÉE À L'ORIGINE. SwiGAN
# intervertit les deux rôles en imprimant « C0(x) = a + bx », d'où l'équation 6.1
# incohérente. La reconstruction est
#
#     C(x) = a x - b (1 - exp(-k x))          k = a/(5b)
#
# qui vaut 0 en 0, croît partout, et tend vers C0. Elle a été vérifiée sur les
# résultats publiés de Heranval : a n - b reproduit leur tableau 4 à 0,07 % près
# à partir des comptages de maisons de leur tableau 5.
LECTURES = {
    # Reconstruction ci-dessus. Constantes à leur magnitude imprimée, rôles
    # rendus à ceux de Heranval.
    "a-marginal": dict(
        b=B_EURO_DEFAUT,
        forme="C(x) = a x - b (1 - exp(-k x))",
        note="a = 464,4 EUR/maison (pente de Heranval) ; C(0) = 0 ; C -> a x - b",
    ),
    # Lecture antérieure, conservée pour reproduire les runs de
    # `calibration_cout_paper.py`. Elle sauvait la forme imprimée par SwiGAN en
    # relisant b à 4,121e2 ; Heranval montre que c'est la forme, non la
    # constante, qui était fautive. À ne pas utiliser pour de nouveaux résultats.
    "b-marginal": dict(
        b=412.1,
        forme="C(x) = a + b x - b exp(-k x)",
        note="PÉRIMÉE (réfutée par Heranval et al.) ; b = 412,1 EUR/bâtiment",
    ),
}
LECTURE_DEFAUT = "a-marginal"


def cout_commune(x, a=A_EURO, b=None, lecture=LECTURE_DEFAUT, k_negatif=False):
    """Équation 6.1 : coût maximal, en euros, d'une commune de x bâtiments.

    Sur le domaine réel, l'exponentielle est plate — 1/k = 5b/a vaut 4,4 millions
    de maisons — et C se réduit à (a - bk) x = 0,8 a x = 371,5 EUR par maison. Le
    facteur 5 de k = a/(5b) est précisément ce qui borne cette pente à 80 % de a
    et garantit que C reste croissante.

    `k_negatif` rétablit le signe imprimé par SwiGAN, k = -a/(5b). C reste alors
    positive et croissante (elle vaut a x + b(exp(|k|x) - 1)), mais elle diverge
    de C0 au lieu d'y converger, ce que l'article dit explicitement ne pas
    vouloir : le « - » imprimé est selon toute vraisemblance la barre de fraction
    de \\frac{a}{5b} mal restituée. La pente effective y est 1,2 a = 557,3 EUR.
    """
    if lecture not in LECTURES:
        raise ValueError(f"lecture inconnue : {lecture} (attendu : {list(LECTURES)})")
    if b is None:
        b = LECTURES[lecture]["b"]
    k = a / (5.0 * b) * (-1.0 if k_negatif else 1.0)
    x = np.asarray(x, dtype=float)
    if lecture == "a-marginal":
        return a * x - b * (1.0 - np.exp(-k * x))
    return a + b * x - b * np.exp(-k * x)


def cout_heranval(x_total, a=A_EURO, b=B_EURO_DEFAUT):
    """C0 appliquée une seule fois au total de maisons exposées, comme Heranval.

    Heranval et al. ajustent leur régression sur 15 points *annuels nationaux*
    (1,7 million de maisons en moyenne, 4,7 millions en 2003) et l'appliquent une
    fois au comptage agrégé : leur tableau 4 se reproduit exactement ainsi. La
    somme commune par commune de SwiGAN retrancherait l'ordonnée à l'origine
    autant de fois qu'il y a de communes éligibles — d'où la correction de 6.1.
    Le résultat n'est pas le même, et l'écart est la mesure de ce choix.
    """
    return np.maximum(a * np.asarray(x_total, dtype=float) - b, 0.0)


# --------------------------------------------------------------------------- #
# Maillon 2 — comptage des bâtiments exposés par commune                       #
# --------------------------------------------------------------------------- #

# Libellés des définitions d'exposition, repris de la partie 2 de l'étude.
EXPOSITIONS = {
    "P0": "toute maison de la commune (lecture littérale)",
    "A2": "commune portant du Moyen/Fort",
    "A1": "commune majoritairement Moyen/Fort",
    "B": "maison en aléa Moyen ou Fort (aléa par bâtiment)",
}


def comptages(dep: str | None = None) -> pd.DataFrame:
    """x_c par commune, sous chacune des quatre définitions d'exposition.

    L'aléa manquant est imputé par la classe majoritaire de la commune — même
    convention que le reste de l'étude. Une commune dont *aucune* maison ne porte
    d'aléa observé n'est pas imputable : `alea_connu` vaut alors False et les
    colonnes A1/A2/B y sont laissées à NaN, pour que l'absence de donnée ne se
    confonde pas avec une exposition nulle.
    """
    fichier = ROOT / "output" / (f"maisons_dep{dep}.csv" if dep else "maisons_grand_est.csv")
    if not fichier.exists():
        raise SystemExit(f"fichier d'exposition absent : {fichier}\n"
                         "  lancer d'abord : python3 extract_maisons.py")

    # Les départements extraits avant l'ajout de `batiment_groupe_risques` n'ont
    # pas de colonne `alea_argile` du tout — l'agrégat régional la porte quand
    # même, `UNION ALL BY NAME` l'ayant remplie de NULL. On rétablit ici la même
    # convention pour qu'un fichier départemental ancien reste lisible.
    colonnes = duckdb.sql(f"SELECT * FROM read_csv('{fichier}', header=true) LIMIT 0").df()
    alea = "alea_argile" if "alea_argile" in colonnes.columns else "NULL AS alea_argile"

    df = duckdb.sql(f"""
        WITH src AS (
            SELECT code_departement_insee AS dep,
                   code_commune_insee     AS commune,
                   {alea}
            FROM read_csv('{fichier}', header=true)
        ),
        majo AS (
            SELECT commune, alea_argile,
                   row_number() OVER (PARTITION BY commune
                                      ORDER BY count(*) DESC, alea_argile) AS rk
            FROM src WHERE alea_argile IS NOT NULL GROUP BY 1, 2
        ),
        impute AS (
            SELECT s.dep, s.commune,
                   coalesce(s.alea_argile, j.alea_argile) AS alea
            FROM src s
            LEFT JOIN (SELECT commune, alea_argile FROM majo WHERE rk = 1) j
                   USING (commune)
        )
        SELECT dep, commune,
               count(*)                                                    AS n_maisons,
               count(alea)                                                 AS n_alea,
               sum(CASE WHEN alea IN ('Moyen', 'Fort') THEN 1 ELSE 0 END)  AS n_mf
        FROM impute GROUP BY 1, 2 ORDER BY 2
    """).df()

    df["commune"] = df["commune"].astype(str).str.zfill(5)
    df["alea_connu"] = df["n_alea"] > 0
    df["P0"] = df["n_maisons"].astype(float)
    connu = df["alea_connu"]
    df["A2"] = np.where(connu & (df["n_mf"] > 0), df["n_maisons"], 0.0)
    df["A1"] = np.where(connu & (df["n_mf"] > df["n_maisons"] / 2), df["n_maisons"], 0.0)
    df["B"] = df["n_mf"].astype(float)
    df.loc[~connu, ["A1", "A2", "B"]] = np.nan
    return df.set_index("commune")


# --------------------------------------------------------------------------- #
# Maillon 3 — coût d'une commune éligible                                      #
# --------------------------------------------------------------------------- #

def couts_communaux(cpt: pd.DataFrame, exposition: str, f_cout) -> pd.Series:
    """C(x_c) en euros, pour chaque commune, sous la définition retenue.

    Deux exclusions, pour la même raison : une commune dont l'aléa est inconnu
    n'a pas d'exposition mesurable, et une commune dont le comptage est nul n'a
    rien d'exposé. Les facturer à x = 0 leur imputerait le terme fixe a de
    l'équation 6.1 sans le comptage qui va avec.
    """
    x = cpt[exposition]
    x = x[x.notna() & (x > 0)]
    return pd.Series(f_cout(x.values), index=x.index, name="cout")


# --------------------------------------------------------------------------- #
# Maillon 1 — éligibilité CatNat : lue, non calculée                           #
# --------------------------------------------------------------------------- #

ALIAS = {
    "commune": ["code_commune_insee", "code_insee", "insee_com", "commune", "code_commune"],
    "annee": ["annee", "année", "year", "an"],
    "trajectoire": ["trajectoire", "trajectory", "traj", "simulation", "membre", "member"],
    "eligible": ["eligible", "éligible", "eligibilite", "is_eligible"],
}


def charger_eligibilite(chemin: Path, annees: tuple[int, int]) -> pd.DataFrame:
    """Table longue (trajectoire, annee, commune) des communes éligibles."""
    lecteur = "read_parquet" if chemin.suffix.lower() in (".parquet", ".pq") else "read_csv"
    opts = "" if lecteur == "read_parquet" else ", header=true"
    brut = duckdb.sql(f"SELECT * FROM {lecteur}('{chemin}'{opts})").df()

    ren, dispo = {}, {c.lower(): c for c in brut.columns}
    for cible, noms in ALIAS.items():
        trouve = next((dispo[n] for n in noms if n in dispo), None)
        if trouve is not None:
            ren[trouve] = cible
    df = brut.rename(columns=ren)

    manquantes = [c for c in ("commune", "annee") if c not in df.columns]
    if manquantes:
        raise SystemExit(f"{chemin} : colonnes introuvables {manquantes}\n"
                         f"  colonnes lues : {list(brut.columns)}")

    if "eligible" in df.columns:
        elig = df["eligible"]
        if elig.dtype == object:
            elig = elig.astype(str).str.strip().str.lower().isin(["true", "1", "oui", "yes"])
        df = df[elig.astype(bool)]
    if "trajectoire" not in df.columns:
        df["trajectoire"] = 0

    df["commune"] = df["commune"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(5)
    df["annee"] = df["annee"].astype(int)
    df = df[df["annee"].between(*annees)]
    return df[["trajectoire", "annee", "commune"]].drop_duplicates()


# --------------------------------------------------------------------------- #
# Agrégation et lecture des résultats                                          #
# --------------------------------------------------------------------------- #

# Les trois façons de passer des comptages communaux à un coût régional.
# L'article ne tranche pas explicitement entre les deux premières (cf. docstring).
def agreger(x, agregation: str, f_cout) -> float:
    """Coût total d'un jeu de communes portant les comptages x, en euros."""
    x = np.asarray(x, dtype=float)
    if agregation == "commune":
        return float(f_cout(x).sum())
    if agregation == "region":
        return float(f_cout(x.sum()))
    return float(cout_heranval(x.sum()))


AGREGATIONS = {
    "commune": "somme de C(x_c) sur les communes éligibles",
    "region": "équation 6.1 appliquée une fois au total exposé X",
    "heranval": "C0 = a X - b, le modèle source non corrigé",
}


def couts_par_trajectoire_annee(elig: pd.DataFrame, cout: pd.Series, cpt: pd.DataFrame,
                                exposition: str, agregation: str, f_cout) -> pd.DataFrame:
    """Coût de chaque couple (trajectoire, année), en euros."""
    df = elig.join(cout, on="commune").join(cpt[exposition].rename("x"), on="commune")
    df = df.dropna(subset=["cout"])
    out = (df.groupby(["trajectoire", "annee"])
             .agg(cout=("cout", "sum"), communes=("commune", "size"), batiments=("x", "sum"))
             .reset_index())
    if agregation == "region":
        out["cout"] = f_cout(out["batiments"].values)
    elif agregation == "heranval":
        out["cout"] = cout_heranval(out["batiments"].values)
    return out


def var(x: np.ndarray, niveau: float) -> float:
    """Value-at-Risk : quantile de la perte annuelle, toutes trajectoires confondues."""
    return float(np.quantile(x, niveau))


def afficher_distribution(par_ta: pd.DataFrame, annees: tuple[int, int]) -> None:
    """Panneau (a) de la figure 13 : la distribution des coûts, lue en chiffres."""
    n_traj = par_ta["trajectoire"].nunique()
    grille = pd.MultiIndex.from_product(
        [par_ta["trajectoire"].unique(), range(annees[0], annees[1] + 1)],
        names=["trajectoire", "annee"])
    # Une trajectoire sans commune éligible une année donnée coûte 0, et doit
    # peser dans la distribution : la table longue ne porte que les lignes
    # éligibles, on complète donc la grille avant tout quantile.
    plein = (par_ta.set_index(["trajectoire", "annee"])
                   .reindex(grille).fillna(0.0).reset_index())

    print("\n" + "=" * 78)
    print(f"COÛT ANNUEL — {n_traj} trajectoire(s), {annees[0]}-{annees[1]}")
    print("=" * 78)
    print(f"  {'année':>6s} {'moyenne':>11s} {'médiane':>11s} {'D1':>11s} {'D9':>11s} "
          f"{'communes':>9s}")
    for an, g in plein.groupby("annee"):
        d1, med, d9 = np.quantile(g["cout"], [0.1, 0.5, 0.9])
        print(f"  {an:6d} {g['cout'].mean() / 1e6:10.1f}M {med / 1e6:10.1f}M "
              f"{d1 / 1e6:10.1f}M {d9 / 1e6:10.1f}M {g['communes'].mean():9.0f}")

    annuel = plein["cout"].values
    print(f"\n  perte annuelle, toutes trajectoires et années confondues "
          f"({len(annuel):,} points)")
    print(f"    moyenne {annuel.mean() / 1e6:.1f} MEUR, "
          f"médiane {np.median(annuel) / 1e6:.1f} MEUR, "
          f"maximum {annuel.max() / 1e6:.1f} MEUR")
    for niv in (0.95, 0.99, 0.995):
        print(f"    VaR {niv * 100:5.1f} % : {var(annuel, niv) / 1e6:8.1f} MEUR")

    par_traj = plein.groupby("trajectoire")["cout"].agg(["sum", "max"])
    print(f"\n  cumul {annees[0]}-{annees[1]} par trajectoire : "
          f"médiane {par_traj['sum'].median() / 1e6:.0f} MEUR, "
          f"D9 {par_traj['sum'].quantile(0.9) / 1e6:.0f} MEUR")
    print(f"  pire année de chaque trajectoire : "
          f"médiane {par_traj['max'].median() / 1e6:.1f} MEUR, "
          f"D9 {par_traj['max'].quantile(0.9) / 1e6:.1f} MEUR")


def afficher_enveloppe(x: pd.Series, agreger) -> None:
    """En l'absence de table d'éligibilité : ce que l'équation 6.1 permet de dire.

    Le coût ne dépend que du jeu de communes éligibles. Pour une part donnée, on
    encadre : les communes les moins peuplées (borne basse), un tirage uniforme
    (cas central), les plus peuplées (borne haute). Le coût étant croissant avec
    le comptage sous les trois agrégations, trier par comptage suffit.
    """
    tri = np.sort(x.values)
    total = agreger(tri)
    rng = np.random.default_rng(RNG)

    print("\n" + "=" * 78)
    print("ENVELOPPE — aucune table d'éligibilité fournie")
    print("=" * 78)
    print(f"  {len(x):,} communes exposées, coût d'une année à éligibilité totale : "
          f"{total / 1e6:.1f} MEUR")
    print("  Tous les montants de ce tableau sont des coûts ANNUELS — le coût d'une")
    print(f"  année où la part indiquée de communes est éligible, pas un cumul {ANNEES_ARTICLE[0]}-"
          f"{ANNEES_ARTICLE[1]}.")
    print(f"\n  {'part éligible':>14s} {'communes':>9s} {'borne basse':>12s} "
          f"{'tirage uniforme':>16s} {'borne haute':>12s}")
    for part in (0.05, 0.10, 0.25, 0.50, 0.75, 1.00):
        k = max(1, int(round(len(tri) * part)))
        unif = np.mean([agreger(x.values[rng.choice(len(tri), k, replace=False)])
                        for _ in range(200)])
        print(f"  {part * 100:13.0f}% {k:9,} {agreger(tri[:k]) / 1e6:11.1f}M "
              f"{unif / 1e6:15.1f}M {agreger(tri[-k:]) / 1e6:11.1f}M")

    if total <= 0:
        print("\n  Coût nul sur tout le balayage : l'agrégation retenue est hors de son")
        print("  domaine sur ce périmètre (cf. « heranval » plus haut).")
        return
    print("\n  Lecture inverse — part de communes qu'il faudrait déclarer éligibles,")
    print("  sous tirage uniforme, pour retrouver les montants de la section 6.3 :")
    for cible, nom in CIBLES_6_3:
        part = f"{cible / total * 100:5.1f} %" if cible <= total else "hors d'atteinte"
        print(f"    {nom:42s} -> {part}")
    print("\n  L'écart entre les deux colonnes extrêmes est d'un ordre de grandeur")
    print("  à part éligible fixée : le coût dépend bien plus de *quelles* communes")
    print("  sont éligibles que de leur nombre. C'est ce que seul le maillon 6.1,")
    print("  absent du dépôt, peut trancher.")


# --------------------------------------------------------------------------- #

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dep", default=None,
                    help="un département (défaut : tout le Grand Est)")
    ap.add_argument("--eligibilite", type=Path, default=None,
                    help="table CSV/Parquet des communes éligibles CatNat (cf. docstring)")
    ap.add_argument("--exposition", default="P0", choices=list(EXPOSITIONS),
                    help="définition du périmètre exposé (défaut : P0, lecture littérale)")
    ap.add_argument("--lecture", default=LECTURE_DEFAUT, choices=list(LECTURES),
                    help="réparation retenue de l'équation 6.1 (cf. LECTURES)")
    ap.add_argument("--b", type=float, default=None,
                    help="surcharge la constante b de la lecture retenue")
    ap.add_argument("--k-negatif", action="store_true",
                    help="rétablir le signe imprimé k = -a/(5b)")
    ap.add_argument("--agregation", default="commune", choices=list(AGREGATIONS),
                    help="commune : somme de C(x_c) ; region : 6.1 une fois sur le total "
                         "exposé ; heranval : C0 = aX - b, le modèle source non corrigé")
    ap.add_argument("--annees", type=int, nargs=2, default=list(ANNEES_ARTICLE),
                    metavar=("DEBUT", "FIN"), help="fenêtre de projection")
    ap.add_argument("--csv", type=Path, default=None,
                    help="écrire le coût par (trajectoire, année) dans ce fichier")
    a = ap.parse_args()
    annees = (a.annees[0], a.annees[1])
    lecture = LECTURES[a.lecture]
    b = lecture["b"] if a.b is None else a.b
    f_cout = partial(cout_commune, b=b, lecture=a.lecture, k_negatif=a.k_negatif)

    cpt = comptages(a.dep)
    perimetre = f"dép {a.dep}" if a.dep else "Grand Est"
    print(f"\n{perimetre} — {int(cpt['n_maisons'].sum()):,} maisons, "
          f"{len(cpt):,} communes")
    signe, k = ("-", -A_EURO / (5 * b)) if a.k_negatif else ("+", A_EURO / (5 * b))
    portee = 1 / abs(k)
    print(f"équation 6.1, lecture « {a.lecture} » : {lecture['forme']}")
    print(f"  a = {A_EURO}, b = {b:,.1f}, k = {signe}a/(5b) = {k:.6g} "
          f"-> 1/|k| = {portee:,.1f} bâtiments" if portee < 1000 else
          f"  a = {A_EURO}, b = {b:,.1f}, k = {signe}a/(5b) = {k:.6g} "
          f"-> 1/|k| = {portee:,.0f} bâtiments")
    print(f"  {lecture['note']}")
    print(f"  coût marginal effectif : {(f_cout(10001) - f_cout(10000)):.1f} EUR/bâtiment")

    sans_alea = int((~cpt["alea_connu"]).sum())
    if sans_alea:
        print(f"attention : {sans_alea:,} communes sur {len(cpt):,} sans aléa argile "
              f"renseigné ({int(cpt.loc[~cpt['alea_connu'], 'n_maisons'].sum()):,} maisons) ;"
              f"\n  seule la définition P0 y est calculable — cf. docstring.")

    print("\n" + "=" * 78)
    print("PÉRIMÈTRE EXPOSÉ — coût d'UNE année où toutes les communes seraient éligibles")
    print("=" * 78)
    print(f"  {'':2s} {'définition':48s} {'communes':>8s} {'bâtiments':>10s} "
          f"{'commune':>8s} {'region':>8s}")
    retenu = None
    for nom, libelle in EXPOSITIONS.items():
        c = couts_communaux(cpt, nom, f_cout)
        n = int(cpt[nom].sum(skipna=True))
        par_com, une_fois = c.sum(), float(f_cout(n)) if n else 0.0
        marque = ">" if nom == a.exposition else " "
        if nom == a.exposition:
            retenu = (n, par_com, une_fois)
        print(f"  {marque:2s} {nom + ' ' + libelle:48s} {len(c):8,} {n:10,} "
              f"{par_com / 1e6:7.1f}M {une_fois / 1e6:7.1f}M")

    n, par_com, une_fois = retenu
    porte = 1 / (A_EURO / (5 * B_EURO_DEFAUT))
    print("\n  « commune » somme C(x_c) sur les communes éligibles ; « region » applique")
    print("  6.1 une fois au total X. L'article ne tranche pas entre les deux ; l'écart")
    print(f"  vaut ici {une_fois / par_com - 1:+.1%}, et reste borné parce que "
          f"1/k = {porte:,.0f}")
    print("  bâtiments, très au-delà de tout X régional : les deux coïncident au premier ordre.")
    print("  Le modèle source non corrigé, C0 = a X - b, donnerait "
          f"{cout_heranval(n) / 1e6:.1f} MEUR ;")
    print(f"  il s'annule sous {B_EURO_DEFAUT / A_EURO:,.0f} maisons, étant ajusté "
          "sur des totaux")
    print("  nationaux de 1,7 à 4,7 millions de maisons.")

    cout = couts_communaux(cpt, a.exposition, f_cout)
    if cout.empty:
        raise SystemExit(f"définition {a.exposition} non calculable sur ce périmètre : "
                         "aucun aléa argile renseigné.")

    if a.eligibilite is None:
        x_exposees = cpt[a.exposition].dropna()
        x_exposees = x_exposees[x_exposees > 0]
        afficher_enveloppe(x_exposees, partial(agreger, agregation=a.agregation,
                                               f_cout=f_cout))
        print("\n  Pour la projection réelle : --eligibilite <table>.")
        return 0

    elig = charger_eligibilite(a.eligibilite, annees)
    inconnues = set(elig["commune"]) - set(cpt.index)
    print(f"\néligibilité : {len(elig):,} lignes commune-année-trajectoire, "
          f"{elig['trajectoire'].nunique()} trajectoire(s), "
          f"{elig['commune'].nunique():,} communes distinctes")
    if inconnues:
        print(f"  {len(inconnues):,} communes éligibles absentes du parc extrait, "
              f"ignorées (ex. {sorted(inconnues)[:3]})")

    par_ta = couts_par_trajectoire_annee(elig, cout, cpt, a.exposition,
                                         a.agregation, f_cout)
    if par_ta.empty:
        raise SystemExit("aucune commune éligible n'a d'exposition mesurable.")
    afficher_distribution(par_ta, annees)

    if a.csv:
        par_ta.to_csv(a.csv, index=False)
        print(f"\ncoûts par (trajectoire, année) écrits dans {a.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
