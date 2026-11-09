#!/usr/bin/env python3
"""Taux de dommage sécheresse : l'effet année remplacé par l'intensité du SWI.

Le GLM sécheresse de glm_taux_dommage.py porte un effet par année de
survenance. Cet effet mêle l'inflation des coûts de réparation, l'intensité de
la sécheresse de l'année et la maturité des sinistres récents, pas encore
réglés. Il ne se projette pas : une trajectoire de SWI générée n'a pas
d'année de sinistre observée. Ici :

  - l'inflation est retirée de la charge avant l'ajustement, par l'indice FFB
    du coût de la construction (indice_ffb.py), moyenne de l'année de
    survenance, en euros de --annee-reference (2024) :
        ch = CH_TOT × ICC(référence) / ICC(survenance)
    comme CCR le fait pour le coût moyen des sinistres ONRN ;
  - l'intensité entre par le rang du SWI de la commune de la maison l'année
    du sinistre, dans la fenêtre glissante de 30 ans du critère CatNat
    (1 = année la plus sèche des 30, 30 = la plus humide). Le rang d'une
    commune est celui de sa maille la plus sèche, celle qui la rend éligible
    (cout_secheresse_departements.py). Cinq formes (--intensite) :
        rang     pente unique sur le rang (défaut)
        classes  quatre classes de période de retour empirique, 30 / rang :
                   exceptionnelle     rang 1      >= 30 ans
                   retour 10-30 ans   rang 2-3    10 à < 30 ans
                   retour 5-10 ans    rang 4-6    5 à < 10 ans
                   ordinaire          rang 7-30   < 5 ans, la référence
                 exp(coef) d'une classe est l'écart de coût à une sécheresse
                 ordinaire ; une classe trop rare rejoint sa voisine ;
        sswi, deficit, deficit2
                 pente unique sur une magnitude de intensite_swi.py, sur
                 climatologie fixe 1981-2010 et sans plafond (moyenne des
                 mailles de la commune pondérée par la surface) ;
  - deux modèles sont comparés, sans et avec contrôle temporel :
        A  niveaux + période + aléa + département + intensité
        B  A + tendance linéaire de l'année de survenance, d'origine
           --annee-reference (--controle tendance, défaut),
           ou A + un effet fixe par année (--controle annee)
    GLM Gamma, lien log, offset log(V), V = 1 800 €/m² × surface de plancher
    (la valeur de reconstruction de glm_taux_dommage.py, ici en euros de
    l'année de référence), erreurs-types groupées par maison, ou par année de
    survenance (--grappes annee) : l'intensité est commune aux sinistres d'une
    même année et d'une même commune, et le groupement par maison la croit
    plus précise qu'elle n'est. Avec une vingtaine d'années, le groupement par
    année reste lui-même approximatif ; severite_secheresse_swi.py pose un
    effet épisode et un effet année aléatoires sur les sinistres de France.

LECTURE
-------
exp(coef du rang) est l'effet multiplicatif d'un rang de plus, c'est-à-dire
d'une année un cran moins sèche dans sa fenêtre de 30 ans : sous 1, les
sécheresses plus rares coûtent plus cher par sinistre. Le script imprime aussi
le rapport rang 1 / rang 30. Avec la tendance, le rang n'est identifié que par
ses écarts à la tendance ; avec un effet fixe par année, que par les écarts
entre communes d'une même année. Le contrôle temporel absorbe ce qui reste de
la maturité des sinistres récents et des changements de gestion, mais aussi la
part de l'intensité qui suit le temps si les sécheresses se rapprochent : la
corrélation entre rang et année est imprimée.

VALIDATION
----------
Le modèle devra prédire la sévérité d'années qu'il n'a pas vues : validation
croisée par année, chaque année de survenance retirée tour à tour. Déviance
Gamma hors échantillon, comparée à celle d'un modèle sans rang ni temps
(caractéristiques de la maison et département). Avec un effet fixe par année,
l'année retirée n'a pas d'effet estimé : le modèle B n'est pas validé ainsi.

RÉSERVES
--------
- Peu d'années de sécheresse distinctes : la pente du rang repose sur elles,
  et les sinistres d'une même année partagent leur rang à quelques mailles
  près. Les erreurs-types groupées par maison n'en tiennent pas compte.
- L'ICC FFB suit le coût d'un immeuble de rapport à Paris, pas celui des
  reprises en sous-œuvre ; la charge est déflatée à l'année de survenance,
  alors qu'un sinistre sécheresse se règle souvent deux à quatre ans après.
- Sévérité conditionnelle, comme glm_taux_dommage.py : ni fréquence, ni
  sinistres absorbés par la franchise.
- Biais de sélection : il n'y a de sinistre sécheresse que dans une commune
  reconnue, sur demande du maire, sous des critères qui ont changé (circulaire
  de 2019 jusqu'aux sécheresses de 2023). Une commune reconnue malgré une
  sécheresse ordinaire l'a sans doute été sur ses dommages, et ses sinistres
  ne sont pas comparables à ceux d'une sécheresse exceptionnelle où tout le
  monde déclare. L'effet fixe par année retire la sélection commune à une
  année (règles, campagne de déclaration), pas celle qui joue entre communes
  d'une même année.

CONFIDENTIALITÉ
---------------
Comme glm_taux_dommage.py : base lue en place par son chemin (CLE, ANNEE,
PERIL, CH_TOT), doublons de CLE retirés, jeu de modélisation jamais écrit.
L'écran et les sorties ne portent que des coefficients et des agrégats ; à
l'écran, les années de moins de --min-modalite sinistres sont regroupées.

Prérequis : apparier_sinistres_maisons.py, extract_maisons.py, indice_ffb.py
(sinon le site de la FFB est interrogé), contours en cache (prix_m2_communes.py).
Usage : python3 glm_secheresse_swi.py [--intensite rang|classes|sswi|deficit|deficit2]
                                      [--controle tendance|annee] [--grappes maison|annee]
                                      [--annee-reference 2024] [--sin CHEMIN]
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import spearmanr

from apparier_sinistres_maisons import lire_sinistres, trouver_base
from cout_secheresse_departements import (CODE, SWI, charger_swi, correspondance,
                                          mailles_anormales, rang_communes)
from eligibilite_catnat import FENETRE_DEFAUT, RANG_5_ANS, RANG_10_ANS
from extract_maisons import GRAND_EST
from glm_taux_dommage import (APPARIEMENT, COLONNES_SIN, COUT_M2, MAISONS, MIN_MODALITE,
                              ajuster, categoriser, comparer, construire, qualite, tableau,
                              table_modalites, tests_globaux)
from indice_ffb import indice_annuel, indice_ffb
from intensite_swi import INDICES, charger_intensite
from prix_m2_communes import referentiel

ROOT = Path(__file__).resolve().parent
SORTIE = ROOT / "output" / "glm_secheresse_swi_{variante}_{quoi}.csv"

ANNEE_REFERENCE = 2024
GRAPPES = {"maison": "batiment_groupe_id", "annee": "annee_num"}
FACTEURS = ["niveaux", "periode", "alea", "dep"]
# Classes d'intensité, de la moins sèche à la plus sèche, et leur rang maximal.
CLASSES = ["ordinaire", "retour 5-10 ans", "retour 10-30 ans", "exceptionnelle"]
RANG_MAX_CLASSE = [FENETRE_DEFAUT, RANG_5_ANS, RANG_10_ANS, 1]
# Références : le département du GLM sécheresse de glm_taux_dommage.py, la
# sécheresse ordinaire.
REFS = {"dep": "57", "intensite": "ordinaire"}


# --------------------------------------------------------------------------- #
# Données                                                                      #
# --------------------------------------------------------------------------- #

def deflateur(annees: pd.Series, reference: int) -> pd.Series:
    """ICC(référence) / ICC(année), moyennes annuelles de l'indice FFB."""
    icc = indice_annuel(indice_ffb(debut=int(annees.min())))
    manquantes = sorted({*annees.unique(), reference} - set(icc.index))
    if manquantes:
        raise SystemExit(f"indice FFB annuel indisponible pour {manquantes} "
                         f"(années complètes {icc.index.min()}-{icc.index.max()})")
    return icc[reference] / annees.map(icc)


def rangs_swi(annees: list[int], dossier: Path) -> pd.DataFrame:
    """Rang du SWI par (commune, année), pour les années qui ont une fenêtre complète."""
    swi, mailles = charger_swi(dossier)
    premiere = swi["annee"].min() + FENETRE_DEFAUT - 1
    annees = [a for a in annees if premiere <= a <= swi["annee"].max()]
    communes = referentiel()
    communes = communes[communes["code_departement"].isin(GRAND_EST)].reset_index(drop=True)
    return rang_communes(mailles_anormales(swi, annees), correspondance(communes, mailles))


def classe_intensite(rang: pd.Series) -> pd.Series:
    """Classe de période de retour empirique (30 / rang) d'un rang sur 30 ans."""
    bornes = [0, *RANG_MAX_CLASSE[::-1][:-1], FENETRE_DEFAUT]
    return pd.cut(rang, bornes, labels=CLASSES[::-1]).astype(str)


def assembler(sin: pd.DataFrame, app: pd.DataFrame, maisons: pd.DataFrame, cout_m2: float,
              reference: int, dossier_swi: Path) -> tuple[pd.DataFrame, list[tuple[str, int]]]:
    """Sinistres sécheresse en euros constants, avec le rang SWI de leur commune-année."""
    d, etapes = construire(sin, app, maisons, "reconstruction", cout_m2)
    d = d[d["secheresse"]].copy()
    etapes.append(("péril sécheresse", len(d)))
    d["annee_num"] = pd.to_numeric(d["annee"], errors="coerce")
    d = d[d["annee_num"].notna()].astype({"annee_num": int})
    etapes.append(("année de survenance lue", len(d)))
    d["deflateur"] = deflateur(d["annee_num"], reference)
    d["ch"] = d["ch"] * d["deflateur"]
    d["taux"] = d["ch"] / d["valeur"]
    rangs = rangs_swi(sorted(d["annee_num"].unique()), dossier_swi)
    d = d.merge(rangs.rename(columns={"annee": "annee_num"}), on=[CODE, "annee_num"], how="left")
    d = d[d["rang"].notna()].copy()
    etapes.append(("rang SWI de la commune-année connu", len(d)))
    magnitudes = charger_intensite()[[CODE, "annee", *INDICES]]
    d = d.merge(magnitudes.rename(columns={"annee": "annee_num"}), on=[CODE, "annee_num"],
                how="left")
    d["intensite"] = classe_intensite(d["rang"])
    d["tendance"] = d["annee_num"] - reference
    return d, etapes


# --------------------------------------------------------------------------- #
# Validation croisée par année                                                 #
# --------------------------------------------------------------------------- #

def deviance_gamma(y: pd.Series, mu: pd.Series) -> pd.Series:
    return 2 * (-np.log(y / mu) + (y - mu) / mu)


def valider_par_annee(d: pd.DataFrame, modeles: dict[str, list[str]]) -> pd.DataFrame:
    """Déviance Gamma de chaque année de survenance, prédite sans elle."""
    famille = sm.families.Gamma(sm.families.links.Log())
    lignes = []
    for an in sorted(d["annee_num"].unique()):
        app, test = d[d["annee_num"] != an], d[d["annee_num"] == an]
        ligne = {"annee": an, "n": len(test)}
        for nom, termes in modeles.items():
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = smf.glm("ch ~ " + " + ".join(termes), data=app, family=famille,
                              offset=np.log(app["valeur"])).fit()
            mu = res.predict(test, offset=np.log(test["valeur"]))
            ligne[nom] = deviance_gamma(test["ch"], mu).sum()
        lignes.append(ligne)
    return pd.DataFrame(lignes)


# --------------------------------------------------------------------------- #
# Rapport (agrégats uniquement)                                                #
# --------------------------------------------------------------------------- #

def _regrouper_annees(d: pd.DataFrame, mini: int) -> pd.Series:
    """Année en clair si elle compte au moins `mini` sinistres, « autres » sinon."""
    n = d["annee_num"].map(d["annee_num"].value_counts())
    return d["annee_num"].astype(str).where(n >= mini, "autres")


def decrire(d: pd.DataFrame, etapes: list[tuple[str, int]], doublons: int, reference: int,
            mini: int) -> None:
    print(f"\ndoublons de CLE retirés de la base : {doublons:,}")
    print("effectif après chaque filtre")
    for libelle, n in etapes:
        print(f"  {libelle:<40}: {n:>7,}")
    print(f"\ncharge déflatée par l'ICC FFB en euros {reference} : facteur de "
          f"{d['deflateur'].min():.2f} à {d['deflateur'].max():.2f}")
    print(f"taux CH / V en euros {reference} : " + ", ".join(
        f"q{int(k * 100)} {v:.2%}" for k, v in d["taux"].quantile([.1, .5, .9]).items()))
    r = d["rang"]
    print(f"rang SWI des sinistres : médiane {r.median():.0f} ; rang <= {RANG_10_ANS} "
          f"(>= 10 ans) {(r <= RANG_10_ANS).mean():.0%}, <= {RANG_5_ANS} (>= 5 ans) "
          f"{(r <= RANG_5_ANS).mean():.0%}, > {RANG_5_ANS} {(r > RANG_5_ANS).mean():.0%}")
    rho = spearmanr(d["rang"], d["annee_num"])[0]
    print(f"corrélation de rang (Spearman) entre rang SWI et année de survenance : {rho:+.2f}")
    print(f"\npar année de survenance (années de moins de {mini} sinistres regroupées)")
    t = d.groupby(_regrouper_annees(d, mini)).agg(
        sinistres=("ch", "size"), rang_median=("rang", "median"),
        part_rang_le_3=("rang", lambda s: (s <= RANG_10_ANS).mean()),
        taux_median=("taux", "median"))
    print(t.to_string(formatters={"rang_median": "{:.0f}".format,
                                  "part_rang_le_3": "{:.0%}".format,
                                  "taux_median": "{:.2%}".format}))
    # Avec un effet fixe par année, une classe n'est comparée à une autre que
    # dans les années où les deux coexistent : ce tableau dit lesquelles.
    print("\nsinistres par année de survenance et classe d'intensité")
    x = pd.crosstab(_regrouper_annees(d, mini), d["intensite"]).reindex(columns=CLASSES,
                                                                         fill_value=0)
    x.loc["total"] = x.sum()
    print(x.to_string())
    g = d.groupby("intensite")["taux"]
    med = g.median()[g.size() >= mini].reindex(CLASSES).dropna()
    print("taux médian par classe : " + ", ".join(f"{c} {v:.2%}" for c, v in med.items()))


def effet_rang(res) -> str:
    """Rapport des taux attendus au rang 1 et au rang 30, avec son IC95."""
    k = 1 - FENETRE_DEFAUT
    b, (lo, hi) = res.params["rang"], res.conf_int().loc["rang"]
    return f"{np.exp(k * b):.2f} [{np.exp(k * hi):.2f}-{np.exp(k * lo):.2f}]"


def effet_magnitude(res, d: pd.DataFrame, indice: str) -> str:
    """Rapport des taux attendus aux déciles 9 et 1 de la magnitude, avec son IC95."""
    k = d[indice].quantile(.9) - d[indice].quantile(.1)
    b, (lo, hi) = res.params[indice], res.conf_int().loc[indice]
    return f"{np.exp(k * b):.2f} [{np.exp(k * lo):.2f}-{np.exp(k * hi):.2f}]"


def rapporter_validation(v: pd.DataFrame, base: str, modeles: list[str],
                         mini: int) -> pd.DataFrame:
    """Gains hors échantillon, total et par année (années de moins de `mini` sinistres regroupées)."""
    print("\nvalidation croisée par année : déviance Gamma hors échantillon "
          f"(gain = 1 - D / D[{base}])")
    total = v[[base, *modeles]].sum()
    for nom in modeles:
        gagnees = (v[nom] < v[base]).sum()
        print(f"  {nom:<28} gain {1 - total[nom] / total[base]:+.1%} ; meilleur que la "
              f"référence {gagnees} années sur {len(v)}")
    par_an = v.assign(annee=v["annee"].astype(str).where(v["n"] >= mini, "autres"))
    par_an = par_an.groupby("annee")[["n", base, *modeles]].sum()
    par_an.loc["total"] = par_an.sum()
    gains = pd.DataFrame({"sinistres": par_an["n"]} | {
        f"gain {nom}": 1 - par_an[nom] / par_an[base] for nom in modeles})
    print(gains.to_string(formatters={f"gain {nom}": "{:+.1%}".format for nom in modeles}))
    return par_an.rename(columns={"n": "sinistres"}).join(gains.drop(columns="sinistres"))


# --------------------------------------------------------------------------- #

def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--intensite", choices=["rang", "classes", *INDICES], default="rang",
                   help="pente unique sur le rang, quatre classes de période de retour, ou "
                        "pente sur une magnitude de intensite_swi.py (défaut rang)")
    p.add_argument("--controle", choices=["tendance", "annee"], default="tendance",
                   help="contrôle temporel du modèle B (défaut tendance)")
    p.add_argument("--grappes", choices=list(GRAPPES), default="maison",
                   help="groupement des erreurs-types (défaut maison)")
    p.add_argument("--annee-reference", type=int, default=ANNEE_REFERENCE,
                   help=f"euros de cette année, origine de la tendance (défaut {ANNEE_REFERENCE})")
    p.add_argument("--cout-m2", type=float, default=COUT_M2,
                   help=f"coût de construction en €/m² de plancher (défaut {COUT_M2})")
    p.add_argument("--sin", type=Path, help="chemin de SIN_EVT_NAT.csv")
    p.add_argument("--swi", type=Path, default=SWI, help="dossier des parquets SWI du Grand Est")
    p.add_argument("--min-modalite", type=int, default=MIN_MODALITE,
                   help=f"effectif minimal d'une modalité (défaut {MIN_MODALITE})")
    a = p.parse_args()
    for f, script in [(APPARIEMENT, "apparier_sinistres_maisons.py"),
                      (MAISONS, "extract_maisons.py")]:
        if not f.exists():
            raise SystemExit(f"{f} introuvable : lancer d'abord {script}")

    chemin = trouver_base(a.sin)
    print(f"Base de sinistres : {chemin}")
    sin, doublons = lire_sinistres(chemin, COLONNES_SIN)
    app = pd.read_csv(APPARIEMENT, dtype={"batiment_groupe_id": str})
    maisons = pd.read_csv(MAISONS, low_memory=False,
                          dtype={"batiment_groupe_id": str, "code_departement_insee": str,
                                 CODE: str})
    d, etapes = assembler(sin, app, maisons, a.cout_m2, a.annee_reference, a.swi)
    swi = {"rang": "rang", "classes": "intensite"}.get(a.intensite, a.intensite)
    if swi in INDICES:
        d = d[d[swi].notna()].copy()
        etapes.append((f"magnitude {swi} connue", len(d)))
    decrire(d, etapes, doublons, a.annee_reference, a.min_modalite)

    facteurs = (FACTEURS + (["intensite"] if swi == "intensite" else [])
                + (["annee"] if a.controle == "annee" else []))
    brut = d
    d, notes = categoriser(d, facteurs, a.min_modalite, REFS, {"intensite": CLASSES})
    print("\nréférences : " + ", ".join(f"{f} = {r}" for f, (r, _) in notes.items()))
    fus = [f"{f} : {', '.join(x)}" for f, (_, x) in notes.items() if x]
    if fus:
        print("  modalités fusionnées — " + " ; ".join(fus))

    base = [f for f in FACTEURS if d[f].nunique() > 1]
    temps = "tendance" if a.controle == "tendance" else "annee"
    modeles = {"A : SWI": base + [swi], f"B : SWI + {a.controle}": base + [swi, temps]}
    res = {nom: ajuster(d, termes, grappe=GRAPPES[a.grappes]) for nom, termes in modeles.items()}
    tabs = [tableau(r, nom) for nom, r in res.items()]
    comparer(tabs)
    print(f"(erreurs-types groupées par {a.grappes})")
    if swi == "rang":
        print(f"\nrapport des taux attendus rang 1 / rang {FENETRE_DEFAUT} [IC95]")
        for nom, r in res.items():
            print(f"  {nom:<28} {effet_rang(r)}")
    elif swi in INDICES:
        print(f"\nrapport des taux attendus aux déciles 9 et 1 de {swi} [IC95]")
        for nom, r in res.items():
            print(f"  {nom:<28} {effet_magnitude(r, d, swi)}")
    if a.controle == "tendance":
        rb = res[f"B : SWI + {a.controle}"]
        b, (lo, hi) = rb.params["tendance"], rb.conf_int().loc["tendance"]
        print(f"tendance du modèle B, en euros constants : {np.exp(b) - 1:+.1%} par an "
              f"[{np.exp(lo) - 1:+.1%} ; {np.exp(hi) - 1:+.1%}]")
    print("\nqualité d'ajustement")
    print(pd.DataFrame({nom: qualite(r) for nom, r in res.items()}).to_string(
        float_format=lambda v: f"{v:,.3f}"))
    tg = pd.concat([tests_globaux(r, nom) for nom, r in res.items()])
    print("\ntests globaux par variable (Wald joint, covariance groupée par maison)")
    for nom, t in tg.groupby("modele", sort=False):
        print(f"[{nom}]")
        print(t.drop(columns="modele").set_index("variable").to_string(
            formatters={"chi2": "{:.1f}".format, "p_valeur": "{:.2g}".format,
                        "p_holm": "{:.2g}".format}))

    reference = "sans SWI ni temps"
    valides = {reference: base, **{n: t for n, t in modeles.items() if "annee" not in t}}
    v = rapporter_validation(valider_par_annee(d, valides), reference,
                             [n for n in valides if n != reference], a.min_modalite)
    if a.controle == "annee":
        print("  (B : l'effet de l'année retirée n'est pas estimable, pas de validation par année)")

    sorties = {"coefs": pd.concat(tabs), "tests": tg,
               "validation": v.rename_axis("annee").reset_index(),
               "modalites": pd.concat([table_modalites(nom, brut, d, facteurs)
                                       for nom in modeles])}
    print()
    prefixe = {"rang": "", "classes": "classes_"}.get(a.intensite, f"{a.intensite}_")
    for quoi, t in sorties.items():
        variante = f"{prefixe}{a.controle}" + ("_grappes_annee" if a.grappes == "annee" else "")
        chemin_sortie = Path(str(SORTIE).format(variante=variante, quoi=quoi))
        chemin_sortie.parent.mkdir(parents=True, exist_ok=True)
        t.to_csv(chemin_sortie, index=False)
        print(f"{quoi} -> {chemin_sortie}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
