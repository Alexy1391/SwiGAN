#!/usr/bin/env python3
"""Taux de destruction sécheresse des communes reconnues : ONRN 1995-2021, prolongé par modèle.

Le coût final part de l'éligibilité CatNat que donne le SWI simulé par SwiGAN
(article, section 6) : pour une commune-année éligible, le nombre de maisons
sinistrées vaut τ × nombre de maisons. Ce script fournit τ, la part des
maisons sinistrées une année de reconnaissance. La probabilité de
reconnaissance n'y figure pas : le SWI en tient lieu.

SOURCE
------
L'indicateur ONRN (CCR, fiche ONRN_Frequence_Sech_9521.pdf) donne pour chaque
commune la fréquence moyenne des sinistres sécheresse indemnisés au titre
CatNat sur 1995-2021 : moyenne des fréquences annuelles nombre de sinistres /
nombre de risques assurés (biens hors véhicules), en classes (0-1, 1-2, 2-5,
5-10, > 10 ‰), avec la représentativité des données (part de marché des
sinistres présents dans la base CCR).

TAUX OBSERVÉ (communes à classe renseignée)
-------------------------------------------
    freq_obs        borne haute de la classe, lecture prudente ; la classe
                    ouverte « > 10 ‰ » est bornée par --borne-ouverte (15 ‰
                    par défaut)
    tau_obs         freq_obs × 27 / R : la fréquence ONRN est une moyenne sur
                    les 27 années, reconnues ou non ; R est le nombre d'années
                    1995-2021 couvertes par un arrêté sécheresse (catnat.csv)
    tau_maison_obs  tau_obs / part des logements situés en maison : le
                    dénominateur ONRN compte tous les risques assurés,
                    appartements compris, alors que la subsidence touche les
                    maisons. On suppose que tous les sinistres portent sur des
                    maisons et qu'un logement vaut un risque (BDNB, Fichiers
                    fonciers). Hypothèse intenable là où les maisons sont
                    minoritaires (sinistres sur les immeubles collectifs) : la
                    correction y multiplie la borne haute 1 ‰ par plus de 3
                    (Mulhouse : 25 %, et la moitié du poids du Haut-Rhin). Sous
                    --part-maisons-min (30 % des logements), la commune est
                    écartée de l'ajustement et reçoit la prédiction.

LA TAILLE DE LA COMMUNE N'EST PAS UNE VARIABLE EXPLICATIVE
----------------------------------------------------------
tau_obs baisse fortement avec la taille de la commune (médiane 13,5 % sous 80
maisons, 2,7 % au-delà de 550). Ce n'est pas un effet mécanique du rapport : si
chaque maison a la même probabilité q d'être touchée, le nombre de sinistres
croît comme le nombre de maisons et la proportion reste q. La pente vient de
la façon dont les communes entrent dans l'échantillon :
  - la part des logements en maison baisse avec la taille, ce qui gonfle le
    dénominateur ONRN (corrigé par tau_maison_obs : une petite partie de la
    pente) ;
  - la reconnaissance suit une demande du maire, déclenchée par des
    sinistres : une petite commune n'est reconnue que si une large part de ses
    maisons est touchée, une grande l'est dès quelques sinistres ;
  - les grandes communes sont reconnues plus souvent (R moyen 2,7 contre 1,3),
    années marginales comprises, ce qui abaisse tau par année.
L'éligibilité issue du SWI ne passe ni par les sinistres ni par la demande du
maire : garder la taille dans le modèle transférerait cette sélection et
prêterait aux petites communes éligibles le taux des seules petites communes
assez touchées pour avoir demandé. Le modèle retenu s'en passe et pondère
chaque commune par ses maisons-années reconnues (× représentativité) : il
estime Σ sinistres / Σ maisons-années, la quantité qui entre dans le coût, où
pèsent surtout les grandes communes, les moins sélectionnées. Le tableau de
validation croisée garde la taille à titre de diagnostic.

MODÈLE
------
Logit fractionnaire (quasi-binomial, Papke-Wooldridge) de tau_maison_obs sur
le département et les parts de maisons en aléa Moyen et Fort (parmi l'aléa
connu ; à défaut, moyenne du département), erreurs-types robustes HC0. Lien
canonique et département en indicatrices : dans chaque département, la
moyenne pondérée des taux ajustés égale celle des taux observés. exp(coef) est
un rapport de cotes, proche d'un effet multiplicatif tant que le taux reste
petit. tau_retenu vaut tau_maison_obs si la commune a une classe et des
maisons majoritaires (au sens de --part-maisons-min), la prédiction sinon ;
il est plafonné à --tau-max (10 %). Le modèle est ajusté sur les taux observés
non plafonnés.

RÉSERVES
--------
- Fréquence, pas taux de destruction au sens du coût : un sinistre compte pour
  un quel que soit son montant.
- Sans la taille, département et aléa expliquent très peu des écarts de taux
  entre communes reconnues (déviance expliquée hors échantillon ≈ 0) : l'aléa
  joue surtout sur la reconnaissance elle-même. Les prédictions restent
  proches de la moyenne de leur département, modulée par l'aléa.
- Le seuil --part-maisons-min est arbitraire ; entre 30 et 60 % les
  coefficients bougent peu.
- tau_maison_obs des petites communes reconnues garde la sélection décrite
  plus haut ; le plafond --tau-max en coupe les excès (environ 400 communes,
  surtout petites et en classe ≥ 5 ‰).
- Borne haute de classe : majorant. Les grandes communes, presque toutes en
  classe 0-1 ‰, sont celles où il majore le plus, et ce sont elles qui pèsent
  dans la moyenne. Le tau au centre de classe est imprimé pour comparaison.
- Moyenne ONRN supposée prise sur 27 ans ; si CCR ne moyenne que les années où
  la commune a des risques en base, tau est surestimé.
- Les communes reconnues sans sinistre répertorié à CCR sont écartées :
  sinistres nuls ou données absentes, la fiche ne permet pas de trancher.
- Les arrêtés des années 1990 couvrent parfois plusieurs années (1989-1998) :
  chaque année couverte compte dans R.

Prérequis : extract_maisons.py (output/maisons_grand_est.csv), exports BDNB
bruts (bdnb_grand_est/dep_*/csv), ../catnat.csv, ../communes-france-2026.parquet.
Usage : python3 taux_destruction_onrn.py [--borne-ouverte 15] [--part-maisons-min 0.3]
                                         [--tau-max 0.10] [--formule "C(dep) + part_mf"]
"""

from __future__ import annotations

import argparse
import re
import sys
import warnings
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.special import xlogy
from scipy.stats import spearmanr
from sklearn.model_selection import KFold

from cout_argile_attendu import TAUX_DEFAUT
from extract_maisons import BDNB, COLS_BATIMENT_GROUPE, GRAND_EST, USAGE_MAISON, _read_csv
from extraire_catnat import CATNAT_DEFAUT, CLE_EPISODE, SOUS_ENSEMBLES, charger, corriger_dates
from glm_taux_dommage import tableau

ROOT = Path(__file__).resolve().parent
ONRN = ROOT / "ONRN_Frequence_SECH_9521.xlsx"
MAISONS = ROOT / "output" / "maisons_grand_est.csv"
COMMUNES = ROOT.parent / "communes-france-2026.parquet"
SORTIE = ROOT / "output" / "taux_destruction_communes.csv"

FEUILLE_FREQUENCE = "Fréq. moy. sinistres"
FEUILLE_REPRESENTATIVITE = "Représentativité"
DEBUT, FIN = 1995, 2021
N_ANNEES = FIN - DEBUT + 1

# Classes ONRN en ‰ : (borne basse, borne haute), None pour la classe ouverte.
CLASSES = {"Entre 0 et 1 ‰": (0, 1), "Entre 1 et 2 ‰": (1, 2), "Entre 2 et 5 ‰": (2, 5),
           "Entre 5 et 10 ‰": (5, 10), "Plus de 10 ‰": (10, None)}
SANS_CLASSE = "Pas de sinistre ou de risque répertoriés à CCR"
BORNE_OUVERTE = 15.0
# Part minimale des logements situés en maison pour corriger le dénominateur.
PART_MAISONS_MIN = 0.3
# Plafond du taux retenu, par maison et année reconnue.
TAU_MAX = 0.10
# Représentativité ONRN, centre de classe : facteur du poids de chaque commune.
REPRESENTATIVITE = {"< 15%": 0.075, "Entre 15 et 30%": 0.225, "Entre 30 et 50%": 0.40,
                    "> 50%": 0.75}

FORMULE = "C(dep) + part_moyen + part_fort"
SPECIFICATIONS = {
    "département": "C(dep)",
    "dép. + part Moyen/Fort": "C(dep) + part_mf",
    "dép. + Moyen + Fort": FORMULE,
    "dép. + Moyen + Fort + taille (diagnostic)": f"{FORMULE} + log_maisons",
}
PLIS = 10
CIBLE = "tau_maison_obs"

NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
RID = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"


# --------------------------------------------------------------------------- #
# Données                                                                      #
# --------------------------------------------------------------------------- #

def lire_feuille(xlsx: Path, nom: str) -> pd.DataFrame:
    """Feuille d'un classeur xlsx, lue sans openpyxl (absent de l'environnement).

    Suffit aux feuilles ONRN : cellules texte ou nombre, sans formule. Les
    lignes vides sont sautées, la première ligne lue est l'en-tête.
    """
    with zipfile.ZipFile(xlsx) as z:
        textes = ["".join(t.text or "" for t in si.iter(f"{NS}t"))
                  for si in ET.fromstring(z.read("xl/sharedStrings.xml")).iter(f"{NS}si")]
        rid = next(s.get(RID) for s in ET.fromstring(z.read("xl/workbook.xml")).iter(f"{NS}sheet")
                   if s.get("name") == nom)
        cible = next(r.get("Target") for r in ET.fromstring(z.read("xl/_rels/workbook.xml.rels"))
                     if r.get("Id") == rid).lstrip("/")
        feuille = ET.fromstring(z.read(cible if cible.startswith("xl/") else f"xl/{cible}"))
    lignes = []
    for row in feuille.iter(f"{NS}row"):
        cellules = {re.match(r"[A-Z]+", c.get("r")).group():
                    textes[int(v.text)] if c.get("t") == "s" else v.text
                    for c in row.iter(f"{NS}c") if (v := c.find(f"{NS}v")) is not None}
        if cellules:
            lignes.append(cellules)
    t = pd.DataFrame(lignes)
    return t.iloc[1:].reset_index(drop=True).set_axis(t.iloc[0].tolist(), axis=1)


def lire_onrn(xlsx: Path, borne_ouverte: float) -> pd.DataFrame:
    """Une ligne par commune : classe ONRN, représentativité, bornes en fraction."""
    freq = lire_feuille(xlsx, FEUILLE_FREQUENCE).iloc[:, :3].set_axis(
        ["code_commune_insee", "libelle_commune", "classe_onrn"], axis=1)
    rep = lire_feuille(xlsx, FEUILLE_REPRESENTATIVITE).iloc[:, [0, 2]].set_axis(
        ["code_commune_insee", "representativite"], axis=1)
    d = freq.merge(rep, on="code_commune_insee", how="left")
    haute = {k: (h or borne_ouverte) for k, (_, h) in CLASSES.items()}
    centre = {k: (b + (h or borne_ouverte)) / 2 for k, (b, h) in CLASSES.items()}
    d["freq_obs"] = d["classe_onrn"].map(haute) / 1000
    d["freq_centre"] = d["classe_onrn"].map(centre) / 1000
    return d


def annees_reconnues(catnat: Path) -> pd.DataFrame:
    """Couples (commune, année) couverts par un arrêté sécheresse, toutes années."""
    cn = charger(catnat)
    cn = cn[cn["type_catastrophe"].isin(SOUS_ENSEMBLES["secheresse_argiles"])].copy()
    cn = corriger_dates(cn).drop_duplicates(CLE_EPISODE)
    debut = cn["date_debut_corrigee"].str[:4].astype(int)
    fin = cn["date_fin_corrigee"].str[:4].astype(int)
    annees = pd.DataFrame({"code_commune_insee": cn["code_geographique"].to_numpy(),
                           "annee": [list(range(a, b + 1)) for a, b in zip(debut, fin)]})
    return annees.explode("annee").astype({"annee": int}).drop_duplicates()


def reconnaissances(catnat: Path) -> pd.DataFrame:
    """Par commune : nombre d'années DEBUT-FIN couvertes par un arrêté sécheresse."""
    annees = annees_reconnues(catnat)
    annees = annees[annees["annee"].between(DEBUT, FIN)]
    return (annees.groupby("code_commune_insee")["annee"].nunique()
            .rename("n_annees_reconnues").reset_index())


def exposition(maisons: Path) -> pd.DataFrame:
    """Par commune : nombre de maisons et parts d'aléa argile parmi l'aléa connu."""
    return duckdb.sql(f"""
        SELECT code_commune_insee,
               count(*)                             AS n_maisons,
               avg((alea_argile = 'Moyen')::DOUBLE) AS part_moyen,
               avg((alea_argile = 'Fort')::DOUBLE)  AS part_fort,
               avg((alea_argile IS NULL)::DOUBLE)   AS part_alea_inconnu
        FROM read_csv('{maisons.as_posix()}', all_varchar = true)
        WHERE usage_ffo = '{USAGE_MAISON}'
        GROUP BY 1
    """).df()


def logements(deps: list[str]) -> pd.DataFrame:
    """Par commune : logements, et logements des bâtiments à usage de maison.

    Lus dans les exports BDNB bruts (nb_log des Fichiers fonciers, usage
    principal BDNB) ; un département dont l'export manque est sauté.
    """
    tables = ["batiment_groupe_ffo_bat", "batiment_groupe",
              "batiment_groupe_synthese_propriete_usage"]
    lectures = []
    for dep in deps:
        csv = BDNB / f"dep_{dep}" / "csv"
        if not all((csv / f"{t}.csv").exists() for t in tables):
            print(f"export BDNB du {dep} incomplet : pas de part de logements en maison")
            continue
        lectures.append(f"""
            SELECT g.code_commune_insee, u.usage_principal_bdnb_open AS usage, f.nb_log
            FROM {_read_csv(csv / 'batiment_groupe_ffo_bat.csv')} f
            JOIN {_read_csv(csv / 'batiment_groupe.csv', COLS_BATIMENT_GROUPE)} g
                USING (batiment_groupe_id)
            LEFT JOIN {_read_csv(csv / 'batiment_groupe_synthese_propriete_usage.csv')} u
                USING (batiment_groupe_id)""")
    if not lectures:
        return pd.DataFrame(columns=["code_commune_insee", "logements", "logements_maisons"])
    return duckdb.sql(f"""
        SELECT code_commune_insee,
               sum(nb_log)                                            AS logements,
               sum(CASE WHEN usage = '{USAGE_MAISON}' THEN nb_log END) AS logements_maisons
        FROM ({' UNION ALL '.join(lectures)})
        GROUP BY 1
    """).df()


def population(parquet: Path) -> pd.DataFrame:
    return pd.read_parquet(parquet, columns=["code_insee", "population"]).rename(
        columns={"code_insee": "code_commune_insee"})


def assembler(onrn: pd.DataFrame, *tables: pd.DataFrame,
              part_maisons_min: float = PART_MAISONS_MIN) -> pd.DataFrame:
    d = onrn[onrn["code_commune_insee"].str[:2].isin(GRAND_EST)]
    for t in tables:
        d = d.merge(t, on="code_commune_insee", how="left")
    d["n_annees_reconnues"] = d["n_annees_reconnues"].fillna(0).astype(int)
    d["dep"] = d["code_commune_insee"].str[:2]
    for p in ["part_moyen", "part_fort"]:
        d[p] = d[p].fillna(d.groupby("dep")[p].transform("mean"))
    d["part_mf"] = d["part_moyen"] + d["part_fort"]
    d["log_maisons"] = np.log(d["n_maisons"])
    d["part_logements_maisons"] = (d["logements_maisons"]
                                   / d["logements"].where(d["logements"] > 0)).clip(upper=1)
    r = d["n_annees_reconnues"].where(d["n_annees_reconnues"] > 0)
    part = d["part_logements_maisons"]
    d["tau_obs"] = d["freq_obs"] * N_ANNEES / r
    d["tau_maison_obs"] = (d["tau_obs"] / part).clip(upper=1)
    d["tau_maison_centre"] = (d["freq_centre"] * N_ANNEES / r / part).clip(upper=1)
    # Communes servant à l'ajustement : taux observé, dénominateur corrigeable.
    d["ajuste"] = d["tau_maison_obs"].notna() & (part >= part_maisons_min)
    d["poids"] = (d["representativite"].map(REPRESENTATIVITE) * d["n_maisons"]
                  * d["n_annees_reconnues"])
    return d.reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Modèle                                                                       #
# --------------------------------------------------------------------------- #

def logit_fractionnaire(d: pd.DataFrame, formule: str):
    """Taux dans [0, 1] : quasi-binomial pondéré par les maisons-années reconnues."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return smf.glm(f"{CIBLE} ~ {formule}", d, family=sm.families.Binomial(),
                       var_weights=d["poids"]).fit(cov_type="HC0")


def deviance(y: np.ndarray, mu: np.ndarray) -> np.ndarray:
    return 2 * (xlogy(y, y / mu) + xlogy(1 - y, (1 - y) / (1 - mu)))


def valider(d: pd.DataFrame, specifications: dict[str, str], plis: int) -> pd.DataFrame:
    """Validation croisée par commune de chaque spécification.

    Part de la déviance pondérée expliquée hors échantillon, en référence à la
    moyenne pondérée du pli d'apprentissage (négative : moins bien que cette
    moyenne), et corrélation de rang entre taux observés et prédits.
    """
    x = d[d["ajuste"]]
    y, w = x[CIBLE].to_numpy(), x["poids"].to_numpy()
    plis = KFold(plis, shuffle=True, random_state=0)
    lignes = {}
    for nom, formule in specifications.items():
        pred, ref = np.empty(len(x)), np.empty(len(x))
        for app, test in plis.split(x):
            a = x.iloc[app]
            pred[test] = logit_fractionnaire(a, formule).predict(x.iloc[test])
            ref[test] = np.average(a[CIBLE], weights=a["poids"])
        lignes[nom] = {"deviance expliquee": 1 - (w * deviance(y, pred)).sum()
                       / (w * deviance(y, ref)).sum(),
                       "spearman": spearmanr(y, pred)[0]}
    return pd.DataFrame(lignes).T


def predire(d: pd.DataFrame, formule: str,
            tau_max: float = TAU_MAX) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Prédiction pour toutes les communes modélisables, et coefficients."""
    m = d[d["n_maisons"].notna()]
    res = logit_fractionnaire(m[m["ajuste"]], formule)
    d = d.copy()
    d.loc[m.index, "tau_maison_pred"] = res.predict(m)
    d["tau_retenu"] = d[CIBLE].where(d["ajuste"], d["tau_maison_pred"]).clip(upper=tau_max)
    d["source"] = np.select([d["ajuste"], d["tau_maison_pred"].notna()],
                            ["ONRN", "modele"], "aucune")
    return d, tableau(res, "tau_maison")


# --------------------------------------------------------------------------- #
# Rapport                                                                      #
# --------------------------------------------------------------------------- #

def _q(s: pd.Series, fmt: str = "{:.2%}") -> str:
    s = s.dropna()
    return f"n {len(s):,} ; " + ", ".join(
        f"q{int(k * 100)} {fmt.format(v)}" for k, v in s.quantile([.1, .5, .9]).items())


def decrire(d: pd.DataFrame) -> None:
    reco = d["n_annees_reconnues"] > 0
    classe = d["freq_obs"].notna()
    print(f"\ncommunes Grand Est (ONRN) : {len(d):,}, dont {d['n_maisons'].notna().sum():,} "
          f"avec des maisons BDNB")
    print(f"  classe renseignée {classe.sum():,} ; reconnues {DEBUT}-{FIN} {reco.sum():,} ; "
          f"les deux {(classe & reco).sum():,}")
    ecartees = reco & ~classe
    ma = d["n_maisons"] * d["n_annees_reconnues"]
    print(f"  reconnues sans sinistre répertorié, écartées : {ecartees.sum():,} communes, "
          f"{ma[ecartees].sum() / ma[reco].sum():.1%} des maisons-années reconnues")
    villes = d[CIBLE].notna() & ~d["ajuste"]
    print(f"  maisons minoritaires, taux prédit au lieu de l'observé : {villes.sum():,} communes, "
          f"{ma[villes].sum() / ma[reco].sum():.1%} des maisons-années reconnues")

    print("\nmédianes par classe ONRN")
    print(d.groupby("classe_onrn").agg(
        communes=("code_commune_insee", "size"), population=("population", "median"),
        maisons=("n_maisons", "median"), part_logements_maisons=("part_logements_maisons", "median"),
        R=("n_annees_reconnues", "median"), part_fort=("part_fort", "median"),
    ).reindex([*CLASSES, SANS_CLASSE]).to_string(float_format="{:,.2f}".format))

    t = d[d["ajuste"]].copy()
    t["maisons"] = pd.qcut(t["n_maisons"], 5)
    print("\ntaux par année reconnue selon la taille (quintiles du nombre de maisons)")
    print(t.groupby("maisons", observed=True).apply(lambda g: pd.Series({
        "communes": len(g), "R_moyen": g["n_annees_reconnues"].mean(),
        "part_logements_maisons": g["part_logements_maisons"].median(),
        "tau_risque_median": g["tau_obs"].median(), "tau_maison_median": g[CIBLE].median(),
        "tau_maison_pondere": np.average(g[CIBLE], weights=g["poids"]),
    }), include_groups=False).to_string(float_format="{:.3f}".format))
    moy = {nom: np.average(t[c], weights=t["poids"])
           for nom, c in [("borne haute", CIBLE), ("centre de classe", "tau_maison_centre")]}
    print("\ntau par maison et année reconnue, moyenne pondérée : "
          + ", ".join(f"{k} {v:.2%}" for k, v in moy.items())
          + f" (cout_argile_attendu.py : {TAUX_DEFAUT:.2%})")


def rapporter(d: pd.DataFrame, cv: pd.DataFrame, coefs: pd.DataFrame, formule: str) -> None:
    print(f"\nvalidation croisée ({PLIS} plis, par commune, pondérée)")
    print(cv.to_string(float_format="{:.3f}".format))
    print(f"\nformule retenue : {formule}")
    print("exp(coef) [IC95], * = p < 0,05")
    for _, c in coefs[coefs["terme"] != "Intercept"].iterrows():
        print(f"  {c['terme']:<14} {c['exp_coef']:5.2f} [{c['ic95_bas']:4.2f}-{c['ic95_haut']:5.2f}]"
              f"{'*' if c['p_valeur'] < .05 else ''}")
    print("\ntaux retenu par source (par maison et année reconnue)")
    for source, g in d[d["source"].ne("aucune")].groupby("source"):
        print(f"  [{source}] {_q(g['tau_retenu'])}")


# --------------------------------------------------------------------------- #

def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--onrn", type=Path, default=ONRN)
    p.add_argument("--catnat", type=Path, default=CATNAT_DEFAUT)
    p.add_argument("--maisons", type=Path, default=MAISONS)
    p.add_argument("--communes", type=Path, default=COMMUNES,
                   help="référentiel des communes portant la population")
    p.add_argument("--borne-ouverte", type=float, default=BORNE_OUVERTE,
                   help=f"borne haute de la classe « > 10 ‰ », en ‰ (défaut {BORNE_OUVERTE:g})")
    p.add_argument("--part-maisons-min", type=float, default=PART_MAISONS_MIN,
                   help="part minimale des logements en maison pour ajuster le modèle "
                        f"(défaut {PART_MAISONS_MIN:g})")
    p.add_argument("--tau-max", type=float, default=TAU_MAX,
                   help=f"plafond du taux retenu (défaut {TAU_MAX:g})")
    p.add_argument("--formule", default=FORMULE,
                   help=f"covariables du modèle (défaut « {FORMULE} »)")
    p.add_argument("--sortie", type=Path, default=SORTIE)
    a = p.parse_args()
    if not a.maisons.exists():
        raise SystemExit(f"{a.maisons} introuvable : lancer d'abord extract_maisons.py")

    d = assembler(lire_onrn(a.onrn, a.borne_ouverte), reconnaissances(a.catnat),
                  exposition(a.maisons), logements(GRAND_EST), population(a.communes),
                  part_maisons_min=a.part_maisons_min)
    decrire(d)
    specifications = dict(SPECIFICATIONS)
    if a.formule not in specifications.values():
        specifications["formule retenue"] = a.formule
    cv = valider(d[d["n_maisons"].notna()], specifications, PLIS)
    d, coefs = predire(d, a.formule, a.tau_max)
    print(f"\ntaux retenus plafonnés à {a.tau_max:.0%} : {d[CIBLE].where(d['ajuste']).gt(a.tau_max).sum():,}")
    rapporter(d, cv, coefs, a.formule)

    colonnes = ["code_commune_insee", "libelle_commune", "dep", "classe_onrn",
                "representativite", "n_annees_reconnues", "population", "n_maisons",
                "part_logements_maisons", "part_moyen", "part_fort", "part_alea_inconnu",
                "freq_obs", "tau_obs", "tau_maison_obs", "tau_maison_pred", "tau_retenu",
                "source"]
    a.sortie.parent.mkdir(parents=True, exist_ok=True)
    d[colonnes].to_csv(a.sortie, index=False)
    sortie_coefs = a.sortie.with_name(a.sortie.stem + "_coefs.csv")
    coefs.to_csv(sortie_coefs, index=False)
    print(f"\ncommunes -> {a.sortie}\ncoefficients -> {sortie_coefs}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
