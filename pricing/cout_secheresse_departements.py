#!/usr/bin/env python3
"""Coût des sinistres sécheresse d'une année, par département du Grand Est.

Assemble les maillons déjà calculés dans `pricing/` :

    éligibilité de la commune c l'année a       SWI, section 6.1 de l'article
    maisons sinistrées          tau_c × N_c     taux_destruction_onrn.py
    coût d'une maison m         t_m × p_d × S_m glm_taux_dommage.py, prix_m2_departements.py

    C_c     = tau_c × Σ_m t_m × p_d × S_m       coût de la commune si elle est reconnue
    coût(a) = Σ_c éligible(c, a) × C_c

tau_c  part des maisons sinistrées une année de reconnaissance (tau_retenu) ;
t_m    taux de dommage du GLM sécheresse, E[CH_TOT / V], selon le nombre de
       niveaux, la période de construction, l'aléa argile et le département ;
p_d    prix médian au m² des maisons du département (DVF ; immobilier.notaires.fr
       pour 57, 67 et 68, absents de DVF) ;
S_m    surface de plancher estimée (emprise × niveaux), celle du GLM. Pour une
       maison écartée du GLM (géométrie fictive, caractéristiques non plausibles),
       la médiane des autres maisons de la commune.
Toutes les maisons de la commune comptent : tau est rapporté à toutes ses
maisons, l'aléa argile entre par t_m et par tau.

ÉLIGIBILITÉ (section 6.1)
-------------------------
Le critère de la brochure Météo-France, celui de eligibilite_catnat.py :
minimum annuel du SWI uniforme mensuel (mois le plus sec), rang dans une
fenêtre glissante de 30 ans. Seuil 1 : rang 1-3 (période de retour >= 10 ans) ;
seuil 2 : rang 1-6 (>= 5 ans) l'année étudiée et dans 2 des 4 années
précédentes. C'est le critère de la circulaire du 29 avril 2024, appliqué aux
sécheresses de 2024 et après ; les années antérieures relevaient d'autres règles.

Une commune est éligible dès qu'une maille couvrant une partie de son
territoire l'est. Les mailles SAFRAN sont des carrés de 8 km centrés sur leurs
(x, y) en Lambert II étendu, intersectés avec les contours communaux 2026 de
prix_m2_communes.py. Le seuil 3 (commune à >= 5 ans, limitrophe d'une commune
éligible) est compté à part, comme les communes « potentiellement éligibles »
de l'article, et n'entre que dans la variante « avec seuil 3 ».

SWI : `swi_u` des parquets de SwiGAN/Outputs/rcp_8.5/Grand Est (data_train
1960-2021, data_val 2022-2023, data_test 2024-2025), le SWI observé : la
fenêtre de 2024 est 1995-2024.

TAUX DE DOMMAGE ET PRIX AU M²
-----------------------------
Le GLM est ajusté sur V = 1 800 €/m² × surface de plancher (valeur de
reconstruction, mode par défaut de glm_taux_dommage.py). Lui appliquer p_d au
lieu de 1 800 €/m² multiplie la charge qu'il prédit par p_d / 1 800 : 0,52 en
Haute-Marne, 1,21 dans le Bas-Rhin. La variante « reconstruction » garde
1 800 €/m² et rend la charge du GLM telle quelle. Le prix DVF est rapporté à la
surface bâtie DVF, plus petite que la surface de plancher géométrique (rapport
médian 0,59 sur les maisons vendues) : p_d × S_m n'est pas la valeur vénale de
la maison, c'est l'assiette du taux t_m, au prix du département.

L'effet année du GLM fixe le niveau des coûts. Il est pris pour toutes les
années à la modalité de --annee-dommage (2024 par défaut, fusionnée dans
« autre » faute de sinistres : exp(β) = 1,18, IC95 0,21-6,7). Les effets
annuels estimés vont de 0,14 à 5,5 : c'est la première source d'incertitude du
niveau. Le Haut-Rhin, trop peu sinistré, prend l'effet de la Moselle.

COMPARAISON ONRN 1995-2021
--------------------------
ONRN_CoutCumu : coût cumulé des sinistres sécheresse 1995-2021 par commune, en
classes, euros 2021, tout le marché (biens assurés hors véhicules, extrapolé).
ONRN_CoutMoyen : coût moyen d'un sinistre sur la même période, en classes.
Dans ce second classeur la colonne des codes INSEE est triée par ordre
croissant, les lignes (commune, classe) par code décroissant : retourner la
colonne réaligne les deux classeurs (noms identiques sur toutes les communes),
et le script le vérifie.

Contreparties du modèle, mêmes communes, même période :
  modèle × reconnaissances   Σ_c C_c × R_c, R_c années 1995-2021 couvertes par
                             un arrêté sécheresse (catnat.csv)
  modèle × éligibilité SWI   Σ_c C_c × E_c, E_c années 1995-2021 éligibles au
                             critère ci-dessus
  coût par sinistre          C_c / (tau_c × N_c), confronté à la classe ONRN du
                             coût moyen de la commune
tau_c venant de la borne haute de la classe de fréquence ONRN, le nombre de
sinistres du modèle est un majorant.

INTENSITÉ (--intensite, --severite)
-----------------------------------
Sans option, C_c ne dépend pas de l'année. --intensite VARIANTE ... module la
fréquence par le multiplicateur g de frequence_secheresse_swi.py (une colonne
par variante, par exemple deficit_iid et deficit_bym2) :
    tau_ct = min(tau_c × g_ct / ḡ_c, tau_max),  ḡ_c = moyenne de g sur les années
             1995-2021 où la commune a été reconnue (moyenne du département pour
             une commune jamais reconnue)
de sorte que modèle × reconnaissances 1995-2021 reste celui de l'ONRN, au
plafond près. Pour une variante ajustée avec --seuil (deficit_iid_seuil), g
comprend le saut γ à l'éligibilité SWI de la commune-année : les années
tarifées le sont toutes, celles de ḡ_c pas toujours, et g / ḡ_c relève alors
le niveau ONRN des communes reconnues hors critère SWI.
--severite VARIANTE ... module en plus le coût par sinistre par
h_ct / h̄ (severite_secheresse_swi.py), h̄ étant la moyenne de h sur les
sinistres du Grand Est qui ont servi à l'estimation ; chaque variante de
fréquence prend le h de son propre indice. Chaque tirage a
posteriori donne un coût : le script imprime la moyenne et l'intervalle à
90 %. --plafond-s ramène l'intensité au maximum observé à l'estimation.

Prérequis : extract_maisons.py, prix_m2_departements.py, prix_m2_communes.py
(contours en cache), taux_destruction_onrn.py, glm_taux_dommage.py (table des
modalités) ; avec --intensite, intensite_swi.py et frequence_secheresse_swi.py,
avec --severite, severite_secheresse_swi.py. Aucune donnée de sinistre n'est
lue ici.
Usage : python3 cout_secheresse_departements.py [--annees 2024 ...]
                                                [--annee-dommage 2024]
                                                [--intensite deficit_iid deficit_bym2]
                                                [--severite deficit sswi] [--plafond-s]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import duckdb
import geopandas as gpd
import numpy as np
import pandas as pd
import shapely

from eligibilite_catnat import (FENETRE_DEFAUT, RANG_5_ANS, RANG_10_ANS, _swi_numerique,
                                appliquer_seuils, indicateur_annuel, rangs_glissants)
from extract_maisons import GRAND_EST, USAGE_MAISON
from extraire_catnat import CATNAT_DEFAUT
from glm_taux_dommage import COUT_M2, PERIODES
from prix_m2_communes import PRIX_ALSACE_MOSELLE, referentiel, voisinages
from taux_destruction_onrn import DEBUT, FIN, annees_reconnues, lire_feuille

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "output"
SWI = ROOT.parent.parent / "SwiGAN" / "Outputs" / "rcp_8.5" / "Grand Est"
FICHIERS_SWI = ["data_train", "data_val", "data_test"]
MAISONS = OUT / "maisons_grand_est.csv"
TAUX = OUT / "taux_destruction_communes.csv"
PRIX = OUT / "prix_m2_departements.csv"
COEFS = OUT / "glm_taux_dommage_reconstruction_coefs.csv"
MODALITES = OUT / "glm_taux_dommage_reconstruction_modalites.csv"
ONRN_CUMUL = ROOT / "ONRN_CoutCumu_SECH_9521.xlsx"
ONRN_MOYEN = ROOT / "ONRN_CoutMoyen_SECH_9521.xlsx"
SORTIE = OUT / "cout_secheresse_{}.csv"

ANNEE = 2024
TIRAGES_COUT = 500
MODELE = "sécheresse"
LAMBERT_2_ETENDU = "EPSG:27572"
COTE_MAILLE = 8000
# Bornes hautes des périodes de construction du GLM (pd.cut, intervalles fermés à droite).
BORNES_PERIODES = [1918, 1945, 1974, 1999]
CODE = "code_commune_insee"
DEPARTEMENTS = {"08": "Ardennes", "10": "Aube", "51": "Marne", "52": "Haute-Marne",
                "54": "Meurthe-et-Moselle", "55": "Meuse", "57": "Moselle",
                "67": "Bas-Rhin", "68": "Haut-Rhin", "88": "Vosges"}


# --------------------------------------------------------------------------- #
# Éligibilité : SWI -> mailles -> communes                                     #
# --------------------------------------------------------------------------- #

def charger_swi(dossier: Path, fichiers: list[str] = FICHIERS_SWI
                ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """SWI mensuel par (maille, année, mois), et les mailles avec leur centre (x, y)."""
    df = pd.concat([pd.read_parquet(dossier / f"{f}.parquet",
                                    columns=["year", "month", "x", "y", "swi_u"])
                    for f in fichiers], ignore_index=True)
    # Un mois en double donnerait 24 valeurs à l'année, qu'indicateur_annuel écarterait.
    if df.duplicated(["year", "month", "x", "y"]).any():
        raise SystemExit(f"{dossier} : des mois figurent dans plusieurs fichiers")
    mailles = df[["x", "y"]].drop_duplicates().sort_values(["x", "y"], ignore_index=True)
    mailles["maille"] = mailles.index
    df = df.merge(mailles, on=["x", "y"]).rename(columns={"year": "annee", "month": "mois"})
    df["swi"] = _swi_numerique(df["swi_u"])
    return df[["maille", "annee", "mois", "swi"]], mailles


def mailles_anormales(swi: pd.DataFrame, annees: list[int]) -> pd.DataFrame:
    """Rang sur 30 ans, seuils 1 et 2, et période de retour >= 5 ans (rp5), par (maille, année)."""
    cube = indicateur_annuel(swi, ["maille"]).pivot(index="maille", columns="annee",
                                                    values="swi_min")
    rangs = rangs_glissants(cube.to_numpy(np.float32), FENETRE_DEFAUT)
    seuils = appliquer_seuils(rangs, RANG_10_ANS, RANG_5_ANS)
    seuils["rp5"] = ~np.isnan(rangs) & (np.nan_to_num(rangs, nan=np.inf) <= RANG_5_ANS)
    sans_rang = [a for a in annees if a not in cube.columns
                 or np.isnan(rangs[:, cube.columns.get_loc(a)]).all()]
    if sans_rang:
        raise SystemExit(f"pas de période de retour pour {sans_rang} : SWI "
                         f"{cube.columns.min()}-{cube.columns.max()}, fenêtre de "
                         f"{FENETRE_DEFAUT} ans")
    j = [cube.columns.get_loc(a) for a in annees]
    return pd.DataFrame({"maille": np.repeat(cube.index.to_numpy(), len(j)),
                         "annee": np.tile(annees, len(cube)),
                         "rang": rangs[:, j].ravel(),
                         **{k: seuils[k][:, j].ravel() for k in ["seuil1", "seuil2", "rp5"]}})


def correspondance(communes: gpd.GeoDataFrame, mailles: pd.DataFrame) -> pd.DataFrame:
    """Couples (commune, maille) dont l'intersection a une surface non nulle (m², `surface`)."""
    demi = COTE_MAILLE / 2
    x, y = mailles["x"].to_numpy(), mailles["y"].to_numpy()
    carres = gpd.GeoDataFrame(mailles[["maille"]], crs=LAMBERT_2_ETENDU,
                              geometry=shapely.box(x - demi, y - demi, x + demi, y + demi))
    c = communes[[CODE, "geometry"]].to_crs(LAMBERT_2_ETENDU)
    inter = gpd.overlay(c, carres, how="intersection", keep_geom_type=True)
    inter["surface"] = inter.area
    return pd.DataFrame(inter.loc[inter["surface"] > 0, [CODE, "maille", "surface"]])


def rang_communes(anormales: pd.DataFrame, corr: pd.DataFrame) -> pd.DataFrame:
    """Par (commune, année) : rang de sa maille la plus sèche, celle qui la rend éligible."""
    return (corr.merge(anormales[["maille", "annee", "rang"]], on="maille")
            .groupby([CODE, "annee"], as_index=False)["rang"].min())


def eligibilite_communes(anormales: pd.DataFrame, corr: pd.DataFrame,
                         communes: gpd.GeoDataFrame, adjacence) -> pd.DataFrame:
    """Par (commune, année) : éligible dès qu'une de ses mailles l'est (seuil 1 ou 2).

    seuil3 : commune non éligible, à période de retour >= 5 ans sur une de ses
    mailles, limitrophe d'une commune éligible. `adjacence` suit l'ordre des
    lignes de `communes`.
    """
    annees = sorted(anormales["annee"].unique())
    idx = pd.MultiIndex.from_product([communes[CODE], annees], names=[CODE, "annee"])
    d = (corr.merge(anormales, on="maille")
         .groupby([CODE, "annee"])[["seuil1", "seuil2", "rp5"]].any()
         .reindex(idx, fill_value=False).reset_index())
    d["eligible"] = d["seuil1"] | d["seuil2"]
    forme = (len(communes), len(annees))
    e = d["eligible"].to_numpy().reshape(forme)
    voisine = (adjacence @ e.astype(float)) > 0
    d["seuil3"] = (d["rp5"].to_numpy().reshape(forme) & ~e & voisine).ravel()
    return d


# --------------------------------------------------------------------------- #
# Coût d'une commune reconnue                                                  #
# --------------------------------------------------------------------------- #

def effets_glm(coefs: Path, modalites: Path) -> tuple[float, dict[str, dict[str, float]]]:
    """Constante et exp(coef) par variable et modalité BRUTE (fusions résolues, référence à 1)."""
    c = pd.read_csv(coefs)
    c = c[c["modele"] == MODELE].set_index("terme")["exp_coef"]
    m = pd.read_csv(modalites, dtype=str)
    effets: dict[str, dict[str, float]] = {}
    for v, brute, modele, ref in m.loc[m["modele"] == MODELE, ["variable", "modalite",
                                       "modalite_modele", "reference"]].itertuples(index=False):
        terme = f"{v} = {modele}"
        if ref != "True" and terme not in c.index:
            raise SystemExit(f"{coefs.name} : terme « {terme} » absent ; relancer "
                             "glm_taux_dommage.py")
        effets.setdefault(v, {})[brute] = 1.0 if ref == "True" else float(c[terme])
    return float(c["Intercept"]), effets


def maisons_par_commune(chemin: Path, effets: dict[str, dict[str, float]]) -> pd.DataFrame:
    """Par commune : maisons, surface de plancher, et surface pondérée par t_m hors constante."""
    periode = " ".join(f"WHEN annee <= {b} THEN '{p}'" for b, p in zip(BORNES_PERIODES, PERIODES))
    g = duckdb.sql(f"""
        WITH brut AS (
            SELECT code_commune_insee, code_departement_insee AS dep,
                   try_cast(nb_niveau AS DOUBLE)                     AS nb_niveau,
                   try_cast(annee_construction AS DOUBLE)            AS annee,
                   coalesce(alea_argile, 'Inconnu')                  AS alea,
                   try_cast(surface_plancher_estimee_m2 AS DOUBLE)   AS surface,
                   plausible = 'true' AND geometrie_fictive = 'false' AS valide
            FROM read_csv('{chemin.as_posix()}', all_varchar = true)
            WHERE usage_ffo = '{USAGE_MAISON}'
        ), m AS (
            SELECT *,
                   CASE WHEN nb_niveau >= 3 THEN '3+'
                        ELSE nb_niveau::INTEGER::VARCHAR END                  AS niveaux,
                   CASE WHEN annee IS NULL THEN 'nan' {periode}
                        ELSE '{PERIODES[-1]}' END                             AS periode,
                   median(CASE WHEN valide THEN surface END)
                       OVER (PARTITION BY code_commune_insee)                 AS med_commune,
                   median(CASE WHEN valide THEN surface END)
                       OVER (PARTITION BY dep)                                AS med_dep
            FROM brut
        )
        SELECT code_commune_insee, dep, niveaux, periode, alea, count(*) AS n,
               sum(CASE WHEN valide THEN surface
                        ELSE coalesce(med_commune, med_dep) END)              AS surface
        FROM m GROUP BY ALL
    """).df()
    facteur = pd.Series(1.0, index=g.index)
    for v in ["niveaux", "periode", "alea"]:
        e = g[v].map(effets.get(v, {}))
        if e.isna().any():
            print(f"  {v} : {g.loc[e.isna(), 'n'].sum():,} maisons de modalité inconnue du GLM "
                  f"({', '.join(map(str, g.loc[e.isna(), v].unique()))}), effet 1")
        facteur *= e.fillna(1.0)
    g["surface_ponderee"] = g["surface"] * facteur
    return (g.groupby([CODE, "dep"], as_index=False)[["n", "surface", "surface_ponderee"]].sum()
            .rename(columns={"n": "n_maisons"}))


def prix_departements(chemin: Path) -> pd.Series:
    p = pd.read_csv(chemin, dtype={"dep": str})
    p = p[p["type_local"] == "Maison"].set_index("dep")["prix_m2_median"]
    return pd.concat([p, pd.Series(PRIX_ALSACE_MOSELLE, dtype=float)]).reindex(GRAND_EST)


# --------------------------------------------------------------------------- #
# ONRN                                                                         #
# --------------------------------------------------------------------------- #

def bornes_classe(libelle: str) -> tuple[float, float]:
    """Bornes en euros d'une classe ONRN : (0, 0) sans sinistre, haute infinie si ouverte."""
    if libelle.startswith("Pas de sinistre"):
        return 0.0, 0.0
    valeurs, unite = [], None
    # « Entre 0 et 2,5 k€ » : l'unité d'un nombre est celle du suivant à défaut de la sienne.
    for nombre, u in reversed(re.findall(r"(\d+(?:,\d+)?)\s*(k|M)?", libelle)):
        unite = u or unite
        valeurs.insert(0, float(nombre.replace(",", ".")) * {"k": 1e3, "M": 1e6}[unite])
    return (valeurs[0], valeurs[1]) if len(valeurs) == 2 else (valeurs[0], np.inf)


def lire_onrn_couts(cumul: Path, moyen: Path) -> pd.DataFrame:
    """Par commune du Grand Est : classes ONRN du coût cumulé et du coût moyen, bornes en euros."""
    noms = [CODE, "libelle_onrn"]
    c = lire_feuille(cumul, "Coûts commune").iloc[:, :3].set_axis(
        [*noms, "classe_cout_cumule"], axis=1)
    m = lire_feuille(moyen, "Coût moy. sinistres").iloc[:, :3].set_axis(
        [*noms, "classe_cout_moyen"], axis=1)

    def accord(m: pd.DataFrame) -> float:
        x = c.merge(m, on=CODE, suffixes=("", "_m"))
        return float((x["libelle_onrn"] == x["libelle_onrn_m"]).mean())

    if accord(m) < 0.99:
        m[CODE] = m[CODE].to_numpy()[::-1]
        if accord(m) < 0.99:
            raise SystemExit(f"{moyen.name} : codes et communes désalignés, réalignement impossible")
        print(f"{moyen.name} : colonne des codes retournée, noms concordants à {accord(m):.1%}")
    d = c.merge(m.drop(columns="libelle_onrn"), on=CODE)
    d = d[d[CODE].str[:2].isin(GRAND_EST)].reset_index(drop=True)
    d["dep"] = d[CODE].str[:2]
    for col, pre in [("classe_cout_cumule", "cumul"), ("classe_cout_moyen", "moyen")]:
        d[[f"{pre}_bas", f"{pre}_haut"]] = d[col].map(bornes_classe).tolist()
    return d


# --------------------------------------------------------------------------- #
# Intensité : tau_ct = tau_c × g / ḡ_c, coût par sinistre × h / h̄              #
# --------------------------------------------------------------------------- #

def lire_tirages(gabarit: Path, variante: str, n: int, graine: int) -> tuple[pd.DataFrame, dict]:
    chemin = Path(str(gabarit).format(variante=variante, quoi="tirages.csv"))
    if not chemin.exists():
        raise SystemExit(f"{chemin} introuvable : lancer d'abord le modèle « {variante} »")
    meta = json.loads(Path(str(gabarit).format(variante=variante, quoi="meta.json"))
                      .read_text(encoding="utf-8"))
    t = pd.read_csv(chemin)
    return t.sample(n=min(n, len(t)), random_state=graine).reset_index(drop=True), meta


def modulations(d: pd.DataFrame, elig: pd.DataFrame, reco: pd.DataFrame, annees_cout: list[int],
                frequences: list[str], severite: list[str], plafond: bool,
                n_tirages: int) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, np.ndarray]]:
    """Coûts totaux par tirage a posteriori, pour chaque variante, et coûts moyens par commune.

    Renvoie {variante: {libellé: coûts par tirage}} et {colonne: coût moyen par commune}.
    """
    # Imports locaux : ces modules importent eux-mêmes celui-ci (CODE, charger_swi).
    from frequence_secheresse_swi import SORTIE as GABARIT_G, exposition_france, multiplicateur
    from intensite_swi import charger_intensite
    from severite_secheresse_swi import SORTIE as GABARIT_H
    from taux_destruction_onrn import TAU_MAX

    annees = sorted(set(range(DEBUT, FIN + 1)) | set(annees_cout))
    historiques = list(range(DEBUT, FIN + 1))
    codes = d[CODE].to_numpy()
    argile = exposition_france().set_index(CODE)["argile"]
    argile = d[CODE].map(argile).fillna(argile[argile.index.str[:2].isin(GRAND_EST)].median())
    argile = argile.to_numpy()
    intensite = charger_intensite()
    intensite = intensite[intensite[CODE].isin(set(codes)) & intensite["annee"].isin(annees)]

    def cube(col: str) -> np.ndarray:
        return (intensite.pivot(index=CODE, columns="annee", values=col)
                .reindex(index=codes, columns=annees).to_numpy(dtype=float))

    def presence(t: pd.DataFrame, filtre: pd.Series | None = None) -> np.ndarray:
        t = t if filtre is None else t[filtre]
        t = t[t["annee"].between(DEBUT, FIN)].assign(v=1.0)
        return (t.drop_duplicates([CODE, "annee"]).pivot(index=CODE, columns="annee", values="v")
                .reindex(index=codes, columns=historiques).fillna(0).to_numpy() > 0)

    reconnue = presence(reco)
    eligible = presence(elig, elig["eligible"])
    # Éligibilité SWI de chaque commune-année : le saut γ des modèles ajustés avec --seuil.
    e_cube = (elig.pivot(index=CODE, columns="annee", values="eligible")
              .reindex(index=codes, columns=annees).fillna(False).to_numpy(dtype=bool))
    cout = d["cout_si_reconnue"].to_numpy()
    tau = d["tau_retenu"].to_numpy()
    tau_ok = np.nan_to_num(tau) > 0
    dep = d["dep"].to_numpy()
    cubes: dict[str, np.ndarray] = {}

    def facteur_brut(tir: pd.DataFrame, meta: dict, k: int) -> np.ndarray:
        """g (ou h) de l'année d'indice k, une ligne par tirage ; 1 sans intensité connue."""
        if meta["intensite"] not in cubes:
            cubes[meta["intensite"]] = cube(meta["intensite"])
        s = cubes[meta["intensite"]][:, k]
        f = np.ones((len(tir), len(codes)))
        ok = ~np.isnan(s)
        f[:, ok] = multiplicateur(tir, meta, s[ok], argile[ok], plafond, eligible=e_cube[ok, k])
        return f

    def facteur(tir: pd.DataFrame, meta: dict, gbar: np.ndarray, sev: tuple | None,
                k: int) -> np.ndarray:
        """Coût de l'année k rapporté à C_c : tau_ct / tau_c (plafonné), × h / h̄ si sev."""
        g = facteur_brut(tir, meta, k) / gbar
        f = np.where(tau_ok, np.minimum(np.nan_to_num(tau) * g, TAU_MAX)
                     / np.where(tau_ok, tau, 1.0), 1.0)
        if sev:
            tir_h, meta_h, hbar = sev
            f = f * facteur_brut(tir_h, meta_h, k) / hbar
        return f

    # h de chaque indice : une variante de fréquence prend le h de son propre indice.
    severites = {}
    for v in severite:
        tir_h, meta_h = lire_tirages(GABARIT_H, v, n_tirages, 1)
        hbar = tir_h["hbar"].to_numpy()[:, None]
        retenu = "retenu" if meta_h.get("retenu") else "NON retenu (intervalle à 90 % contenant 0)"
        print(f"\nsévérité {v} : h {retenu}, P(βh > 0) = {meta_h['p_positif']:.2f}")
        severites[meta_h["intensite"]] = (tir_h, meta_h, hbar)

    totaux, par_commune = {}, {}
    for variante in frequences:
        tir, meta = lire_tirages(GABARIT_G, variante, n_tirages, 0)
        sev = severites.get(meta["intensite"])
        if severites and sev is None:
            raise SystemExit(f"--severite : aucune variante pour l'indice {meta['intensite']} "
                             f"de {variante}")
        if sev and len(tir) != len(sev[0]):
            raise SystemExit("tirages de fréquence et de sévérité en nombres différents")
        print(f"fréquence {variante} : {meta['communes_annees']:,} communes-années "
              f"{meta['annees'][0]}-{meta['annees'][1]}, R-hat max {meta['r_hat_max']:.3f}, "
              f"{meta['divergences']} divergences")
        # ḡ_c sur les années reconnues 1995-2021 ; à défaut, moyenne du département.
        somme = np.zeros((len(tir), len(codes)))
        for j, an in enumerate(historiques):
            somme += facteur_brut(tir, meta, annees.index(an)) * reconnue[:, j]
        n_reco = reconnue.sum(axis=1)
        gbar = somme / np.maximum(n_reco, 1)
        for x in np.unique(dep):
            dans = dep == x
            if (n_reco[dans] == 0).any() and n_reco[dans].sum() > 0:
                gbar[:, dans & (n_reco == 0)] = (somme[:, dans].sum(axis=1)
                                                 / n_reco[dans].sum())[:, None]
        gbar[:, n_reco == 0] = np.where(gbar[:, n_reco == 0] > 0, gbar[:, n_reco == 0], 1.0)

        for avec_h in [False, True] if sev else [False]:
            nom = f"{variante} × h" if avec_h else variante
            res = {}
            for an in annees_cout:
                e = d[f"eligible_{an}"].to_numpy()
                f = facteur(tir, meta, gbar, sev if avec_h else None, annees.index(an))
                res[f"{an} (éligibilité SWI)"] = (f * (cout * e)).sum(axis=1)
                par_commune[f"cout_{an}_{nom.replace(' × ', '_x_')}"] = (f * (cout * e)).mean(axis=0)
            hist_r = np.zeros(len(tir))
            hist_e = np.zeros(len(tir))
            for j, an in enumerate(historiques):
                f = facteur(tir, meta, gbar, sev if avec_h else None, annees.index(an))
                hist_r += (f * (cout * reconnue[:, j])).sum(axis=1)
                hist_e += (f * (cout * eligible[:, j])).sum(axis=1)
            res[f"{DEBUT}-{FIN} × arrêtés"] = hist_r
            res[f"{DEBUT}-{FIN} × éligibilité SWI"] = hist_e
            totaux[nom] = res
    sans = {f"{an} (éligibilité SWI)": np.array([(cout * d[f"eligible_{an}"]).sum()])
            for an in annees_cout}
    sans[f"{DEBUT}-{FIN} × arrêtés"] = np.array([(cout[:, None] * reconnue).sum()])
    sans[f"{DEBUT}-{FIN} × éligibilité SWI"] = np.array([(cout[:, None] * eligible).sum()])
    return {"sans intensité": sans, **totaux}, par_commune


def rapporter_intensite(totaux: dict[str, dict[str, np.ndarray]]) -> pd.DataFrame:
    print("\n=== Coût total Grand Est selon l'intensité (M€, moyenne a posteriori [5 % - 95 %]) ===")
    lignes = []
    for variante, res in totaux.items():
        for libelle, v in res.items():
            lignes.append({"variante": variante, "periode": libelle, "moyenne": v.mean(),
                           "q05": np.quantile(v, .05), "q95": np.quantile(v, .95)})
    t = pd.DataFrame(lignes)
    large = t.assign(cellule=[f"{_m(m)}" if q05 == q95 else f"{_m(m)} [{_m(q05)} - {_m(q95)}]"
                              for m, q05, q95 in zip(t["moyenne"], t["q05"], t["q95"])])
    large = large.pivot(index="periode", columns="variante", values="cellule")
    print(large.reindex(index=list(next(iter(totaux.values()))), columns=list(totaux)).to_string())
    return t


# --------------------------------------------------------------------------- #
# Synthèse                                                                     #
# --------------------------------------------------------------------------- #

def par_departement(d: pd.DataFrame, onrn: pd.DataFrame, annees: list[int]) -> pd.DataFrame:
    g = d.groupby("dep")

    def somme(s: pd.Series) -> pd.Series:
        return s.groupby(d["dep"]).sum()

    t = pd.DataFrame({"departement": pd.Series(DEPARTEMENTS), "communes": g.size(),
                      "maisons": g["n_maisons"].sum(), "prix_m2": g["prix_m2"].first()})
    for an in annees:
        t[f"communes_eligibles_{an}"] = g[f"eligible_{an}"].sum()
        t[f"communes_seuil3_{an}"] = g[f"seuil3_{an}"].sum()
        t[f"communes_reconnues_{an}"] = g[f"reconnue_{an}"].sum()
        t[f"cout_{an}"] = g[f"cout_{an}"].sum()
        t[f"cout_{an}_reconstruction"] = somme(d[f"cout_{an}"] * COUT_M2 / d["prix_m2"])
        t[f"cout_{an}_avec_seuil3"] = somme(
            d["cout_si_reconnue"] * (d[f"eligible_{an}"] | d[f"seuil3_{an}"]))
    t["cout_si_toutes_reconnues"] = g["cout_si_reconnue"].sum()
    t["communes_annees_reconnues_9521"] = g["n_annees_reconnues"].sum()
    t["communes_annees_eligibles_swi_9521"] = g["n_annees_eligibles_swi"].sum()
    t["modele_x_reconnaissances_9521"] = somme(d["cout_si_reconnue"] * d["n_annees_reconnues"])
    t["modele_x_eligibilite_swi_9521"] = somme(d["cout_si_reconnue"] * d["n_annees_eligibles_swi"])
    o = onrn.groupby("dep")
    t["onrn_cumul_9521_bas"] = o["cumul_bas"].sum()
    t["onrn_cumul_9521_haut"] = o["cumul_haut"].sum()
    # Coût par sinistre : moyenne pondérée par les sinistres attendus 1995-2021.
    sinistres = d["tau_retenu"] * d["n_maisons"] * d["n_annees_reconnues"]
    t["cout_par_sinistre_modele"] = t["modele_x_reconnaissances_9521"] / somme(sinistres)
    return t


def confronter_cout_moyen(d: pd.DataFrame, onrn: pd.DataFrame) -> pd.DataFrame:
    """Coût par sinistre du modèle, rangé dans les classes ONRN du coût moyen."""
    classes = (onrn.loc[onrn["moyen_haut"] > 0, ["classe_cout_moyen", "moyen_bas"]]
               .drop_duplicates().sort_values("moyen_bas"))
    ordre = classes["classe_cout_moyen"].tolist()
    x = d.merge(onrn[[CODE, "classe_cout_moyen"]], on=CODE)
    x = x[x["classe_cout_moyen"].isin(ordre) & x["cout_par_sinistre"].notna()]
    x["rang_onrn"] = x["classe_cout_moyen"].map({c: i for i, c in enumerate(ordre)})
    x["rang_modele"] = np.searchsorted(classes["moyen_bas"].to_numpy(),
                                       x["cout_par_sinistre"].to_numpy(), side="right") - 1
    haut = len(ordre) - 1
    g = x.groupby("dep")
    return pd.DataFrame({
        "communes": g.size(),
        "modele_mediane_k€": g["cout_par_sinistre"].median() / 1e3,
        "classe_mediane_onrn": g["rang_onrn"].median().round().astype(int).map(dict(enumerate(ordre))),
        "classe_mediane_modele": g["rang_modele"].median().round().astype(int).map(
            dict(enumerate(ordre))),
        f"onrn_{ordre[-1]}": g["rang_onrn"].apply(lambda r: (r == haut).mean()),
        f"modele_{ordre[-1]}": g["rang_modele"].apply(lambda r: (r == haut).mean()),
        "meme_classe": (x["rang_onrn"] == x["rang_modele"]).groupby(x["dep"]).mean(),
    })


def _m(v: float) -> str:
    return f"{v / 1e6:,.1f}"


def rapporter(t: pd.DataFrame, cm: pd.DataFrame, annees: list[int]) -> None:
    tot = t.select_dtypes("number").sum()
    for an in annees:
        print(f"\n=== {an} : coût estimé des sinistres sécheresse (M€) ===")
        u = pd.DataFrame({
            "département": t["departement"],
            "maisons": t["maisons"].map("{:,.0f}".format),
            "€/m²": t["prix_m2"].map("{:,.0f}".format),
            "éligibles": t[f"communes_eligibles_{an}"],
            "seuil 3": t[f"communes_seuil3_{an}"],
            "reconnues (arrêtés)": t[f"communes_reconnues_{an}"],
            "coût": t[f"cout_{an}"].map(_m),
            "reconstruction": t[f"cout_{an}_reconstruction"].map(_m),
            "avec seuil 3": t[f"cout_{an}_avec_seuil3"].map(_m),
        })
        u.loc["total"] = ["Grand Est", f"{tot['maisons']:,.0f}", "",
                          int(tot[f"communes_eligibles_{an}"]), int(tot[f"communes_seuil3_{an}"]),
                          int(tot[f"communes_reconnues_{an}"]), _m(tot[f"cout_{an}"]),
                          _m(tot[f"cout_{an}_reconstruction"]),
                          _m(tot[f"cout_{an}_avec_seuil3"])]
        print(u.to_string())

    print(f"\n=== Comparaison ONRN {DEBUT}-{FIN} (M€ ; ONRN : euros 2021, tout le marché) ===")
    u = pd.DataFrame({
        "département": t["departement"],
        "si toutes reconnues": t["cout_si_toutes_reconnues"].map(_m),
        "R (comm.-ans)": t["communes_annees_reconnues_9521"].astype(int),
        "E SWI (comm.-ans)": t["communes_annees_eligibles_swi_9521"].astype(int),
        "modèle × R": t["modele_x_reconnaissances_9521"].map(_m),
        "modèle × E": t["modele_x_eligibilite_swi_9521"].map(_m),
        "ONRN cumul": [f"{_m(b)} - {_m(h)}" for b, h in
                       zip(t["onrn_cumul_9521_bas"], t["onrn_cumul_9521_haut"])],
        "ONRN / an": [f"{_m(b / (FIN - DEBUT + 1))} - {_m(h / (FIN - DEBUT + 1))}" for b, h in
                      zip(t["onrn_cumul_9521_bas"], t["onrn_cumul_9521_haut"])],
    })
    u.loc["total"] = ["Grand Est", _m(tot["cout_si_toutes_reconnues"]),
                      int(tot["communes_annees_reconnues_9521"]),
                      int(tot["communes_annees_eligibles_swi_9521"]),
                      _m(tot["modele_x_reconnaissances_9521"]),
                      _m(tot["modele_x_eligibilite_swi_9521"]),
                      f"{_m(tot['onrn_cumul_9521_bas'])} - {_m(tot['onrn_cumul_9521_haut'])}",
                      f"{_m(tot['onrn_cumul_9521_bas'] / (FIN - DEBUT + 1))} - "
                      f"{_m(tot['onrn_cumul_9521_haut'] / (FIN - DEBUT + 1))}"]
    print(u.to_string())

    print("\n=== Coût par sinistre : modèle et classes ONRN du coût moyen (communes à classe) ===")
    cm = cm.copy()
    cm.insert(0, "modele_moyenne_k€", t["cout_par_sinistre_modele"] / 1e3)
    print(cm.rename(index=DEPARTEMENTS).to_string(float_format="{:.2f}".format))


# --------------------------------------------------------------------------- #

def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--annees", type=int, nargs="+", default=[ANNEE],
                   help=f"années de sinistre à estimer (défaut {ANNEE})")
    p.add_argument("--annee-dommage", default=str(ANNEE),
                   help=f"année dont l'effet GLM fixe le niveau des coûts (défaut {ANNEE})")
    p.add_argument("--swi", type=Path, default=SWI, help="dossier des parquets SWI du Grand Est")
    p.add_argument("--catnat", type=Path, default=CATNAT_DEFAUT)
    p.add_argument("--intensite", nargs="+", default=[], metavar="VARIANTE",
                   help="variantes de frequence_secheresse_swi.py modulant tau, p. ex. "
                        "deficit_iid deficit_bym2")
    p.add_argument("--severite", nargs="+", default=[], metavar="VARIANTE",
                   help="variantes de severite_secheresse_swi.py modulant le coût par sinistre, "
                        "une par indice de --intensite, p. ex. deficit sswi deficit2")
    p.add_argument("--plafond-s", action="store_true",
                   help="intensité ramenée au maximum observé à l'estimation")
    p.add_argument("--tirages-cout", type=int, default=TIRAGES_COUT,
                   help=f"tirages a posteriori propagés au coût (défaut {TIRAGES_COUT})")
    a = p.parse_args()
    if a.severite and not a.intensite:
        p.error("--severite demande --intensite")
    for f, script in [(MAISONS, "extract_maisons.py"), (PRIX, "prix_m2_departements.py"),
                      (TAUX, "taux_destruction_onrn.py"), (MODALITES, "glm_taux_dommage.py")]:
        if not f.exists():
            raise SystemExit(f"{f} introuvable : lancer d'abord {script}")

    constante, effets = effets_glm(COEFS, MODALITES)
    if a.annee_dommage not in effets["annee"]:
        raise SystemExit(f"--annee-dommage {a.annee_dommage} absente du GLM {MODELE} : "
                         f"{', '.join(sorted(effets['annee']))}")
    niveau = constante * effets["annee"][a.annee_dommage]
    e_an = sorted(set(effets["annee"].values()))
    print(f"GLM {MODELE} : constante {constante:.4f}, effet {a.annee_dommage} "
          f"{effets['annee'][a.annee_dommage]:.2f} (effets annuels de {e_an[0]:.2f} à {e_an[-1]:.2f})")

    # Éligibilité
    swi, mailles = charger_swi(a.swi)
    annees = sorted(set(a.annees) | set(range(DEBUT, FIN + 1)))
    anormales = mailles_anormales(swi, annees)
    print(f"SWI : {len(mailles)} mailles, {swi['annee'].min()}-{swi['annee'].max()}")
    for an in a.annees:
        x = anormales[anormales["annee"] == an]
        print(f"  {an} : seuil 1 {x['seuil1'].sum()}, seuil 2 {x['seuil2'].sum()}, "
              f"période de retour >= 5 ans {x['rp5'].sum()} mailles")
    communes = referentiel()
    communes = communes[communes["code_departement"].isin(GRAND_EST)].reset_index(drop=True)
    corr = correspondance(communes, mailles)
    sans = ~communes[CODE].isin(corr[CODE])
    print(f"communes 2026 : {len(communes):,} ; {corr['maille'].nunique()} mailles les couvrent, "
          f"{corr.groupby(CODE).size().median():.0f} par commune en médiane ; "
          f"{sans.sum()} communes sans maille, jamais éligibles")
    elig = eligibilite_communes(anormales, corr, communes, voisinages(communes, 1)[0])

    # Coût d'une commune reconnue
    print("maisons :")
    d = maisons_par_commune(MAISONS, effets)
    taux = pd.read_csv(TAUX, dtype={CODE: str}, usecols=[CODE, "libelle_commune", "tau_retenu"])
    d = d.merge(taux, on=CODE, how="left")
    prix = prix_departements(PRIX)
    d["prix_m2"] = d["dep"].map(prix)
    d["effet_dep"] = d["dep"].map(effets["dep"]).fillna(1.0)
    d["cout_par_sinistre"] = (niveau * d["effet_dep"] * d["prix_m2"]
                              * d["surface_ponderee"] / d["n_maisons"])
    d["cout_si_reconnue"] = (d["tau_retenu"] * d["n_maisons"] * d["cout_par_sinistre"]).fillna(0)
    print(f"  {d['n_maisons'].sum():,.0f} maisons dans {len(d):,} communes ; "
          f"tau manquant : {d['tau_retenu'].isna().sum()} communes")

    reco = annees_reconnues(a.catnat)
    r = reco[reco["annee"].between(DEBUT, FIN)].groupby(CODE).size()
    e = elig[elig["annee"].between(DEBUT, FIN)].groupby(CODE)["eligible"].sum()
    d["n_annees_reconnues"] = d[CODE].map(r).fillna(0).astype(int)
    d["n_annees_eligibles_swi"] = d[CODE].map(e).fillna(0).astype(int)
    for an in a.annees:
        x = elig[elig["annee"] == an]
        d[f"eligible_{an}"] = d[CODE].isin(x.loc[x["eligible"], CODE])
        d[f"seuil3_{an}"] = d[CODE].isin(x.loc[x["seuil3"], CODE])
        d[f"reconnue_{an}"] = d[CODE].isin(reco.loc[reco["annee"] == an, CODE])
        d[f"cout_{an}"] = d["cout_si_reconnue"].where(d[f"eligible_{an}"], 0.0)

    onrn = lire_onrn_couts(ONRN_CUMUL, ONRN_MOYEN)
    t = par_departement(d, onrn, a.annees)
    rapporter(t, confronter_cout_moyen(d, onrn), a.annees)

    sorties = {"communes": None, "departements": None}
    if a.intensite:
        totaux, par_commune = modulations(d, elig, reco, a.annees, a.intensite, a.severite,
                                          a.plafond_s, a.tirages_cout)
        sorties["intensite"] = rapporter_intensite(totaux)
        par_commune = pd.DataFrame(par_commune, index=d.index)
        d = pd.concat([d, par_commune], axis=1)
        for col in par_commune:
            t[col] = d.groupby("dep")[col].sum()
        print("\ncoût moyen a posteriori par département (M€)")
        print(t[["departement", *par_commune.columns]].to_string(
            formatters={c: _m for c in par_commune.columns}))

    d = d.merge(onrn[[CODE, "classe_cout_cumule", "classe_cout_moyen"]], on=CODE, how="left")
    OUT.mkdir(exist_ok=True)
    sorties.update({"communes": d, "departements": t.rename_axis("dep").reset_index()})
    print()
    for quoi, x in sorties.items():
        x.to_csv(str(SORTIE).format(quoi), index=False)
        print(f"{quoi} -> {str(SORTIE).format(quoi)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
