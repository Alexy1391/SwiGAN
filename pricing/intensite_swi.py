#!/usr/bin/env python3
"""Intensité des sécheresses par commune et par année : le SWI en magnitude.

Le rang du SWI sur 30 ans (eligibilite_catnat.py) décide de l'éligibilité, mais
mesure mal l'intensité : 88 % des sinistres du Grand Est ont un rang de 1 à 3,
2003, 2018 et 2022 sont tous « rang 1 » là où ils frappent, et la fenêtre
glissante range le même SWI différemment d'une décennie à l'autre. Ce script
mesure l'intensité sur une climatologie FIXE (--reference, 1981-2010), la même
pour l'observé et pour les trajectoires SwiGAN : un futur plus sec s'y lit
comme une intensité plus forte, au lieu de se fondre dans sa propre fenêtre.

TROIS MAGNITUDES, PAR MAILLE SAFRAN
-----------------------------------
  sswi      anomalie du mois le plus sec, en écarts-types de la référence :
                sswi = (moyenne_ref - SWI_min) / écart-type_ref
            SWI_min est l'indicateur annuel de la brochure (minimum des douze
            SWI mensuels). Positif quand l'année est plus sèche que la normale,
            sans plafond, contrairement au rang.
  deficit   déficit de juin à octobre sous le quantile 20 % du mois :
                deficit = Σ_{mois 6..10} max(0, q20_ref(maille, mois) - SWI)
            en SWI × mois, nul les années ordinaires ; la saison où les
            argiles se rétractent.
  deficit2  deficit de l'année et de la précédente : les retraits successifs
            (2018-2020) s'additionnent.

Les magnitudes d'une commune sont la moyenne de celles des mailles qui la
couvrent, pondérées par la surface d'intersection : la position des maisons
n'est connue qu'en Grand Est (BDNB), une pondération par les maisons ne
s'étendrait pas à la France. Le rang, les seuils 1 et 2 et l'éligibilité
restent ceux de la brochure : la maille la plus sèche décide (rang minimal,
éligible dès qu'une maille l'est).

IDENTIFICATION (diagnostic imprimé)
-----------------------------------
Dans un modèle à effet commune et effet année, la pente de S ne repose que sur
la part de S qui n'est ni une moyenne de commune ni une moyenne d'année. Le
script imprime cette part de variance, sur toutes les communes-années et sur
les seules communes-années reconnues (l'échantillon du modèle de fréquence),
en France et en Grand Est. Sous 5 %, la pente n'est pas identifiable, quel que
soit le modèle.

CACHE
-----
cache/intensite_swi_france.parquet : une ligne par (commune 2026, année),
données publiques seulement (SWI, contours). Reconstruit avec --reconstruire,
ou s'il manque. La correspondance commune-maille, la plus longue à calculer,
est gardée à part (cache/correspondance_communes_mailles_france.parquet),
indexée par les coordonnées des mailles.

Prérequis : SWI de SwiGAN/Outputs/rcp_8.5/France ; contours en cache
(prix_m2_communes.py). Le diagnostic lit ../catnat.csv.
Usage : python3 intensite_swi.py [--reconstruire] [--annees 2003 2022]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import spearmanr

from cout_secheresse_departements import CODE, charger_swi, correspondance
from eligibilite_catnat import (FENETRE_DEFAUT, RANG_5_ANS, RANG_10_ANS, appliquer_seuils,
                                rangs_glissants)
from extract_maisons import GRAND_EST
from extraire_catnat import CATNAT_DEFAUT
from prix_m2_communes import CACHE, referentiel, table_passage
from taux_destruction_onrn import annees_reconnues

ROOT = Path(__file__).resolve().parent
SWI_FRANCE = ROOT.parent.parent / "SwiGAN" / "Outputs" / "rcp_8.5" / "France"
FICHIERS_FRANCE = ["data_train_france", "data_val_france", "data_test_france"]
CACHE_INTENSITE = CACHE / "intensite_swi_france.parquet"
CACHE_CORRESPONDANCE = CACHE / "correspondance_communes_mailles_france.parquet"

REFERENCE = (1981, 2010)
MOIS_DEFICIT = [6, 7, 8, 9, 10]
QUANTILE_DEFICIT = 0.2
INDICES = ["sswi", "deficit", "deficit2"]
ANNEES_DIAGNOSTIC = (2003, 2022)
PART_MINIMALE = 0.05


# --------------------------------------------------------------------------- #
# Magnitudes par maille                                                        #
# --------------------------------------------------------------------------- #

def cube_mensuel(swi: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """SWI en tableau (maille, année, mois), NaN où le mois manque."""
    mailles = np.sort(swi["maille"].unique())
    annees = np.arange(swi["annee"].min(), swi["annee"].max() + 1)
    cube = np.full((len(mailles), len(annees), 12), np.nan, dtype=np.float64)
    cube[np.searchsorted(mailles, swi["maille"]), swi["annee"].to_numpy() - annees[0],
         swi["mois"].to_numpy() - 1] = swi["swi"].to_numpy()
    return cube, mailles, annees


def climatologie(cube: np.ndarray, annees: np.ndarray, reference: tuple[int, int]) -> dict:
    """Moyenne et écart-type de SWI_min, quantile 20 % de chaque mois, sur la référence."""
    ref = (annees >= reference[0]) & (annees <= reference[1])
    if ref.sum() < reference[1] - reference[0] + 1:
        raise SystemExit(f"référence {reference} incomplète : SWI {annees[0]}-{annees[-1]}")
    mini = cube[:, ref, :].min(axis=2)
    return {"moyenne": mini.mean(axis=1), "ecart_type": mini.std(axis=1, ddof=1),
            "q20": np.quantile(cube[:, ref, :], QUANTILE_DEFICIT, axis=1)}


def magnitudes(cube: np.ndarray, clim: dict) -> dict[str, np.ndarray]:
    """sswi, deficit et deficit2 par (maille, année) ; NaN si l'année est incomplète."""
    complete = ~np.isnan(cube).any(axis=2)
    swi_min = np.where(complete, np.nanmin(np.where(np.isnan(cube), np.inf, cube), axis=2),
                       np.nan)
    sswi = (clim["moyenne"][:, None] - swi_min) / clim["ecart_type"][:, None]
    m = np.array(MOIS_DEFICIT) - 1
    manque = np.clip(clim["q20"][:, None, m] - cube[:, :, m], 0, None)
    deficit = np.where(complete, manque.sum(axis=2), np.nan)
    deficit2 = np.full_like(deficit, np.nan)
    deficit2[:, 1:] = deficit[:, 1:] + deficit[:, :-1]
    return {"swi_min": swi_min, "sswi": sswi, "deficit": deficit, "deficit2": deficit2}


# --------------------------------------------------------------------------- #
# Des mailles aux communes                                                     #
# --------------------------------------------------------------------------- #

def communes_metropole():
    """Communes 2026 de métropole (Corse comprise) : la grille SAFRAN n'en couvre pas d'autres."""
    c = referentiel()
    return c[~c["code_departement"].str.startswith("97")].reset_index(drop=True)


def correspondance_france(communes, mailles: pd.DataFrame, reconstruire: bool) -> pd.DataFrame:
    """(commune, maille, surface), en cache par les coordonnées (x, y) des mailles."""
    if CACHE_CORRESPONDANCE.exists() and not reconstruire:
        corr = pd.read_parquet(CACHE_CORRESPONDANCE)
    else:
        print("correspondance communes-mailles (France) : intersection des contours ...",
              flush=True)
        corr = correspondance(communes, mailles).merge(mailles, on="maille")
        corr = corr[[CODE, "x", "y", "surface"]]
        CACHE_CORRESPONDANCE.parent.mkdir(parents=True, exist_ok=True)
        corr.to_parquet(CACHE_CORRESPONDANCE, index=False)
    return corr.merge(mailles, on=["x", "y"])[[CODE, "maille", "surface"]]


def vers_communes(corr: pd.DataFrame, mailles: np.ndarray, annees: np.ndarray,
                  mag: dict[str, np.ndarray], rangs: np.ndarray,
                  seuils: dict[str, np.ndarray]) -> pd.DataFrame:
    """Une ligne par (commune, année) : magnitudes pondérées, rang et seuils de la maille la plus sèche."""
    corr = corr.sort_values([CODE, "maille"], ignore_index=True)
    codes, debut = np.unique(corr[CODE].to_numpy(), return_index=True)
    ligne = np.searchsorted(codes, corr[CODE].to_numpy())
    col = np.searchsorted(mailles, corr["maille"].to_numpy())
    poids = corr["surface"].to_numpy() / corr.groupby(CODE)["surface"].transform("sum").to_numpy()
    w = sparse.csr_matrix((poids, (ligne, col)), shape=(len(codes), len(mailles)))
    sortie = {k: (w @ np.nan_to_num(v, nan=0.0)).astype(np.float32) for k, v in mag.items()}
    # Une commune dont une maille n'a pas de valeur n'en a pas : la moyenne serait tronquée.
    for k, v in mag.items():
        trous = sparse.csr_matrix((np.ones(len(col)), (ligne, col)),
                                  shape=w.shape) @ np.isnan(v).astype(float)
        sortie[k][trous > 0] = np.nan
    r = rangs[col]
    sortie["rang"] = np.fmin.reduceat(r, debut, axis=0)
    for k in ["seuil1", "seuil2"]:
        sortie[k] = np.logical_or.reduceat(seuils[k][col], debut, axis=0)
    d = pd.DataFrame({CODE: np.repeat(codes, len(annees)), "annee": np.tile(annees, len(codes)),
                      **{k: v.ravel() for k, v in sortie.items()}})
    d["eligible"] = d["seuil1"] | d["seuil2"]
    return d


def construire(dossier: Path, reference: tuple[int, int], reconstruire: bool) -> pd.DataFrame:
    swi, mailles = charger_swi(dossier, FICHIERS_FRANCE)
    print(f"SWI : {len(mailles):,} mailles, {swi['annee'].min()}-{swi['annee'].max()}")
    cube, ids, annees = cube_mensuel(swi)
    mag = magnitudes(cube, climatologie(cube, annees, reference))
    rangs = rangs_glissants(mag["swi_min"].astype(np.float32), FENETRE_DEFAUT)
    seuils = appliquer_seuils(rangs, RANG_10_ANS, RANG_5_ANS)
    communes = communes_metropole()
    corr = correspondance_france(communes, mailles, reconstruire)
    d = vers_communes(corr, ids, annees, mag, rangs, seuils)
    sans = len(communes) - d[CODE].nunique()
    print(f"communes 2026 de métropole : {len(communes):,}, dont {sans} sans maille SAFRAN")
    return d


def charger_intensite(reconstruire: bool = False, dossier: Path = SWI_FRANCE,
                      reference: tuple[int, int] = REFERENCE) -> pd.DataFrame:
    """Table (commune, année) des magnitudes, du rang et de l'éligibilité, depuis le cache."""
    if CACHE_INTENSITE.exists() and not reconstruire:
        return pd.read_parquet(CACHE_INTENSITE)
    d = construire(dossier, reference, reconstruire)
    CACHE_INTENSITE.parent.mkdir(parents=True, exist_ok=True)
    d.to_parquet(CACHE_INTENSITE, index=False)
    print(f"intensité -> {CACHE_INTENSITE}")
    return d


def reconnues_2026(catnat: Path) -> pd.DataFrame:
    """Communes-années couvertes par un arrêté sécheresse, codes ramenés aux communes 2026."""
    r = annees_reconnues(catnat)
    actuels = set(communes_metropole()[CODE])
    r[CODE] = r[CODE].map(table_passage(r[CODE], actuels))
    return r.dropna(subset=[CODE]).drop_duplicates([CODE, "annee"]).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Diagnostic d'identification                                                  #
# --------------------------------------------------------------------------- #

def part_residuelle(d: pd.DataFrame, col: str, iterations: int = 200,
                    tolerance: float = 1e-10) -> dict[str, float]:
    """Part de la variance de `col` hors effets commune, hors effets année, hors les deux.

    Deux effets fixes croisés sur un panel incomplet : projections alternées
    (moyenne par commune retirée, puis par année, jusqu'à convergence).
    """
    x = d[col].to_numpy(dtype=float)
    x = x - x.mean()
    total = (x ** 2).sum()
    commune, annee = d[CODE].to_numpy(), d["annee"].to_numpy()

    def retirer(v: np.ndarray, g: np.ndarray) -> np.ndarray:
        return v - pd.Series(v).groupby(g).transform("mean").to_numpy()

    r = x.copy()
    for _ in range(iterations):
        avant = r
        r = retirer(retirer(r, commune), annee)
        if np.abs(r - avant).max() < tolerance:
            break
    return {"hors annee": (retirer(x, annee) ** 2).sum() / total,
            "hors commune": (retirer(x, commune) ** 2).sum() / total,
            "hors commune et annee": (r ** 2).sum() / total}


def diagnostiquer(d: pd.DataFrame, reco: pd.DataFrame, annees: tuple[int, int]) -> pd.DataFrame:
    d = d[d["annee"].between(*annees)].dropna(subset=INDICES)
    reco = reco[reco["annee"].between(*annees)]
    echantillons = {"toutes": d, "reconnues": d.merge(reco[[CODE, "annee"]], on=[CODE, "annee"])}
    lignes = []
    for perimetre, garde in [("France", None), ("Grand Est", GRAND_EST)]:
        for nom, e in echantillons.items():
            if garde is not None:
                e = e[e[CODE].str[:2].isin(garde)]
            for col in [*INDICES, "rang"]:
                x = e.dropna(subset=[col])
                lignes.append({"perimetre": perimetre, "echantillon": nom, "indice": col,
                               "communes_annees": len(x), "communes": x[CODE].nunique(),
                               **part_residuelle(x, col)})
    return pd.DataFrame(lignes)


def decrire(d: pd.DataFrame, reco: pd.DataFrame) -> None:
    x = d.dropna(subset=INDICES)
    print("\nmagnitudes moyennes par année (France, communes non pondérées) et part des "
          "communes éligibles ; années de plus de 2 000 communes reconnues")
    n_reco = reco.groupby("annee").size()
    grandes = n_reco[(n_reco > 2000) & (n_reco.index >= 1990)].index
    t = x[x["annee"].isin(grandes)].groupby("annee").agg(
        sswi=("sswi", "mean"), sswi_p90=("sswi", lambda s: s.quantile(.9)),
        deficit=("deficit", "mean"), deficit2=("deficit2", "mean"),
        eligibles=("eligible", "mean"))
    t["reconnues"] = n_reco.reindex(t.index)
    print(t.to_string(float_format=lambda v: f"{v:.2f}"))
    y = x[x["annee"] >= 1989]
    print("\ncorrélations de rang (Spearman) entre indices, communes-années 1989-2025 :")
    cols = [*INDICES, "rang"]
    rho = spearmanr(y[cols].to_numpy()).statistic
    print(pd.DataFrame(rho, index=cols, columns=cols).to_string(float_format=lambda v: f"{v:+.2f}"))


# --------------------------------------------------------------------------- #

def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--reconstruire", action="store_true",
                   help="recalculer le cache (correspondance comprise)")
    p.add_argument("--swi", type=Path, default=SWI_FRANCE, help="dossier des parquets SWI France")
    p.add_argument("--reference", type=int, nargs=2, default=list(REFERENCE),
                   metavar=("DEBUT", "FIN"), help="années de la climatologie (défaut 1981 2010)")
    p.add_argument("--annees", type=int, nargs=2, default=list(ANNEES_DIAGNOSTIC),
                   metavar=("DEBUT", "FIN"), help="fenêtre du diagnostic (défaut 2003 2022)")
    p.add_argument("--catnat", type=Path, default=CATNAT_DEFAUT)
    a = p.parse_args()

    d = charger_intensite(a.reconstruire, a.swi, tuple(a.reference))
    reco = reconnues_2026(a.catnat)
    decrire(d, reco)
    diag = diagnostiquer(d, reco, tuple(a.annees))
    print(f"\npart de la variance de S hors effets commune et année, {a.annees[0]}-{a.annees[1]} "
          "(rang : 1 = le plus sec)")
    t = diag.set_index(["perimetre", "echantillon", "indice"])
    print(t.to_string(formatters={"communes_annees": "{:,}".format, "communes": "{:,}".format,
                                  "hors annee": "{:.1%}".format, "hors commune": "{:.1%}".format,
                                  "hors commune et annee": "{:.1%}".format}))
    cle = diag[(diag["perimetre"] == "France") & (diag["echantillon"] == "reconnues")
               & diag["indice"].isin(INDICES)]
    pire = cle["hors commune et annee"].min()
    verdict = "identifiable" if pire >= PART_MINIMALE else "NON identifiable"
    print(f"\nFrance, communes-années reconnues : part résiduelle minimale {pire:.1%} "
          f"(seuil {PART_MINIMALE:.0%}) -> pente de S {verdict}")
    sortie = ROOT / "output" / "intensite_swi_diagnostic.csv"
    sortie.parent.mkdir(exist_ok=True)
    diag.to_csv(sortie, index=False)
    print(f"diagnostic -> {sortie}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
