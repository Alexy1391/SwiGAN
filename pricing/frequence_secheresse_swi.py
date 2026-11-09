#!/usr/bin/env python3
"""Fréquence des sinistres sécheresse et intensité du SWI : modèle hiérarchique national.

Le coût d'une commune reconnue repose sur τ_c, la part de ses maisons sinistrées
une année de reconnaissance (taux_destruction_onrn.py) : une moyenne ONRN sur
1995-2021, la même chaque année, qu'elle soit sèche ou exceptionnelle. Ce
script estime comment la fréquence varie avec l'intensité de la sécheresse,
sur les sinistres Allianz de toute la France : les sécheresses touchent des
régions différentes selon les années (2003, 2011, 2018, 2022), ce qui donne
bien plus de contrastes intensité × lieu que les cinq grandes années du seul
Grand Est (intensite_swi.py : part de variance identifiante 14 % en France,
7 % en Grand Est pour le déficit).

MODÈLE (une ligne par commune-année reconnue, --annees 2003 2022)
----------------------------------------------------------------
    N_ct ~ BinomialeNégative(μ_ct, α)
    log μ_ct = log(maisons_c) + β0 + βC·zC_c + βS·zS_ct + βSC·zS_ct·zC_c + a_c + v_t

N_ct    sinistres sécheresse Allianz de charge > 0 survenus (DTSURSIN, à
        défaut ANNEE) dans la commune c l'année t, positionnés dans les
        contours 2026 ; les communes-années reconnues sans sinistre comptent 0,
        sauf les années sans aucun sinistre dans toute la base (2000-2002 :
        la base commence en 2003), qui sont écartées ;
maisons maisons individuelles de la commune (SDES, Fideli 2017) ;
zC      part des maisons en aléa argile moyen ou fort (BRGM 2019), centrée
        réduite ; zS l'intensité de --intensite (intensite_swi.py), centrée réduite ;
v_t     effet année aléatoire, σ_v ~ demi-normale(--prior-annee) : il absorbe
        ce que l'intensité n'explique pas (règles de reconnaissance, gestion,
        maturité), et une année nouvelle se prédit avec v ~ N(0, σ_v²) ;
a_c     effet commune, qui absorbe la part de marché inconnue d'Allianz (pas
        de table d'exposition) et la vulnérabilité propre de la commune :
          iid   a_c = σ_a·θ_c                                (--effet-commune iid)
          bym2  a_c = σ_a·(√(ρ/s)·φ_c + √(1-ρ)·θ_c)          (--effet-commune bym2)
        φ suit un ICAR sur le graphe des communes limitrophes de métropole
        (îles rattachées à la commune la plus proche), s est le facteur
        d'échelle de Riebler et al. (2016), ρ la part spatiale de la variance.
Priors : β0 ~ N(0, 2) (taux moyen dans l'offset), βC, βS, βSC ~ N(0, 1),
σ_a ~ demi-normale(1), α ~ Gamma(2, 0,1), ρ ~ Bêta(1, 1).

Avec a_c et v_t, la pente de l'intensité vient de ses variations au sein d'une
commune d'une année reconnue à l'autre, au-delà de ce qui est commun à toute
l'année. Un effet commune spatialement structuré (BYM2) peut absorber une part
de l'intensité si celle-ci est elle-même lisse dans l'espace (confusion
spatiale, Hodges et Reich 2010) : comparer βS entre iid et bym2 en est le test.

SEUIL ET PENTE (--seuil)
------------------------
Avant 2024, des communes ont été reconnues sans franchir le critère SWI
(circulaire de 2019, communes limitrophes) ; leur fréquence est plus basse.
Sur les seules communes-années éligibles (--echantillon eligibles), la pente
de l'intensité devient presque nulle : une seule pente log-linéaire mêle le
saut au seuil et le gradient au-delà, et extrapolerait le saut aux sécheresses
plus intenses. --seuil ajoute γ·E_ct, E = 1 si la commune-année est éligible
(seuil 1 ou 2, intensite_swi.py) : γ porte le saut, βS le seul gradient.

MULTIPLICATEUR DE FRÉQUENCE ET CALIBRATION (lecture de cout_secheresse_departements.py)
--------------------------------------------------------------------------------------
    g(S, argile, E) = exp(γ·E + βS·zS + βSC·zS·zC)        (γ = 0 sans --seuil)
Le niveau reste celui de l'ONRN, l'intensité ne fait que le moduler :
    τ_ct = τ_c × g_ct / ḡ_c,   ḡ_c = moyenne de g sur les années 1995-2021
                                     où la commune a été reconnue
si bien que la moyenne de τ_ct sur ces années redonne τ_c. Une année tarifée
selon la règle de 2024 est éligible (E = 1) ; les années reconnues de ḡ_c le
sont ou non : avec --seuil, g / ḡ_c corrige ainsi le niveau ONRN, moyenne
d'années éligibles et non éligibles, avant de le moduler par l'intensité. Les
tirages a posteriori de (γ, βS, βSC) sont écrits, avec la standardisation de S
et de l'argile, pour propager l'incertitude jusqu'au coût.

VALIDATION (--valider K)
------------------------
Les années sont réparties en K plis ; chaque pli est prédit par un modèle
ajusté sans lui, avec et sans intensité. Densité prédictive d'une année
retirée : son effet année est intégré (quadrature de Gauss-Hermite sur
N(0, σ_v²)), une commune absente de l'apprentissage reçoit un effet tiré dans
N(0, σ_a²). Disponible pour l'effet commune iid.

CONFIDENTIALITÉ
---------------
Base lue en place par son chemin (CLE, LATITUDE, LONGITUDE, GEOCODING_LEVEL,
PERIL, ANNEE, DTSURSIN, CH_TOT), doublons de CLE retirés ; ni les sinistres ni
les comptages par commune ne sont écrits. Les sorties ne portent que des
paramètres du modèle (effets communes exclus) et des agrégats ; à l'écran, les
années de moins de --min-modalite sinistres sont regroupées.

Paris, Lyon et Marseille sont écartés : le SDES les compte en villes entières,
les contours 2026 en arrondissements.

MÉMOIRE
-------
nutpie garde chaque variable libre à chaque itération, rodage compris : φ
porte 34 788 valeurs. Le modèle bym2 tient dans 14 Go de RAM avec
--rodage 700 --tirages 700 (4 chaînes, 15 min), pas avec 1000 + 1000.

Prérequis : intensite_swi.py (cache), contours (prix_m2_communes.py).
Usage : python3 frequence_secheresse_swi.py [--intensite deficit] [--effet-commune iid|bym2]
            [--annees 2003 2022] [--echantillon reconnues|eligibles] [--prior-annee 0.5]
            [--valider 5] [--rodage 1000] [--tirages 1000] [--sin CHEMIN]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings
from pathlib import Path

os.environ.setdefault("PYTENSOR_FLAGS", "cxx=")

import arviz as az
import geopandas as gpd
import numpy as np
import nutpie
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import splu
from scipy.special import gammaln, logsumexp
from sklearn.neighbors import BallTree

from apparier_sinistres_maisons import (LAMBERT93, LAT_MAX, LAT_MIN, LON_MAX, LON_MIN, WGS84,
                                        lire_sinistres, trouver_base)
from cout_secheresse_departements import CODE
from extraire_catnat import CATNAT_DEFAUT
from glm_taux_dommage import MIN_MODALITE
from intensite_swi import INDICES, charger_intensite, communes_metropole, reconnues_2026
from prix_m2_communes import CACHE, table_passage, telecharger, voisinages
from sinistres_valeur import lire_annee, lire_montant
from taux_destruction_onrn import lire_feuille

ROOT = Path(__file__).resolve().parent
SORTIE = ROOT / "output" / "frequence_secheresse_swi_{variante}_{quoi}"

URL_RGA = "https://www.statistiques.developpement-durable.gouv.fr/media/4553/download?inline"
CACHE_EXPOSITION = CACHE / "exposition_france.parquet"
COLONNES_SIN = ["CLE", "LATITUDE", "LONGITUDE", "GEOCODING_LEVEL", "PERIL", "ANNEE",
                "DTSURSIN", "CH_TOT"]
# Codes des arrondissements de Paris, Lyon et Marseille (le SDES ne les distingue pas).
ARRONDISSEMENTS = ("751", "6938", "132")

ANNEES = (2003, 2022)
INTENSITE = "deficit"
PRIOR_ANNEE = 0.5
TIRAGES, RODAGE, CHAINES = 1000, 1000, 4
NOEUDS_QUADRATURE = 20
PARAMETRES = ["beta0", "beta_argile", "gamma_eligible", "beta_s", "beta_s_argile",
              "sigma_commune", "sigma_annee", "alpha", "rho"]


# --------------------------------------------------------------------------- #
# Données                                                                      #
# --------------------------------------------------------------------------- #

def exposition_france() -> pd.DataFrame:
    """Par commune 2026 : maisons individuelles (SDES) et part en aléa argile moyen ou fort."""
    if CACHE_EXPOSITION.exists():
        return pd.read_parquet(CACHE_EXPOSITION)
    t = lire_feuille(telecharger(URL_RGA, "sdes/indicateur_rga_communes.xlsx"), "RGA_2019_commune")
    for c in ["nb_logement", "nb_logement_alea_moyen_fort"]:
        t[c] = pd.to_numeric(t[c])
    actuels = set(communes_metropole()[CODE])
    t[CODE] = t["insee_com"].map(table_passage(t["insee_com"], actuels))
    g = (t.dropna(subset=[CODE]).groupby(CODE)[["nb_logement", "nb_logement_alea_moyen_fort"]]
         .sum().reset_index())
    g["argile"] = g["nb_logement_alea_moyen_fort"] / g["nb_logement"].where(g["nb_logement"] > 0)
    g = g.rename(columns={"nb_logement": "maisons"})[[CODE, "maisons", "argile"]]
    CACHE_EXPOSITION.parent.mkdir(parents=True, exist_ok=True)
    g.to_parquet(CACHE_EXPOSITION, index=False)
    return g


def sinistres_secheresse(chemin: Path, niveaux: list[str]
                         ) -> tuple[pd.DataFrame, list[tuple[str, int]], int, dict[str, float]]:
    """Sinistres sécheresse de charge > 0 : commune 2026, année de survenance, charge.

    Ne garde ni la CLE ni la position : le jeu reste en mémoire et n'est jamais écrit.
    """
    brut, doublons = lire_sinistres(chemin, COLONNES_SIN)
    etapes = [("lignes de la base, doublons retirés", len(brut))]
    d = brut[brut["PERIL"].str.contains("cheresse", case=False)].copy()
    etapes.append(("péril sécheresse", len(d)))
    d["ch"] = lire_montant(d["CH_TOT"])
    d = d[d["ch"] > 0]
    etapes.append(("charge CH_TOT > 0", len(d)))
    lat, lon = lire_montant(d["LATITUDE"]), lire_montant(d["LONGITUDE"])
    d = d[lat.between(LAT_MIN, LAT_MAX) & lon.between(LON_MIN, LON_MAX)]
    etapes.append(("position en métropole", len(d)))
    if niveaux != ["TOUS"]:
        d = d[d["GEOCODING_LEVEL"].isin(niveaux)]
        etapes.append((f"GEOCODING_LEVEL {', '.join(niveaux)}", len(d)))
    survenance, annee = lire_annee(d["DTSURSIN"]), lire_annee(d["ANNEE"])
    deux = survenance.notna() & annee.notna()
    qualite = {"DTSURSIN lue": float(survenance.notna().mean()),
               "DTSURSIN = ANNEE": float((survenance[deux] == annee[deux]).mean())}
    d["annee"] = survenance.fillna(annee)
    d = d[d["annee"].notna()]
    etapes.append(("année de survenance lue", len(d)))

    points = gpd.GeoDataFrame(d[["annee", "ch"]].astype({"annee": int}),
                              geometry=gpd.points_from_xy(lon[d.index], lat[d.index]), crs=WGS84)
    communes = communes_metropole()[[CODE, "geometry"]].to_crs(LAMBERT93)
    j = gpd.sjoin(points.to_crs(LAMBERT93), communes, how="inner", predicate="within")
    j = j[~j.index.duplicated()]
    etapes.append(("dans une commune 2026", len(j)))
    j = j[~j[CODE].str.startswith(ARRONDISSEMENTS)]
    etapes.append(("hors Paris, Lyon, Marseille", len(j)))
    return pd.DataFrame(j[[CODE, "annee", "ch"]]).reset_index(drop=True), etapes, doublons, qualite


def assembler(sin: pd.DataFrame, reco: pd.DataFrame, intensite: pd.DataFrame,
              expo: pd.DataFrame, annees: tuple[int, int], indice: str,
              echantillon: str) -> tuple[pd.DataFrame, list[tuple[str, int]]]:
    """Communes-années reconnues avec leur nombre de sinistres, maisons, argile et intensité."""
    r = reco[reco["annee"].between(*annees) & ~reco[CODE].str.startswith(ARRONDISSEMENTS)]
    etapes = [(f"communes-années reconnues {annees[0]}-{annees[1]}", len(r))]
    # Une année reconnue sans aucun sinistre sécheresse dans toute la base n'est
    # pas couverte par la base : ses zéros ne seraient pas des fréquences nulles.
    hors = sorted(set(r["annee"]) - set(sin["annee"]))
    if hors:
        r = r[~r["annee"].isin(hors)]
        etapes.append((f"années couvertes par la base (hors {', '.join(map(str, hors))})",
                       len(r)))
    r = r.merge(expo, on=CODE)
    r = r[(r["maisons"] > 0) & r["argile"].notna()]
    etapes.append(("avec des maisons (SDES)", len(r)))
    r = r.merge(intensite[[CODE, "annee", indice, "eligible"]], on=[CODE, "annee"])
    r = r[r[indice].notna()]
    etapes.append(("avec une intensité SWI", len(r)))
    if echantillon == "eligibles":
        r = r[r["eligible"]]
        etapes.append(("éligibles au critère SWI (seuil 1 ou 2)", len(r)))
    n = sin.groupby([CODE, "annee"]).size().rename("sinistres").reset_index()
    r = r.merge(n, on=[CODE, "annee"], how="left").fillna({"sinistres": 0})
    r["sinistres"] = r["sinistres"].astype(int)
    r["s"] = r[indice]
    return r.reset_index(drop=True), etapes


def standardiser(d: pd.DataFrame) -> dict[str, float]:
    return {"s_moyenne": float(d["s"].mean()), "s_ecart_type": float(d["s"].std()),
            "argile_moyenne": float(d["argile"].mean()),
            "argile_ecart_type": float(d["argile"].std()),
            "s_min": float(d["s"].min()), "s_max": float(d["s"].max())}


def centrer(s, argile, meta: dict) -> tuple[np.ndarray, np.ndarray]:
    return ((np.asarray(s, dtype=float) - meta["s_moyenne"]) / meta["s_ecart_type"],
            (np.asarray(argile, dtype=float) - meta["argile_moyenne"]) / meta["argile_ecart_type"])


def multiplicateur(tirages: pd.DataFrame, meta: dict, s, argile,
                   plafond: bool = False, eligible=None) -> np.ndarray:
    """g = exp(γ·E + βS·zS + βSC·zS·zC), une ligne par tirage a posteriori, une colonne par point.

    plafond  : S ramené au maximum observé dans l'échantillon d'estimation (pas d'extrapolation) ;
    eligible : E, éligibilité SWI de la commune-année, pour un modèle ajusté avec --seuil
               (sans elle, ou sans γ dans les tirages, le terme est omis).
    """
    s = np.asarray(s, dtype=float)
    if plafond:
        s = np.minimum(s, meta["s_max"])
    zs, zc = centrer(s, argile, meta)
    b = tirages["beta_s"].to_numpy()[:, None]
    bc = tirages["beta_s_argile"].to_numpy()[:, None]
    eta = b * zs[None, :] + bc * (zs * zc)[None, :]
    if eligible is not None and "gamma_eligible" in tirages:
        e = np.asarray(eligible, dtype=float)
        eta = eta + tirages["gamma_eligible"].to_numpy()[:, None] * e[None, :]
    return np.exp(eta)


# --------------------------------------------------------------------------- #
# Graphe des communes et facteur d'échelle BYM2                                #
# --------------------------------------------------------------------------- #

def graphe_communes(communes: gpd.GeoDataFrame) -> tuple[np.ndarray, np.ndarray, int]:
    """Arêtes (i < j) entre communes limitrophes ; chaque île est reliée à la commune la plus proche."""
    a = voisinages(communes, 1)[0]
    a = ((a + a.T) > 0).astype(float)
    n_comp, etiquette = connected_components(a, directed=False)
    i, j = sparse.triu(a, k=1).nonzero()
    if n_comp > 1:
        centres = communes.to_crs(LAMBERT93).geometry.centroid
        xy = np.column_stack([centres.x, centres.y])
        principale = np.bincount(etiquette).argmax()
        dans = np.flatnonzero(etiquette == principale)
        arbre = BallTree(xy[dans])
        ajout_i, ajout_j = [], []
        for k in set(range(n_comp)) - {principale}:
            membres = np.flatnonzero(etiquette == k)
            dist, idx = arbre.query(xy[membres], k=1)
            m = int(np.argmin(dist[:, 0]))
            ajout_i.append(membres[m])
            ajout_j.append(dans[idx[m, 0]])
        i, j = np.concatenate([i, ajout_i]), np.concatenate([j, ajout_j])
    return i.astype(np.int64), j.astype(np.int64), n_comp


def facteur_echelle(n: int, i: np.ndarray, j: np.ndarray, echantillon: int = 2000,
                    graine: int = 0) -> float:
    """Moyenne géométrique des variances marginales de l'ICAR sous contrainte de somme nulle.

    diag(L⁺) par le laplacien « mis à la terre » (nœud 0 retiré, L0 définie
    positive) : L⁺ = P·G·P, G = L0⁻¹ complétée de zéros, P = I - 11ᵀ/n, d'où
    L⁺_ii = G_ii - 2 (G1)_i / n + 1ᵀG1 / n². Estimée sur un échantillon de nœuds.
    """
    w = sparse.coo_matrix((np.ones(len(i)), (i, j)), shape=(n, n))
    w = (w + w.T).tocsr()
    lap = (sparse.diags(np.asarray(w.sum(axis=1)).ravel()) - w).tocsc()
    lu = splu(lap[1:, 1:].tocsc())
    g1 = lu.solve(np.ones(n - 1))
    s1 = g1.sum()
    noeuds = np.random.default_rng(graine).choice(np.arange(1, n), size=min(echantillon, n - 1),
                                                   replace=False)
    variances = []
    for bloc in np.array_split(noeuds, max(1, len(noeuds) // 250)):
        e = np.zeros((n - 1, len(bloc)))
        e[bloc - 1, np.arange(len(bloc))] = 1.0
        gii = lu.solve(e)[bloc - 1, np.arange(len(bloc))]
        variances.append(gii - 2 * g1[bloc - 1] / n + s1 / n ** 2)
    return float(np.exp(np.mean(np.log(np.concatenate(variances)))))


# --------------------------------------------------------------------------- #
# Modèle                                                                       #
# --------------------------------------------------------------------------- #

def construire_modele(d: pd.DataFrame, meta: dict, effet: str, avec_s: bool,
                      prior_annee: float, graphe: dict | None = None,
                      seuil: bool = False) -> tuple[pm.Model, dict]:
    communes, ic = np.unique(d[CODE].to_numpy(), return_inverse=True)
    annees, it = np.unique(d["annee"].to_numpy(), return_inverse=True)
    zs, zc = centrer(d["s"], d["argile"], meta)
    # Taux moyen dans l'offset : β0 n'est plus qu'un écart, d'ordre 1.
    taux = d["sinistres"].sum() / d["maisons"].sum()
    offset = np.log(d["maisons"].to_numpy()) + np.log(taux)
    y = d["sinistres"].to_numpy()
    with pm.Model() as m:
        b0 = pm.Normal("beta0", 0, 2)
        bc = pm.Normal("beta_argile", 0, 1)
        eta = offset + bc * zc
        if seuil:
            ge = pm.Normal("gamma_eligible", 0, 1)
            eta = eta + ge * d["eligible"].to_numpy(dtype=float)
        if avec_s:
            bs = pm.Normal("beta_s", 0, 1)
            bsc = pm.Normal("beta_s_argile", 0, 1)
            eta = eta + bs * zs + bsc * zs * zc
        sa = pm.HalfNormal("sigma_commune", 1)
        theta = pm.Normal("theta", 0, 1, shape=len(communes))
        if effet == "bym2":
            n = graphe["n"]
            phi = pm.Flat("phi", shape=n)
            pm.Potential("icar", -0.5 * pt.sum((phi[graphe["i"]] - phi[graphe["j"]]) ** 2))
            pm.Potential("somme_nulle",
                         pm.logp(pm.Normal.dist(0, 0.001 * n), pt.sum(phi)))
            rho = pm.Beta("rho", 1, 1)
            position = graphe["position"].loc[communes].to_numpy()
            a = sa * (pt.sqrt(rho / graphe["echelle"]) * phi[position] + pt.sqrt(1 - rho) * theta)
        else:
            a = sa * theta
        sv = pm.HalfNormal("sigma_annee", prior_annee)
        # Effet année centré sur β0 : chaque année compte des centaines ou des
        # milliers de communes, et β0 séparé d'effets non centrés mélangeait mal.
        v = pm.Normal("v_annee", b0, sv, shape=len(annees))
        eta = eta + a[ic] + v[it]
        alpha = pm.Gamma("alpha", 2, 0.1)
        pm.NegativeBinomial("sinistres", mu=pt.exp(eta), alpha=alpha, observed=y)
    return m, {"communes": communes, "annees": annees, "offset_taux": float(np.log(taux))}


def echantillonner(m: pm.Model, tirages: int, rodage: int, chaines: int, graine: int):
    """NUTS (nutpie, backend numba : pas de compilateur C++ sous Windows).

    Le rodage n'entre pas dans la trace ; nutpie réserve pourtant la mémoire
    de toutes les itérations, champs latents compris (voir MÉMOIRE en tête).
    """
    t = time.time()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        compile_ = nutpie.compile_pymc_model(m, backend="numba")
    t_compile = time.time() - t
    trace = nutpie.sample(compile_, draws=tirages, tune=rodage, chains=chaines, seed=graine,
                          save_warmup=False, progress_bar=False)
    print(f"  compilation {t_compile:.0f} s, échantillonnage {time.time() - t - t_compile:.0f} s ; "
          f"divergences {int(trace.sample_stats['diverging'].sum())}", flush=True)
    return trace


def resumer(trace, noms: list[str]) -> pd.DataFrame:
    """Moyenne, écart-type, quantiles 5/50/95 %, P(> 0), R-hat et ESS des paramètres scalaires."""
    noms = [n for n in noms if n in trace.posterior]
    s = az.summary(trace, var_names=noms, kind="diagnostics")
    lignes = []
    for n in noms:
        x = trace.posterior[n].to_numpy().ravel()
        lignes.append({"parametre": n, "moyenne": x.mean(), "ecart_type": x.std(),
                       "q05": np.quantile(x, .05), "q50": np.quantile(x, .5),
                       "q95": np.quantile(x, .95), "p_positif": (x > 0).mean(),
                       "r_hat": s.loc[n, "r_hat"], "ess_bulk": s.loc[n, "ess_bulk"]})
    return pd.DataFrame(lignes)


def effets_annee(trace, annees: np.ndarray) -> pd.DataFrame:
    """exp(v_t - β0) : écart de chaque année à l'année moyenne."""
    v = (trace.posterior["v_annee"].to_numpy()
         - trace.posterior["beta0"].to_numpy()[..., None]).reshape(-1, len(annees))
    e = np.exp(v)
    return pd.DataFrame({"parametre": [f"annee = {a}" for a in annees], "moyenne": e.mean(axis=0),
                         "ecart_type": e.std(axis=0), "q05": np.quantile(e, .05, axis=0),
                         "q50": np.quantile(e, .5, axis=0), "q95": np.quantile(e, .95, axis=0)})


def tirages_posterieurs(trace, noms: list[str]) -> pd.DataFrame:
    return pd.DataFrame({n: trace.posterior[n].to_numpy().ravel()
                         for n in noms if n in trace.posterior})


# --------------------------------------------------------------------------- #
# Validation par plis d'années (effet commune iid)                             #
# --------------------------------------------------------------------------- #

def _logpmf_nb(y: np.ndarray, mu: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    return (gammaln(y + alpha) - gammaln(alpha) - gammaln(y + 1)
            + alpha * np.log(alpha / (alpha + mu)) + y * np.log(mu / (alpha + mu)))


def densite_predictive(trace, info: dict, test: pd.DataFrame, meta: dict, avec_s: bool,
                       graine: int, max_tirages: int = 400) -> pd.Series:
    """log p(N d'une année retirée), effet année intégré, par année du pli."""
    post = trace.posterior
    noms = ["beta0", "beta_argile", "sigma_commune", "sigma_annee", "alpha"]
    tir = {n: post[n].to_numpy().ravel() for n in noms + (["beta_s", "beta_s_argile"]
                                                           if avec_s else [])}
    theta = post["theta"].to_numpy().reshape(-1, len(info["communes"]))
    rng = np.random.default_rng(graine)
    garde = rng.choice(len(tir["beta0"]), size=min(max_tirages, len(tir["beta0"])), replace=False)
    zs, zc = centrer(test["s"], test["argile"], meta)
    pos = pd.Index(info["communes"]).get_indexer(test[CODE])
    connue = pos >= 0
    noeuds, poids = np.polynomial.hermite.hermgauss(NOEUDS_QUADRATURE)
    y = test["sinistres"].to_numpy()
    base = np.log(test["maisons"].to_numpy()) + info["offset_taux"]
    annees = test["annee"].to_numpy()
    lp = {a: [] for a in np.unique(annees)}
    gamma = post["gamma_eligible"].to_numpy().ravel() if "gamma_eligible" in post else None
    eligible = test["eligible"].to_numpy(dtype=float)
    for s in garde:
        eta = base + tir["beta0"][s] + tir["beta_argile"][s] * zc
        if gamma is not None:
            eta = eta + gamma[s] * eligible
        if avec_s:
            eta = eta + tir["beta_s"][s] * zs + tir["beta_s_argile"][s] * zs * zc
        a = np.where(connue, tir["sigma_commune"][s] * theta[s, np.maximum(pos, 0)],
                     tir["sigma_commune"][s] * rng.standard_normal(len(pos)))
        eta = eta + a
        v = np.sqrt(2) * tir["sigma_annee"][s] * noeuds
        for an in lp:
            k = annees == an
            mu = np.exp(eta[k][:, None] + v[None, :])
            ll = _logpmf_nb(y[k][:, None], mu, tir["alpha"][s]).sum(axis=0)
            lp[an].append(logsumexp(ll, b=poids / np.sqrt(np.pi)))
    return pd.Series({an: logsumexp(v) - np.log(len(v)) for an, v in lp.items()})


def valider(d: pd.DataFrame, meta: dict, plis: int, prior_annee: float, tirages: int,
            rodage: int, chaines: int, graine: int, seuil: bool = False) -> pd.DataFrame:
    """Avec et sans intensité ; l'effet du seuil (--seuil) reste dans les deux."""
    annees = np.unique(d["annee"])
    pli = {a: i % plis for i, a in enumerate(annees)}
    resultats = []
    for k in range(plis):
        test = d[d["annee"].map(pli) == k]
        app = d[d["annee"].map(pli) != k]
        print(f"pli {k + 1}/{plis} : années retirées {sorted(test['annee'].unique())}", flush=True)
        ligne = {}
        for avec_s in (True, False):
            m, info = construire_modele(app, meta, "iid", avec_s, prior_annee, seuil=seuil)
            trace = echantillonner(m, tirages, rodage, chaines, graine + k)
            ligne["avec S" if avec_s else "sans S"] = densite_predictive(
                trace, info, test, meta, avec_s, graine + k)
        resultats.append(pd.DataFrame(ligne))
    v = pd.concat(resultats).sort_index().rename_axis("annee").reset_index()
    n = d.groupby("annee").agg(communes_annees=("sinistres", "size"),
                               sinistres=("sinistres", "sum"))
    return v.merge(n, on="annee")


# --------------------------------------------------------------------------- #
# Rapport (agrégats uniquement)                                                #
# --------------------------------------------------------------------------- #

def decrire(sin: pd.DataFrame, d: pd.DataFrame, etapes_sin, etapes_d, doublons: int,
            qualite: dict, annees: tuple[int, int], mini: int) -> None:
    print(f"\ndoublons de CLE retirés de la base : {doublons:,}")
    print("sinistres, effectif après chaque filtre")
    for libelle, n in etapes_sin:
        print(f"  {libelle:<40}: {n:>8,}")
    print("  " + ", ".join(f"{k} {v:.1%}" for k, v in qualite.items()))
    dans = sin[sin["annee"].between(*annees)]
    print(f"\ncommunes-années du modèle, effectif après chaque filtre")
    for libelle, n in etapes_d:
        print(f"  {libelle:<40}: {n:>8,}")
    print(f"sinistres {annees[0]}-{annees[1]} : {len(dans):,}, dont {d['sinistres'].sum():,} "
          f"({d['sinistres'].sum() / max(len(dans), 1):.1%}) dans une commune-année du modèle ; "
          f"communes-années sans sinistre Allianz : {(d['sinistres'] == 0).mean():.1%}")
    g = d.groupby("annee").agg(communes_annees=("sinistres", "size"), sinistres=("sinistres", "sum"),
                               s_moyen=("s", "mean"))
    petites = g.index[g["sinistres"] < mini]
    if len(petites):
        autres = g.loc[petites].sum()
        autres["s_moyen"] = d.loc[d["annee"].isin(petites), "s"].mean()
        g = pd.concat([g.drop(index=petites), autres.to_frame("autres").T])
    g["sinistres_pour_1000_maisons"] = (
        d.groupby("annee")["sinistres"].sum() / d.groupby("annee")["maisons"].sum() * 1000
    ).reindex(g.index)
    print(f"\npar année (années de moins de {mini} sinistres regroupées)")
    print(g.to_string(formatters={"communes_annees": "{:,.0f}".format,
                                  "sinistres": "{:,.0f}".format, "s_moyen": "{:.3f}".format,
                                  "sinistres_pour_1000_maisons": "{:.2f}".format}))


def rapporter(coefs: pd.DataFrame, tirages: pd.DataFrame, d: pd.DataFrame, meta: dict) -> None:
    print("\nparamètres a posteriori (moyenne, intervalle à 90 %, P(> 0), R-hat, ESS)")
    print(coefs.set_index("parametre").to_string(float_format=lambda v: f"{v:.3f}"))
    q10, q90 = d["s"].quantile([.1, .9])
    print(f"\nmultiplicateur g entre S au décile 9 ({q90:.3f}) et au décile 1 ({q10:.3f}) "
          "des communes-années du modèle, selon la part de maisons en argile moyenne/forte :")
    for libelle, part in [("aucune", 0.0), ("médiane", float(d["argile"].median())),
                          ("toutes", 1.0)]:
        g = multiplicateur(tirages, meta, [q90, q10], [part, part])
        r = g[:, 0] / g[:, 1]
        print(f"  maisons en argile : {libelle} ({part:.1%}) : × {np.mean(r):.2f} "
              f"[{np.quantile(r, .05):.2f} - "
              f"{np.quantile(r, .95):.2f}]")
    if "gamma_eligible" in tirages:
        e = np.exp(tirages["gamma_eligible"])
        print(f"saut au seuil d'éligibilité SWI exp(γ) : × {e.mean():.2f} "
              f"[{e.quantile(.05):.2f} - {e.quantile(.95):.2f}] ; part des communes-années "
              f"éligibles {d['eligible'].mean():.0%}")
    echantillon = tirages.sample(n=min(200, len(tirages)), random_state=0)
    g = multiplicateur(echantillon, meta, d["s"], d["argile"],
                       eligible=d["eligible"]).mean(axis=0)
    print("multiplicateur moyen a posteriori par année (communes-années du modèle) : "
          + ", ".join(f"{a} {v:.2f}" for a, v in pd.Series(g).groupby(d["annee"].to_numpy())
                      .mean().items()))


# --------------------------------------------------------------------------- #

def variante(a) -> str:
    v = f"{a.intensite}_{a.effet_commune}"
    if a.echantillon != "reconnues":
        v += f"_{a.echantillon}"
    if tuple(a.annees) != ANNEES:
        v += f"_{a.annees[0]}-{a.annees[1]}"
    if a.prior_annee != PRIOR_ANNEE:
        v += f"_sv{a.prior_annee:g}"
    if a.seuil:
        v += "_seuil"
    return v


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--intensite", choices=INDICES, default=INTENSITE,
                   help=f"magnitude SWI de intensite_swi.py (défaut {INTENSITE})")
    p.add_argument("--effet-commune", choices=["iid", "bym2"], default="iid")
    p.add_argument("--annees", type=int, nargs=2, default=list(ANNEES), metavar=("DEBUT", "FIN"),
                   help=f"années de survenance (défaut {ANNEES[0]} {ANNEES[1]})")
    p.add_argument("--echantillon", choices=["reconnues", "eligibles"], default="reconnues",
                   help="toutes les communes-années reconnues, ou les seules éligibles au SWI")
    p.add_argument("--prior-annee", type=float, default=PRIOR_ANNEE,
                   help=f"échelle de la demi-normale de σ_v (défaut {PRIOR_ANNEE})")
    p.add_argument("--seuil", action="store_true",
                   help="effet propre de l'éligibilité SWI (seuil 1 ou 2), distinct de la pente")
    p.add_argument("--niveaux", nargs="+", default=["TOUS"], metavar="NIVEAU",
                   help="GEOCODING_LEVEL retenus (défaut : tous)")
    p.add_argument("--valider", type=int, default=0, metavar="K",
                   help="validation par K plis d'années, effet commune iid (défaut : non)")
    p.add_argument("--tirages", type=int, default=TIRAGES)
    p.add_argument("--rodage", type=int, default=RODAGE)
    p.add_argument("--chaines", type=int, default=CHAINES)
    p.add_argument("--graine", type=int, default=20261030)
    p.add_argument("--sin", type=Path, help="chemin de SIN_EVT_NAT.csv")
    p.add_argument("--catnat", type=Path, default=CATNAT_DEFAUT)
    p.add_argument("--min-modalite", type=int, default=MIN_MODALITE)
    a = p.parse_args()
    a.annees = tuple(a.annees)
    nom = variante(a)

    chemin = trouver_base(a.sin)
    print(f"Base de sinistres : {chemin}")
    sin, etapes_sin, doublons, qualite = sinistres_secheresse(chemin, a.niveaux)
    d, etapes_d = assembler(sin, reconnues_2026(a.catnat), charger_intensite(),
                            exposition_france(), a.annees, a.intensite, a.echantillon)
    decrire(sin, d, etapes_sin, etapes_d, doublons, qualite, a.annees, a.min_modalite)
    meta = {"intensite": a.intensite, "effet_commune": a.effet_commune, "annees": list(a.annees),
            "echantillon": a.echantillon, "prior_annee": a.prior_annee, "seuil": a.seuil,
            **standardiser(d),
            "communes_annees": len(d), "communes": int(d[CODE].nunique()),
            "sinistres": int(d["sinistres"].sum())}

    graphe = None
    if a.effet_commune == "bym2":
        communes = communes_metropole()
        i, j, n_comp = graphe_communes(communes)
        echelle = facteur_echelle(len(communes), i, j)
        position = pd.Series(np.arange(len(communes)), index=communes[CODE])
        graphe = {"n": len(communes), "i": i, "j": j, "echelle": echelle,
                  "position": position}
        print(f"\ngraphe BYM2 : {len(communes):,} communes, {len(i):,} arêtes ({n_comp - 1} "
              f"composantes isolées rattachées), facteur d'échelle {echelle:.3f}")
        meta["facteur_echelle"] = echelle

    print(f"\n[{nom}] échantillonnage : {a.chaines} chaînes × ({a.rodage} rodage + {a.tirages} "
          "tirages)", flush=True)
    m, info = construire_modele(d, meta, a.effet_commune, True, a.prior_annee, graphe, a.seuil)
    trace = echantillonner(m, a.tirages, a.rodage, a.chaines, a.graine)
    coefs = resumer(trace, PARAMETRES)
    meta["r_hat_max"] = float(coefs["r_hat"].max())
    meta["divergences"] = int(trace.sample_stats["diverging"].sum())
    tirages = tirages_posterieurs(trace, ["beta0", "beta_argile", "gamma_eligible", "beta_s",
                                          "beta_s_argile"])
    rapporter(coefs, tirages, d, meta)
    annuels = effets_annee(trace, info["annees"])
    print("\neffet année exp(v_t), moyenne a posteriori : "
          + ", ".join(f"{p.split(' = ')[1]} {v:.2f}" for p, v in
                      zip(annuels["parametre"], annuels["moyenne"])))

    sorties = {"coefs.csv": pd.concat([coefs, annuels], ignore_index=True),
               "tirages.csv": tirages}
    if a.valider:
        if a.effet_commune != "iid":
            print("\n--valider : effet commune iid seulement, validation sautée")
        else:
            print(f"\nvalidation par {a.valider} plis d'années (avec et sans intensité)")
            v = valider(d, meta, a.valider, a.prior_annee, max(300, a.tirages // 2),
                        max(300, a.rodage // 2), min(a.chaines, 4), a.graine, a.seuil)
            v["gain"] = v["avec S"] - v["sans S"]
            v["annee_affichee"] = v["annee"].astype(str).where(v["sinistres"] >= a.min_modalite,
                                                               "autres")
            t = v.groupby("annee_affichee")[["communes_annees", "sinistres", "avec S", "sans S",
                                              "gain"]].sum()
            t.loc["total"] = t.sum()
            print("densité prédictive log p(N_année) hors échantillon ; gain > 0 = l'intensité aide")
            print(t.to_string(float_format=lambda v: f"{v:,.1f}"))
            print(f"années mieux prédites avec l'intensité : {(v['gain'] > 0).sum()} sur {len(v)}")
            sorties["validation.csv"] = v.drop(columns="annee_affichee")
            meta["gain_validation"] = float(v["gain"].sum())

    print()
    for quoi, t in sorties.items():
        chemin_sortie = Path(str(SORTIE).format(variante=nom, quoi=quoi))
        chemin_sortie.parent.mkdir(parents=True, exist_ok=True)
        t.to_csv(chemin_sortie, index=False)
        print(f"{quoi} -> {chemin_sortie}")
    chemin_meta = Path(str(SORTIE).format(variante=nom, quoi="meta.json"))
    chemin_meta.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"meta.json -> {chemin_meta}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
