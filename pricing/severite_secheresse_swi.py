#!/usr/bin/env python3
"""Coût d'un sinistre sécheresse et intensité du SWI : modèle hiérarchique national, h(S).

glm_secheresse_swi.py n'a trouvé, en Grand Est, aucun effet de l'intensité sur
le coût par sinistre, et un effet inversé avec un effet fixe par année. Ce
script reprend la question sur les sinistres de toute la France, avec la
magnitude de intensite_swi.py au lieu du rang, et une structure qui donne à la
pente l'incertitude qui lui revient.

MODÈLE (un sinistre de charge > 0 dans une commune-année reconnue, --annees 2003 2022)
------------------------------------------------------------------------------------
    ch_i ~ Gamma(k, k / μ_i)                    moyenne μ_i, coefficient de variation 1/√k
    log μ_i = β0 + βC·zC_c + βh·zS_ct + βhC·zS_ct·zC_c + d_dép + u_ct + v_t

ch      CH_TOT déflatée par l'indice FFB du coût de la construction, en euros
        de --annee-reference (comme glm_secheresse_swi.py) ;
zS, zC  intensité de la commune-année et part de ses maisons en aléa argile
        moyen ou fort (SDES), centrées réduites sur les sinistres du modèle ;
d_dép   effet département aléatoire : niveau des prix, taille des maisons,
        pratiques de gestion ;
u_ct    effet commune-année aléatoire (« épisode ») : les sinistres d'une même
        commune une même année partagent leur intensité, leur sol, souvent leur
        expert. Sans lui, chaque sinistre compterait comme une observation
        indépendante de S, et la pente paraîtrait bien plus précise qu'elle
        n'est (pseudo-réplication) ;
v_t     effet année aléatoire : inflation résiduelle, maturité, règles de
        reconnaissance.
Priors : βh, βhC ~ N(0, 0,2²), sceptiques : à un écart-type a priori, un
écart-type d'intensité change le coût de ± 20 %. βC ~ N(0, 0,5²) ;
σ_dép, σ_épisode, σ_année ~ demi-normale(0,5) ; k ~ Gamma(2, 1).

h ET SA CALIBRATION
-------------------
    h(S, argile) = exp(βh·zS + βhC·zS·zC)
Le coût d'une maison sinistrée est déjà fixé par t_m × p_d × S_m
(glm_taux_dommage.py, ajusté sur les sinistres du Grand Est). h ne fait que le
moduler selon l'intensité, sans en changer le niveau moyen :
    facteur_ct = h(S_ct, argile_c) / h̄
    h̄ = moyenne de h sur les sinistres sécheresse du Grand Est de l'échantillon
Sur les sinistres qui ont servi à fixer le niveau, le facteur vaut 1 en
moyenne ; une commune-année plus sèche que la moyenne de ces sinistres le
relève si βh > 0, l'abaisse si βh < 0. h̄ est calculé tirage par tirage et
écrit avec les tirages : l'incertitude sur βh passe ainsi au coût.

RETENU OU NON
-------------
h est déclaré « retenu » si l'intervalle a posteriori à 90 % de βh exclut 0.
Sinon le coût reste sans modulation de sévérité (la piste 1 de la note).
cout_secheresse_departements.py ne l'applique que si --severite le demande, et
imprime alors le coût avec et sans h.

CONFIDENTIALITÉ
---------------
Comme frequence_secheresse_swi.py : base lue en place, aucun sinistre ni
agrégat par commune écrit ; seulement des paramètres et des moyennes.

Prérequis : intensite_swi.py (cache), indice_ffb.py (sinon le site de la FFB
est interrogé), contours (prix_m2_communes.py).
Usage : python3 severite_secheresse_swi.py [--intensite deficit] [--annees 2003 2022]
            [--echantillon reconnues|eligibles] [--annee-reference 2024] [--sin CHEMIN]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("PYTENSOR_FLAGS", "cxx=")

import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt

from apparier_sinistres_maisons import trouver_base
from cout_secheresse_departements import CODE
from extract_maisons import GRAND_EST
from extraire_catnat import CATNAT_DEFAUT
from frequence_secheresse_swi import (ANNEES, CHAINES, INTENSITE, RODAGE, TIRAGES, assembler,
                                      centrer, echantillonner, effets_annee, exposition_france,
                                      multiplicateur, resumer, sinistres_secheresse,
                                      standardiser, tirages_posterieurs)
from glm_secheresse_swi import ANNEE_REFERENCE, deflateur
from glm_taux_dommage import MIN_MODALITE
from intensite_swi import INDICES, charger_intensite, reconnues_2026

ROOT = Path(__file__).resolve().parent
SORTIE = ROOT / "output" / "severite_secheresse_swi_{variante}_{quoi}"

PRIOR_INTENSITE = 0.2
NIVEAU_RETENU = 0.90
PARAMETRES = ["beta0", "beta_argile", "beta_s", "beta_s_argile", "sigma_departement",
              "sigma_episode", "sigma_annee", "forme"]


# --------------------------------------------------------------------------- #
# Données et modèle                                                            #
# --------------------------------------------------------------------------- #

def sinistres_du_modele(sin: pd.DataFrame, d: pd.DataFrame, reference: int) -> pd.DataFrame:
    """Sinistres des communes-années du modèle, charge en euros constants."""
    x = sin.merge(d[[CODE, "annee", "s", "argile"]], on=[CODE, "annee"])
    x["ch"] = x["ch"] * deflateur(x["annee"], reference).to_numpy()
    x["dep"] = x[CODE].str[:2]
    return x.reset_index(drop=True)


def construire_modele(x: pd.DataFrame, meta: dict, prior: float) -> tuple[pm.Model, dict]:
    deps, idep = np.unique(x["dep"].to_numpy(), return_inverse=True)
    episodes, iep = np.unique((x[CODE] + "_" + x["annee"].astype(str)).to_numpy(),
                              return_inverse=True)
    annees, it = np.unique(x["annee"].to_numpy(), return_inverse=True)
    zs, zc = centrer(x["s"], x["argile"], meta)
    niveau = float(np.log(x["ch"].mean()))
    with pm.Model() as m:
        b0 = pm.Normal("beta0", 0, 1)
        bc = pm.Normal("beta_argile", 0, 0.5)
        bh = pm.Normal("beta_s", 0, prior)
        bhc = pm.Normal("beta_s_argile", 0, prior)
        # Départements et années portent des centaines de sinistres : effets
        # centrés (l'année autour de β0) ; les épisodes, souvent un ou deux
        # sinistres, restent non centrés.
        sd = pm.HalfNormal("sigma_departement", 0.5)
        dep = pm.Normal("d_departement", 0, sd, shape=len(deps))
        su = pm.HalfNormal("sigma_episode", 0.5)
        zu = pm.Normal("z_episode", 0, 1, shape=len(episodes))
        sv = pm.HalfNormal("sigma_annee", 0.5)
        v = pm.Normal("v_annee", b0, sv, shape=len(annees))
        eta = (niveau + bc * zc + bh * zs + bhc * zs * zc
               + dep[idep] + su * zu[iep] + v[it])
        k = pm.Gamma("forme", 2, 1)
        pm.Gamma("ch", alpha=k, beta=k / pt.exp(eta), observed=x["ch"].to_numpy())
    return m, {"annees": annees, "departements": len(deps), "episodes": len(episodes)}


def calibration(tirages: pd.DataFrame, meta: dict, x: pd.DataFrame) -> np.ndarray:
    """h̄ par tirage : moyenne de h sur les sinistres du Grand Est de l'échantillon."""
    ge = x[x[CODE].str[:2].isin(GRAND_EST)]
    if ge.empty:
        raise SystemExit("aucun sinistre du Grand Est dans l'échantillon : h̄ indéfini")
    return multiplicateur(tirages, meta, ge["s"], ge["argile"]).mean(axis=1)


# --------------------------------------------------------------------------- #
# Rapport (agrégats uniquement)                                                #
# --------------------------------------------------------------------------- #

def decrire(x: pd.DataFrame, reference: int, mini: int) -> None:
    print(f"\nsinistres du modèle : {len(x):,} dans {x[CODE].nunique():,} communes, "
          f"{(x[CODE] + x['annee'].astype(str)).nunique():,} communes-années, "
          f"{x['dep'].nunique()} départements ; dont Grand Est "
          f"{x[CODE].str[:2].isin(GRAND_EST).sum():,}")
    print(f"charge en euros {reference} : " + ", ".join(
        f"q{int(q * 100)} {v:,.0f} €" for q, v in x["ch"].quantile([.1, .5, .9]).items()))
    g = x.groupby("annee").agg(sinistres=("ch", "size"), charge_mediane=("ch", "median"),
                               s_moyen=("s", "mean"))
    petites = g.index[g["sinistres"] < mini]
    if len(petites):
        r = x[x["annee"].isin(petites)]
        g = pd.concat([g.drop(index=petites), pd.DataFrame(
            {"sinistres": [len(r)], "charge_mediane": [r["ch"].median()],
             "s_moyen": [r["s"].mean()]}, index=["autres"])])
    print(f"par année (années de moins de {mini} sinistres regroupées)")
    print(g.to_string(formatters={"sinistres": "{:,.0f}".format,
                                  "charge_mediane": "{:,.0f}".format, "s_moyen": "{:.3f}".format}))


def rapporter(coefs: pd.DataFrame, tirages: pd.DataFrame, meta: dict, x: pd.DataFrame) -> None:
    print("\nparamètres a posteriori (moyenne, intervalle à 90 %, P(> 0), R-hat, ESS)")
    print(coefs.set_index("parametre").to_string(float_format=lambda v: f"{v:.3f}"))
    b = tirages["beta_s"]
    lo, hi = b.quantile([(1 - NIVEAU_RETENU) / 2, (1 + NIVEAU_RETENU) / 2])
    print(f"\nβh : {b.mean():+.3f} [{lo:+.3f} ; {hi:+.3f}], P(βh > 0) = {(b > 0).mean():.2f} -> "
          f"h {'RETENU' if meta['retenu'] else 'non retenu'}")
    q10, q90 = x["s"].quantile([.1, .9])
    for libelle, part in [("aucune", 0.0), ("médiane", float(x["argile"].median())),
                          ("toutes", 1.0)]:
        h = multiplicateur(tirages, meta, [q90, q10], [part, part])
        r = h[:, 0] / h[:, 1]
        print(f"  coût au décile 9 / décile 1 d'intensité, maisons en argile : {libelle} "
              f"({part:.1%}) : × {np.mean(r):.2f} "
              f"[{np.quantile(r, .05):.2f} - {np.quantile(r, .95):.2f}]")
    ge = x[x[CODE].str[:2].isin(GRAND_EST)]
    t = tirages.sample(n=min(500, len(tirages)), random_state=0)
    f = (multiplicateur(t, meta, ge["s"], ge["argile"])
         / t["hbar"].to_numpy()[:, None]).mean(axis=0)
    print("facteur h / h̄ moyen des sinistres du Grand Est par année : "
          + ", ".join(f"{a} {v:.2f}" for a, v in
                      pd.Series(f).groupby(ge["annee"].to_numpy()).mean().items()))


# --------------------------------------------------------------------------- #

def variante(a) -> str:
    v = a.intensite
    if a.echantillon != "reconnues":
        v += f"_{a.echantillon}"
    if tuple(a.annees) != ANNEES:
        v += f"_{a.annees[0]}-{a.annees[1]}"
    if a.prior_intensite != PRIOR_INTENSITE:
        v += f"_ph{a.prior_intensite:g}"
    return v


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--intensite", choices=INDICES, default=INTENSITE)
    p.add_argument("--annees", type=int, nargs=2, default=list(ANNEES), metavar=("DEBUT", "FIN"))
    p.add_argument("--echantillon", choices=["reconnues", "eligibles"], default="reconnues")
    p.add_argument("--prior-intensite", type=float, default=PRIOR_INTENSITE,
                   help=f"écart-type a priori de βh et βhC (défaut {PRIOR_INTENSITE})")
    p.add_argument("--annee-reference", type=int, default=ANNEE_REFERENCE)
    p.add_argument("--niveaux", nargs="+", default=["TOUS"], metavar="NIVEAU")
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
    sin, etapes, doublons, _ = sinistres_secheresse(chemin, a.niveaux)
    print(f"doublons de CLE retirés : {doublons:,} ; sinistres sécheresse retenus : {len(sin):,}")
    d, _ = assembler(sin, reconnues_2026(a.catnat), charger_intensite(), exposition_france(),
                     a.annees, a.intensite, a.echantillon)
    x = sinistres_du_modele(sin, d, a.annee_reference)
    decrire(x, a.annee_reference, a.min_modalite)
    meta = {"intensite": a.intensite, "annees": list(a.annees), "echantillon": a.echantillon,
            "annee_reference": a.annee_reference, "prior_intensite": a.prior_intensite,
            **standardiser(x), "sinistres": len(x)}

    print(f"\n[{nom}] échantillonnage : {a.chaines} chaînes × ({a.rodage} rodage + {a.tirages} "
          "tirages)", flush=True)
    m, info = construire_modele(x, meta, a.prior_intensite)
    trace = echantillonner(m, a.tirages, a.rodage, a.chaines, a.graine)
    coefs = resumer(trace, PARAMETRES)
    tirages = tirages_posterieurs(trace, ["beta0", "beta_argile", "beta_s", "beta_s_argile"])
    tirages["hbar"] = calibration(tirages, meta, x)
    b = tirages["beta_s"]
    lo, hi = b.quantile([(1 - NIVEAU_RETENU) / 2, (1 + NIVEAU_RETENU) / 2])
    meta.update({"retenu": bool(lo > 0 or hi < 0), "p_positif": float((b > 0).mean()),
                 "r_hat_max": float(coefs["r_hat"].max()),
                 "divergences": int(trace.sample_stats["diverging"].sum()),
                 "episodes": info["episodes"], "departements": info["departements"]})
    rapporter(coefs, tirages, meta, x)
    annuels = effets_annee(trace, info["annees"])
    print("\neffet année exp(v_t), moyenne a posteriori : "
          + ", ".join(f"{p.split(' = ')[1]} {v:.2f}" for p, v in
                      zip(annuels["parametre"], annuels["moyenne"])))

    print()
    for quoi, t in {"coefs.csv": pd.concat([coefs, annuels], ignore_index=True),
                    "tirages.csv": tirages}.items():
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
