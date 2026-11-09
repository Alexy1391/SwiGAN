#!/usr/bin/env python3
"""Taux de dommage des maisons sinistrées : GLM du coût rapporté à la valeur.

Pour les sinistres rattachés à une maison individuelle par
apparier_sinistres_maisons.py, on modélise le taux

    taux = CH_TOT / V

en fonction des caractéristiques de la maison. Deux ajustements, à comparer :
tous périls confondus (péril en variable explicative), puis sécheresse seule.

DEUX VALEURS V POSSIBLES (--valeur)
-----------------------------------
reconstruction (défaut)
    V = surface de plancher estimée × coût de construction au m² (--cout-m2).
    La surface est celle d'extract_maisons.py : emprise au sol × nombre de
    niveaux. Toutes les maisons en ont une, Alsace-Moselle comprise. Le coût
    au m² étant le même pour toutes, il ne déplace que la constante du modèle :
    les exp(β) des caractéristiques n'en dépendent pas, seul le niveau du taux
    en dépend.
dvf
    V = valeur foncière médiane des ventes (batiment_groupe_dvf_open_statistique),
    pour les seules maisons vendues d'un seul tenant depuis 2014. Valeur vénale,
    terrain compris : en Grand Est le coût des sinistres n'en dépend pas
    (élasticité ≈ 0), si bien que le taux y reflète surtout le prix de la maison.

MODÈLE
------
GLM Gamma, lien log, réponse CH_TOT, offset log(valeur) :

    E[CH_TOT | X, V] = V · exp(Xβ)    soit    E[taux | X] = exp(Xβ)

exp(β) est l'effet multiplicatif d'une caractéristique sur le taux de dommage :
1,30 = taux 30 % plus élevé que la modalité de référence, toutes choses égales
par ailleurs. La loi Gamma convient à un coût strictement positif, asymétrique,
dont la dispersion croît avec la moyenne. Une variante libère le coefficient de
log(V) : une élasticité coût / valeur nettement inférieure à 1 signale que le
coût suit mal V, et que le taux baisse mécaniquement quand V augmente.

Covariables : nombre de niveaux, période de construction, aléa argile,
département, année de survenance, péril dans le modèle tous périls, et log de
l'emprise au sol en mode dvf (en mode reconstruction, V en dépend déjà :
l'emprise ne sert qu'à identifier l'élasticité). Le matériau des murs a été
retiré : sans effet global sur la sécheresse (test de Wald, p = 0,27). Une
modalité de moins de --min-modalite observations est fusionnée (avec sa voisine pour une variable ordonnée, dans
« autre » sinon). Les erreurs-types sont groupées par maison : une maison peut
porter plusieurs sinistres.

RÉSERVES DE LECTURE
-------------------
- Sévérité conditionnelle, pas fréquence : on ne sait pas quelles maisons du
  parc sont assurées, donc pas quelle est la probabilité d'un sinistre.
- CH_TOT est nette de franchise et de recours : les petits sinistres absorbés
  par la franchise n'apparaissent pas (charge nulle, exclue).
- reconstruction : la surface de plancher est géométrique (emprise ×
  niveaux), pas habitable, et le coût au m² ne varie ni selon la région ni
  selon le matériau ; l'effet département en absorbe une partie. Les maisons à emprise fictive (sans géométrie réelle) sont exclues.
- dvf : valeur vénale terrain compris ; seules les maisons vendues depuis 2014
  en ont une, aucune en Alsace-Moselle (57, 67, 68 : livre foncier) ; la vente
  et le sinistre ne datent pas de la même année.
- L'effet année de survenance absorbe le niveau général des coûts.

CONFIDENTIALITÉ
---------------
La base de sinistres est lue en place, par son chemin, colonnes CLE, ANNEE,
PERIL et CH_TOT seulement, doublons de CLE retirés comme à l'appariement. Le
jeu de modélisation n'est jamais écrit sur disque ; l'écran et
`output/glm_taux_dommage_<valeur>_coefs.csv` ne portent que des agrégats et des
coefficients, `_modalites.csv` que la correspondance entre modalités brutes et
modalités du modèle (fusions comprises), sans effectif.

Prérequis : apparier_sinistres_maisons.py puis extract_maisons.py.
Usage : python3 glm_taux_dommage.py [--valeur reconstruction|dvf] [--cout-m2 1800]
                                    [--sin CHEMIN] [--min-modalite 10]
"""

from __future__ import annotations

import argparse
import re
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests

from apparier_sinistres_maisons import BDNB, GRAND_EST, lire_sinistres, trouver_base
from sinistres_valeur import VALEUR_MAX, VALEUR_MIN, lire_annee, lire_montant

ROOT = Path(__file__).resolve().parent
APPARIEMENT = ROOT / "output" / "sinistres_maisons_grand_est.csv"
MAISONS = ROOT / "output" / "maisons_grand_est.csv"
SORTIE = ROOT / "output" / "glm_taux_dommage_{valeur}_coefs.csv"

COLONNES_SIN = ["CLE", "ANNEE", "PERIL", "CH_TOT"]
# Coût de construction d'une maison individuelle, en €/m² de plancher : ordre
# de grandeur, à remplacer par une référence (barème assureur, indice FFB).
# N'affecte que la constante du modèle.
COUT_M2 = 1800
MIN_MODALITE = 10
MIN_OBS_MODELE = 30

# Variables ordonnées : modalités dans l'ordre, la référence en tête de
# REFERENCE. Les autres variables prennent pour référence la plus fréquente.
PERIODES = ["<1919", "1919-1945", "1946-1974", "1975-1999", "2000+"]
ORDONNEES = {"niveaux": ["1", "2", "3+"], "periode": PERIODES,
             "alea": ["Faible", "Moyen", "Fort"]}
REFERENCE = {"niveaux": "1", "periode": "1946-1974", "alea": "Faible"}


# --------------------------------------------------------------------------- #
# Données                                                                      #
# --------------------------------------------------------------------------- #

def dvf_statistique(deps: list[str]) -> pd.DataFrame:
    parts = []
    for dep in deps:
        csv = BDNB / f"dep_{dep}" / "csv" / "batiment_groupe_dvf_open_statistique.csv"
        if csv.exists():
            parts.append(pd.read_csv(csv, sep=";", dtype={"batiment_groupe_id": str},
                                     usecols=["batiment_groupe_id", "valeur_fonciere_median",
                                              "nb_appartement_mutee",
                                              "nb_locaux_tertiaire_mutee"]))
    return pd.concat([p for p in parts if not p.empty])


def construire(sin: pd.DataFrame, app: pd.DataFrame, maisons: pd.DataFrame,
               valeur: str, cout_m2: float) -> tuple[pd.DataFrame, list[tuple[str, int]]]:
    """Jeu de modélisation et effectif restant après chaque filtre."""
    d = app[["ligne_source", "batiment_groupe_id"]].merge(sin, on="ligne_source")
    d = d.merge(maisons, on="batiment_groupe_id", how="left")
    d["ch"] = lire_montant(d["CH_TOT"])

    if valeur == "dvf":
        d = d.merge(dvf_statistique(GRAND_EST), on="batiment_groupe_id", how="left")
        d["valeur"] = d["valeur_fonciere_median"]
        filtres = [
            ("maison avec valeur DVF", d["valeur"].notna()),
            ("vente d'une maison seule", d["dvf_mutation_simple"].eq(True)
                & d["nb_appartement_mutee"].fillna(0).eq(0)
                & d["nb_locaux_tertiaire_mutee"].fillna(0).eq(0)),
            (f"valeur dans [{VALEUR_MIN:,} ; {VALEUR_MAX:,}] €",
                d["valeur"].between(VALEUR_MIN, VALEUR_MAX)),
        ]
    else:
        d["valeur"] = d["surface_plancher_estimee_m2"] * cout_m2
        filtres = [("emprise réelle (géométrie non fictive)", d["geometrie_fictive"].eq(False))]

    etapes = [("sinistres rattachés à une maison", len(d))]
    for libelle, garde in [
        ("charge CH_TOT > 0", d["ch"] > 0),
        *filtres,
        ("caractéristiques plausibles", d["plausible"].eq(True)),
    ]:
        d = d[garde.reindex(d.index)]
        etapes.append((libelle, len(d)))

    d = d.copy()
    d["taux"] = d["ch"] / d["valeur"]
    d["niveaux"] = np.where(d["nb_niveau"] >= 3, "3+", d["nb_niveau"].astype(int).astype(str))
    d["periode"] = pd.cut(d["annee_construction"], [-np.inf, 1918, 1945, 1974, 1999, np.inf],
                          labels=PERIODES).astype(str)
    d["alea"] = d["alea_argile"].fillna("Inconnu")
    d["log_surface"] = np.log(d["surface_sol_m2"])
    d["dep"] = d["code_departement_insee"].astype(str).str.zfill(2)
    d["annee"] = lire_annee(d["ANNEE"]).astype(str)
    d["peril"] = d["PERIL"].replace("", "(vide)")
    d["secheresse"] = d["PERIL"].str.contains("cheresse", case=False)
    d["annee_vente"] = pd.to_datetime(d["dvf_date_mutation"], errors="coerce").dt.year
    return d, etapes


# --------------------------------------------------------------------------- #
# Modalités                                                                    #
# --------------------------------------------------------------------------- #

def fusionner_rares(s: pd.Series, ordre: list[str] | None, mini: int) -> tuple[pd.Series, list]:
    """Fusionne les modalités de moins de `mini` observations.

    Variable ordonnée : la plus petite modalité rejoint sa voisine la moins
    fournie, jusqu'à ce que toutes atteignent le seuil ; une modalité hors de
    l'ordre (aléa « Inconnu ») trop rare est versée dans la plus fréquente.
    Variable nominale : les modalités rares forment « autre », versé dans la
    plus fréquente s'il reste lui-même sous le seuil.
    """
    s, fusions = s.astype(str), []
    if ordre is not None:
        hors = ~s.isin(ordre)
        n_hors = s[hors].value_counts()
        rares = n_hors.index[n_hors < mini].tolist()
        if rares and (~hors).any():
            cible = s[~hors].value_counts().index[0]
            s = s.where(~s.isin(rares), cible)
            fusions.append(f"{', '.join(rares)} -> {cible}")
        niveaux = [m for m in ordre if (s == m).any()]
        while len(niveaux) > 1:
            n = s.value_counts()
            petit = min(niveaux, key=lambda m: n[m])
            if n[petit] >= mini:
                break
            i = niveaux.index(petit)
            voisins = [niveaux[j] for j in (i - 1, i + 1) if 0 <= j < len(niveaux)]
            cible = min(voisins, key=lambda m: n[m])
            a, b = sorted([petit, cible], key=niveaux.index)
            neuf = f"{a.split('/')[0]}/{b.split('/')[-1]}"
            s = s.replace({a: neuf, b: neuf})
            niveaux[min(niveaux.index(a), niveaux.index(b))] = neuf
            niveaux.remove(b)
            fusions.append(f"{a} + {b}")
        return s, fusions
    n = s.value_counts()
    rares = n.index[n < mini].tolist()
    if rares:
        s = s.where(~s.isin(rares), "autre")
        fusions.append(f"{', '.join(rares)} -> autre")
        if (s == "autre").sum() < mini and s.nunique() > 1:
            s = s.where(s != "autre", s[s != "autre"].value_counts().index[0])
    return s, fusions


def categoriser(d: pd.DataFrame, facteurs: list[str], mini: int,
                refs: dict[str, str] | None = None,
                ordonnees: dict[str, list[str]] | None = None) -> tuple[pd.DataFrame, dict]:
    """Variables explicatives catégorielles, la référence en première modalité.

    `refs` impose les références d'un modèle précédent quand la modalité existe
    encore (éventuellement fusionnée) : sans cela, deux modèles ajustés sur des
    échantillons différents ne seraient pas comparables terme à terme.
    `ordonnees` ajoute des variables ordonnées à ORDONNEES.
    """
    ordres = {**ORDONNEES, **(ordonnees or {})}
    d, notes = d.copy(), {}
    for f in facteurs:
        s, fusions = fusionner_rares(d[f], ordres.get(f), mini)
        n = s.value_counts()
        ref = (refs or {}).get(f, REFERENCE.get(f))
        ref = ref.split("/")[0] if ref else None
        ref = next((m for m in n.index if ref and ref in m.split("/")), n.index[0]) \
            if ref else n.index[0]
        # Modalités dans l'ordre naturel (périodes, années), la référence en tête.
        rang = {m: i for i, m in enumerate(ordres.get(f, []))}
        autres = sorted((m for m in n.index if m != ref),
                        key=lambda m: (rang.get(m.split("/")[0], len(rang)), m))
        d[f] = pd.Categorical(s, categories=[ref] + autres)
        notes[f] = (ref, fusions)
    return d, notes


def table_modalites(nom: str, brut: pd.DataFrame, d: pd.DataFrame,
                    facteurs: list[str]) -> pd.DataFrame:
    """Modalité brute -> modalité du modèle, fusions comprises, et la référence.

    De quoi appliquer les coefficients à d'autres maisons
    (cout_secheresse_departements.py). `d` est `brut` passé par categoriser.
    """
    return pd.concat([pd.DataFrame({"modele": nom, "variable": f, "modalite": brut[f].astype(str),
                                    "modalite_modele": d[f].astype(str)})
                      .drop_duplicates()
                      .assign(reference=lambda t, f=f: t["modalite_modele"].eq(
                          d[f].cat.categories[0]))
                      for f in facteurs])


# --------------------------------------------------------------------------- #
# Ajustement                                                                   #
# --------------------------------------------------------------------------- #

def ajuster(d: pd.DataFrame, termes: list[str], offset: bool = True,
            grappe: str = "batiment_groupe_id"):
    """GLM Gamma lien log, erreurs-types groupées par `grappe` (la maison par défaut)."""
    formule = "ch ~ " + " + ".join(termes)
    groupes = pd.factorize(d[grappe])[0]
    famille = sm.families.Gamma(sm.families.links.Log())
    kw = {"offset": np.log(d["valeur"])} if offset else {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return smf.glm(formule, data=d, family=famille, **kw).fit(
            cov_type="cluster", cov_kwds={"groups": groupes})


def tableau(res, nom: str) -> pd.DataFrame:
    ic = res.conf_int()
    t = pd.DataFrame({"modele": nom, "terme": res.params.index,
                      "exp_coef": np.exp(res.params), "ic95_bas": np.exp(ic[0]),
                      "ic95_haut": np.exp(ic[1]), "p_valeur": res.pvalues,
                      "n": int(res.nobs)})
    t["terme"] = [re.sub(r"^(\w+)\[T\.(.+)\]$", r"\1 = \2", x) for x in t["terme"]]
    return t.reset_index(drop=True)


def tests_globaux(res, nom: str) -> pd.DataFrame:
    """Test de Wald joint de chaque variable, toutes modalités ensemble.

    Il répond à « la variable joue-t-elle ? » sans dépendre de la modalité de
    référence, et s'appuie sur la covariance groupée par maison, comme les
    intervalles. Les p-values sont aussi corrigées par Holm pour la
    multiplicité des tests.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        w = res.wald_test_terms(skip_single=False, scalar=True).table
    w = w.drop(index="Intercept", errors="ignore")
    t = pd.DataFrame({"modele": nom, "variable": w.index, "ddl": w["df_constraint"].astype(int),
                      "chi2": w["statistic"].astype(float), "p_valeur": w["pvalue"].astype(float)})
    t["p_holm"] = multipletests(t["p_valeur"], method="holm")[1]
    return t.reset_index(drop=True)


def qualite(res) -> dict[str, float]:
    return {"n": int(res.nobs), "parametres": int(res.df_model + 1),
            "dispersion": res.scale, "deviance": res.deviance,
            "deviance expliquee": 1 - res.deviance / res.null_deviance, "AIC": res.aic}


# --------------------------------------------------------------------------- #
# Rapport (agrégats uniquement)                                                #
# --------------------------------------------------------------------------- #

def _quantiles(s: pd.Series, fmt: str) -> str:
    return ", ".join(f"q{int(k * 100)} {fmt.format(v)}"
                     for k, v in s.quantile([.1, .25, .5, .75, .9]).items())


def decrire(d: pd.DataFrame, etapes: list[tuple[str, int]], doublons: int,
            valeur: str) -> None:
    print(f"\ndoublons de CLE retirés de la base : {doublons:,}")
    print("effectif après chaque filtre")
    for libelle, n in etapes:
        print(f"  {libelle:<40}: {n:>7,}")
    if d.empty:
        return
    print(f"\nvaleur V ({valeur}) : " + _quantiles(d["valeur"], "{:,.0f} €"))
    print("taux CH_TOT / V : " + _quantiles(d["taux"], "{:.2%}")
          + f" ; supérieur à 1 : {(d['taux'] > 1).sum():,}")
    if valeur == "dvf":
        ecart = (pd.to_numeric(d["annee"], errors="coerce") - d["annee_vente"]).dropna()
        if len(ecart):
            print(f"année de survenance - année de vente : médiane {ecart.median():+.0f}, "
                  f"q10 {ecart.quantile(.1):+.0f}, q90 {ecart.quantile(.9):+.0f}")
    print("\nsinistres et taux médian par péril")
    print(d.groupby("peril")["taux"].agg(n="size", taux_median="median")
          .sort_values("n", ascending=False)
          .to_string(formatters={"taux_median": "{:.2%}".format}))


def comparer(tabs: list[pd.DataFrame]) -> None:
    t = pd.concat(tabs)
    t = t[t["terme"] != "Intercept"]
    t["cellule"] = [f"{e:6.2f} [{b:5.2f}-{h:5.2f}]{'*' if p < .05 else ' '}"
                    for e, b, h, p in zip(t["exp_coef"], t["ic95_bas"], t["ic95_haut"],
                                          t["p_valeur"])]
    large = t.pivot(index="terme", columns="modele", values="cellule")
    ordre = list(dict.fromkeys(t["terme"]))
    large = large.reindex(ordre).fillna("")[[x["modele"].iloc[0] for x in tabs]]
    print("\nexp(coef) [IC95], * = p < 0,05 — effet multiplicatif sur le taux de dommage")
    with pd.option_context("display.max_rows", None, "display.width", 200):
        print(large.to_string())


# --------------------------------------------------------------------------- #

def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--valeur", choices=["reconstruction", "dvf"], default="reconstruction",
                   help="valeur V au dénominateur du taux (défaut reconstruction)")
    p.add_argument("--cout-m2", type=float, default=COUT_M2,
                   help=f"coût de construction en €/m² de plancher (défaut {COUT_M2})")
    p.add_argument("--sin", type=Path, help="chemin de SIN_EVT_NAT.csv")
    p.add_argument("--appariement", type=Path, default=APPARIEMENT)
    p.add_argument("--maisons", type=Path, default=MAISONS)
    p.add_argument("--min-modalite", type=int, default=MIN_MODALITE,
                   help=f"effectif minimal d'une modalité (défaut {MIN_MODALITE})")
    p.add_argument("--sortie", type=Path)
    a = p.parse_args()
    sortie = a.sortie or Path(str(SORTIE).format(valeur=a.valeur))

    for f, script in [(a.appariement, "apparier_sinistres_maisons.py"),
                      (a.maisons, "extract_maisons.py")]:
        if not f.exists():
            raise SystemExit(f"{f} introuvable : lancer d'abord {script}")

    chemin = trouver_base(a.sin)
    print(f"Base de sinistres : {chemin}")
    sin, doublons = lire_sinistres(chemin, COLONNES_SIN)
    app = pd.read_csv(a.appariement, dtype={"batiment_groupe_id": str})
    maisons = pd.read_csv(a.maisons, low_memory=False,
                          dtype={"batiment_groupe_id": str, "code_departement_insee": str})
    d, etapes = construire(sin, app, maisons, a.valeur, a.cout_m2)
    decrire(d, etapes, doublons, a.valeur)

    base = ["niveaux", "periode", "alea", "log_surface", "dep", "annee"]
    if a.valeur == "reconstruction":
        # V = emprise × niveaux × coût : avec log(emprise) en covariable, la
        # variante à log(V) libre serait quasi colinéaire. L'élasticité au
        # plancher s'identifie par l'emprise, à nombre de niveaux donné.
        base.remove("log_surface")
    modeles = [("tous périls", d, base + ["peril"]), ("sécheresse", d[d["secheresse"]], base)]
    tabs, tests, lignes_qualite, refs, modalites = [], [], {}, None, []
    for nom, donnees, termes in modeles:
        if len(donnees) < MIN_OBS_MODELE:
            print(f"\n[{nom}] {len(donnees)} sinistres : trop peu pour ajuster le modèle")
            continue
        facteurs = [t for t in termes if t != "log_surface"]
        brut = donnees
        donnees, notes = categoriser(donnees, facteurs, a.min_modalite, refs)
        modalites.append(table_modalites(nom, brut, donnees, facteurs))
        refs = refs or {f: r for f, (r, _) in notes.items()}
        # Une variable réduite à une modalité n'apporte rien et rend X singulière.
        termes = [t for t in termes if t == "log_surface" or donnees[t].nunique() > 1]
        res = ajuster(donnees, termes)
        libre = ajuster(donnees, termes + ["np.log(valeur)"], offset=False)
        tabs.append(tableau(res, nom))
        tests.append(tests_globaux(res, nom))
        q = qualite(res)
        b, (lo, hi) = libre.params["np.log(valeur)"], libre.conf_int().loc["np.log(valeur)"]
        q["elasticite cout/valeur"] = f"{b:.2f} [{lo:.2f}-{hi:.2f}]"
        lignes_qualite[nom] = q
        print(f"\n[{nom}] références : " + ", ".join(f"{f} = {r}" for f, (r, _) in notes.items()))
        fus = [f"{f} : {', '.join(x)}" for f, (_, x) in notes.items() if x]
        if fus:
            print("  modalités fusionnées — " + " ; ".join(fus))

    if not tabs:
        return 1
    comparer(tabs)
    qt = pd.DataFrame(lignes_qualite)
    print("\nqualité d'ajustement")
    print(qt.to_string(formatters=None, float_format=lambda v: f"{v:,.3f}"))
    tg = pd.concat(tests)
    print("\ntests globaux par variable (Wald joint, covariance groupée par maison)")
    for nom, t in tg.groupby("modele", sort=False):
        print(f"[{nom}]")
        print(t.drop(columns="modele").set_index("variable").to_string(
            formatters={"chi2": "{:.1f}".format, "p_valeur": "{:.2g}".format,
                        "p_holm": "{:.2g}".format}))
    sortie.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(tabs).to_csv(sortie, index=False)
    sortie_tests = sortie.with_name(sortie.stem.removesuffix("_coefs") + "_tests.csv")
    tg.to_csv(sortie_tests, index=False)
    sortie_modalites = sortie.with_name(sortie.stem.removesuffix("_coefs") + "_modalites.csv")
    pd.concat(modalites).sort_values(["modele", "variable", "modalite"]).to_csv(
        sortie_modalites, index=False)
    print(f"\ncoefficients -> {sortie}\ntests globaux -> {sortie_tests}"
          f"\nmodalités -> {sortie_modalites}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
