#!/usr/bin/env python3
"""Arrêtés CatNat par commune : extraction par type de péril.

`catnat.csv`, à la racine de l'espace de travail, n'a pas d'en-tête : une ligne
par (commune, épisode reconnu, arrêté), 16 champs séparés par « ; ». Ce script
en tire deux sous-ensembles, écrits avec en-tête dans `output/` :

    mouvements_de_terrain   type « Mouvements de terrain »
    secheresse_argiles      types « Sécheresse » et « Retrait-gonflement des argiles »

Le premier est le filtre littéral. Il ne contient PAS la subsidence de l'article
SwiGAN : ce sont des glissements, coulées, éboulements et effondrements, épisodes
de quelques jours (médiane 5 j), souvent reconnus avec une inondation ou une
tempête — la tempête du 25-29 décembre 1999 en fournit à elle seule les deux
tiers. Le retrait-gonflement des argiles consécutif à la sécheresse est rangé
sous « Sécheresse » (épisodes trimestriels, médiane 92 j, pics en 2003, 2018-2020
et 2022), et sous « Retrait-gonflement des argiles » pour quelques arrêtés de
2020-2021. C'est le second fichier qu'il faut confronter à l'éligibilité de
`eligibilite_catnat.py`.

DATES AVANCÉES D'UN JOUR
------------------------
Les arrêtés publiés à partir de 2022 portent une heure 01:00 ou 02:00. La date y
est le plus souvent juste, mais sur une partie des lignes toute la période est
avancée d'un jour : « 2020-12-31 -> 2021-12-30 » désigne l'année civile 2021, et
une affectation par année de début la range en 2020. Une période de sécheresse
étant alignée sur les mois, le décalage se lit sans ambiguïté — un début au
dernier jour d'un mois, une fin à la veille du dernier jour — et
`date_debut_corrigee` / `date_fin_corrigee` le corrigent. Les colonnes d'origine
restent telles quelles, et rien n'est corrigé sur les autres périls, dont les
dates ne sont pas alignées.

DOUBLONS
--------
Un même couple (commune, type, début, fin) revient parfois plusieurs fois, à
l'identique ou sous un second arrêté. Les sorties conservent toutes les lignes ;
un comptage de reconnaissances doit dédoublonner sur ces quatre colonnes.

Usage : python3 extraire_catnat.py [--catnat ../catnat.csv]
"""

import argparse
import csv
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
CATNAT_DEFAUT = ROOT.parent / "catnat.csv"

COLONNES = ["index", "code_geographique", "libelle_geographique", "epci", "libelle_epci",
            "departement", "libelle_departement", "region", "ept", "libelle_petr",
            "code_pnr", "libelle_pnr", "type_catastrophe", "date_debut", "date_fin",
            "date_arrete"]

SOUS_ENSEMBLES = {
    "mouvements_de_terrain": ["Mouvements de terrain"],
    "secheresse_argiles": ["Sécheresse", "Retrait-gonflement des argiles"],
}
CORRIGE = "secheresse_argiles"
CLE_EPISODE = ["code_geographique", "type_catastrophe", "date_debut", "date_fin"]


def charger(chemin: Path) -> pd.DataFrame:
    """Lecture brute, tout en texte, guillemets non interprétés.

    Des libellés d'EPCI et de PETR contiennent des guillemets : lus avec les
    options par défaut, pandas les retire, et modifie ainsi 1 107 lignes.
    """
    return pd.read_csv(chemin, sep=";", header=None, names=COLONNES, dtype=str,
                       quoting=csv.QUOTE_NONE, keep_default_na=False)


def corriger_dates(df: pd.DataFrame) -> pd.DataFrame:
    """Rend aux périodes de sécheresse avancées d'un jour leurs vraies bornes."""
    un_jour = pd.Timedelta(days=1)
    for col, avancee in (("date_debut", lambda j: j.dt.is_month_end),
                         ("date_fin", lambda j: (j + un_jour).dt.is_month_end)):
        t = pd.to_datetime(df[col], format="%Y-%m-%d %H:%M:%S.%f")
        jour = t.dt.normalize()
        decale = (t.dt.hour != 0) & avancee(jour)
        df[f"{col}_corrigee"] = (jour + decale * un_jour).dt.strftime("%Y-%m-%d")
        df[f"{col}_decalee"] = decale
    return df


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--catnat", type=Path, default=CATNAT_DEFAUT,
                    help="export CatNat sans en-tête (défaut : ../catnat.csv)")
    a = ap.parse_args()

    df = charger(a.catnat)
    print(f"{a.catnat.name} : {len(df):,} lignes, {df['code_geographique'].nunique():,} communes")

    for nom, types in SOUS_ENSEMBLES.items():
        sous = df[df["type_catastrophe"].isin(types)].copy()
        repetees = int(sous.duplicated(CLE_EPISODE).sum())
        if nom == CORRIGE:
            sous = corriger_dates(sous)
            n_debut = int(sous.pop("date_debut_decalee").sum())
            n_fin = int(sous.pop("date_fin_decalee").sum())
        sortie = ROOT / "output" / f"catnat_{nom}.csv"
        sous.to_csv(sortie, sep=";", index=False)

        print(f"\n{nom} : {len(sous):,} lignes, {sous['code_geographique'].nunique():,} communes "
              f"-> {sortie.relative_to(ROOT)}")
        print("  " + ", ".join(f"{t} {n:,}" for t, n in
                               sous["type_catastrophe"].value_counts().items()))
        print(f"  {repetees:,} lignes répètent un épisode (commune, type, début, fin) déjà présent")
        if nom == CORRIGE:
            print(f"  dates avancées d'un jour, corrigées : {n_debut:,} débuts, {n_fin:,} fins")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
