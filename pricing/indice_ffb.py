#!/usr/bin/env python3
"""Indice FFB du coût de la construction (ICC FFB), relevé sur le site de la FFB.

Indice trimestriel, base 1 au 1er janvier 1941, calculé par la Fédération
française du bâtiment sur le prix de revient d'un immeuble de rapport courant à
Paris, hors terrain. Son objet initial est l'indexation des polices
d'assurance ; CCR s'en sert pour exprimer en euros constants le coût moyen des
sinistres ONRN.

La page n'affiche que deux années : l'année cherchée et la précédente (sans
recherche, les deux dernières). La recherche par année est une publication
ASP.NET : un POST qui renvoie les champs cachés de la page (__VIEWSTATE...)
avec le champ TB_Annee. On l'interroge donc une année sur deux, de la plus
récente à --debut, avec une pause entre deux requêtes.

Sortie : cache/indice_ffb.csv (annee, trimestre, indice), que les autres
scripts relisent sans retélécharger tant qu'il couvre les années demandées
(--rafraichir pour forcer). indice_annuel() en tire la moyenne des quatre
trimestres de chaque année complète.

Usage : python3 indice_ffb.py [--debut 1995] [--rafraichir]
"""

import argparse
import html
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "cache" / "indice_ffb.csv"
URL = ("https://www.outils.ffbatiment.fr/federation-francaise-du-batiment/le-batiment-et-vous/"
       "en_chiffres/indices-index/Chiffres_Index_FFB_Construction.html")
CONTENU = "ctl00$ctl00$ContentPlaceHolderGlobal$ContentPlaceHolderContenu$"
DEBUT = 1995
PAUSE = 1.0  # secondes entre deux requêtes


def _page(annee: int | None = None, etat: dict[str, str] | None = None) -> str:
    """La page telle que publiée, ou après une recherche de l'année `annee`."""
    data = None
    if annee is not None:
        champs = {**(etat or {}), "__EVENTTARGET": CONTENU + "But_Recherche",
                  "__EVENTARGUMENT": "", CONTENU + "TB_Annee": str(annee)}
        data = urllib.parse.urlencode(champs).encode()
    req = urllib.request.Request(URL, data=data, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read().decode("utf-8")


def _etat(page: str) -> dict[str, str]:
    """Champs cachés ASP.NET à renvoyer avec la recherche."""
    return {n: html.unescape(v) for n, v in re.findall(
        r'<input[^>]*name="(__VIEWSTATE\w*|__EVENTVALIDATION)"[^>]*value="([^"]*)"', page)}


def _valeurs(page: str) -> list[tuple[int, int, float]]:
    """(année, trimestre, indice) des blocs « Année » affichés par la page."""
    annees = dict(re.findall(r'LB_IndAnnee(\d)">(\d{4})<', page))
    lignes = re.findall(r'LV_IndFFB_Annee(\d)_ctrl\d+_LB_IndTrim">(\d)<.*?LB_IndIccFFB">([^<]*)<',
                        page, flags=re.S)
    return [(int(annees[bloc]), int(t), float(re.sub(r"\s", "", v).replace(",", ".")))
            for bloc, t, v in lignes]


def telecharger(debut: int = DEBUT) -> pd.DataFrame:
    """Toutes les valeurs trimestrielles depuis `debut`, lues sur le site."""
    page = _page()
    valeurs = _valeurs(page)
    if not valeurs:
        raise SystemExit(f"{URL} : aucune valeur lue, la page a changé")
    annee = min(a for a, _, _ in valeurs) - 1
    while annee >= debut:
        time.sleep(PAUSE)
        page = _page(annee, _etat(page))
        lues = _valeurs(page)
        if not lues:
            raise SystemExit(f"recherche de l'année {annee} : aucune valeur renvoyée")
        valeurs += lues
        annee = min(a for a, _, _ in lues) - 1
    d = pd.DataFrame(valeurs, columns=["annee", "trimestre", "indice"]).drop_duplicates()
    if d.duplicated(["annee", "trimestre"]).any():
        raise SystemExit("valeurs contradictoires pour un même trimestre")
    return d.sort_values(["annee", "trimestre"], ignore_index=True)


def indice_ffb(debut: int = DEBUT, rafraichir: bool = False) -> pd.DataFrame:
    """Indice trimestriel : le cache s'il couvre `debut`, sinon le site (et le cache réécrit)."""
    if CACHE.exists() and not rafraichir:
        d = pd.read_csv(CACHE)
        if d["annee"].min() <= debut:
            return d
    d = telecharger(min(debut, DEBUT))
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    d.to_csv(CACHE, index=False)
    return d


def indice_annuel(d: pd.DataFrame) -> pd.Series:
    """Moyenne des quatre trimestres, pour les seules années complètes."""
    g = d.groupby("annee")["indice"]
    return g.mean()[g.size() == 4]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--debut", type=int, default=DEBUT,
                   help=f"première année à relever (défaut {DEBUT})")
    p.add_argument("--rafraichir", action="store_true", help="retélécharger même si le cache existe")
    a = p.parse_args()
    d = indice_ffb(a.debut, a.rafraichir)
    an = indice_annuel(d)
    print(f"ICC FFB : {len(d)} trimestres, {d['annee'].min()} T{d['trimestre'].iloc[0]} - "
          f"{d['annee'].max()} T{d['trimestre'].iloc[-1]} -> {CACHE}")
    print("moyennes annuelles (années complètes), et variation sur un an :")
    print(pd.DataFrame({"indice": an.round(1), "variation": an.pct_change().map(
        lambda v: "" if pd.isna(v) else f"{v:+.1%}")}).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
