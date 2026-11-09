#!/usr/bin/env python3
"""Appariement géographique des sinistres SIN_EVT_NAT aux maisons individuelles
de la BDNB, en région Grand Est.

Chaque sinistre porte une position (LATITUDE, LONGITUDE en WGS84) ; chaque
groupe de bâtiments BDNB une emprise au sol (polygone `geom_groupe` de
`batiment_groupe`, en Lambert-93). Le sinistre est rattaché au bâtiment le plus
proche dans un rayon de --distance-max mètres, à distance nulle s'il tombe dans
l'emprise. On ne garde que les rattachements à une maison individuelle
(`usage_principal_bdnb_open = 'Résidentiel individuel'`, même périmètre que
extract_maisons.py).

POURQUOI APPARIER SUR TOUS LES BÂTIMENTS
----------------------------------------
Le plus proche voisin est cherché parmi tous les bâtiments, pas seulement les
maisons : un sinistre géocodé sur un immeuble collectif ou un commerce ne doit
pas être reporté sur la maison d'à côté. Il est rattaché à ce bâtiment puis
écarté. Seules les dépendances (garages, abris) sont ignorées : un point qui
tombe sur le garage d'une maison va à la maison voisine.

CONFIDENTIALITÉ
---------------
La base de sinistres est lue en place, depuis son chemin réseau, et n'est
jamais copiée. Une CLE répétée est un doublon : seule sa première ligne est
gardée (`lire_sinistres`, partagée avec glm_taux_dommage.py). La sortie
(`output/sinistres_maisons_grand_est.csv`, hors dépôt) ne contient que la clé
du sinistre, son numéro de ligne dans la source, son niveau de géocodage et le
rattachement BDNB : les autres variables se retrouvent par jointure sur CLE ou
`ligne_source`. Le rapport à l'écran ne donne que des effectifs agrégés, jamais
de ligne individuelle.

QUALITÉ DU RATTACHEMENT
-----------------------
- `appariement` vaut `dans_emprise` (point dans le polygone) ou `proximite`
  (point hors de toute emprise, maison la plus proche à `distance_m` mètres).
  Un géocodage à l'adresse place souvent le point sur la voie, devant le
  bâtiment : d'où la tolérance de --distance-max mètres (30 par défaut).
- Un géocodage à la rue, au code postal ou à la commune (STREET, POSTCODE,
  CITY) place le point au milieu de la voie ou au centre de la commune : la
  maison la plus proche est arbitraire, et tous les sinistres d'une commune
  tombent sur la même. Seuls EXACT et INTERPOLATED_STREET sont donc appariés
  par défaut ; `--niveaux TOUS` les reprend tous, pour comparaison.
- Les bâtiments sans usage BDNB (sans rattachement aux Fichiers Fonciers,
  surtout des annexes : emprise médiane 36 m²) comptent comme « autre
  bâtiment » : un point qui y tombe n'est pas reporté sur une maison.
- La région est délimitée par les contours communaux (geo.api.gouv.fr,
  téléchargés dans `cache/contours/` au premier lancement).

Usage :
    python3 apparier_sinistres_maisons.py
    python3 apparier_sinistres_maisons.py --distance-max 20 --dep 08 51
    python3 apparier_sinistres_maisons.py --niveaux TOUS
    python3 apparier_sinistres_maisons.py --sin "\\\\serveur\\partage\\SIN_EVT_NAT.csv"
"""

from __future__ import annotations

import argparse
import os
import sys
import urllib.request
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely

from sinistres_valeur import SIN_CANDIDATS, lire_montant, ouvrir

ROOT = Path(__file__).resolve().parent
BDNB = ROOT / "bdnb_grand_est"
SORTIE = ROOT / "output" / "sinistres_maisons_grand_est.csv"
CONTOURS = ROOT / "cache" / "contours" / "communes_44.geojson"
URL_CONTOURS = ("https://geo.api.gouv.fr/communes?codeRegion=44"
                "&fields=code,nom,codeDepartement&format=geojson&geometry=contour")

# Chemin de la base : variable d'environnement, sinon le partage réseau, sinon
# les emplacements déjà connus de sinistres_valeur.py.
SIN_DEFAUT = [Path(p) for p in [os.environ.get("SIN_EVT_NAT")] if p] + [
    Path(r"\\srv-data01\commun_datalab$\Chaire allianz\SIN_EVT_NAT.csv"), *SIN_CANDIDATS]

GRAND_EST = ["08", "10", "51", "52", "54", "55", "57", "67", "68", "88"]
WGS84, LAMBERT93 = "EPSG:4326", "EPSG:2154"

USAGE_MAISON = "Résidentiel individuel"
USAGE_IGNORE = "Dépendance"
SANS_USAGE = "(sans usage)"
DISTANCE_MAX = 30.0
NIVEAUX_FIABLES = ["EXACT", "INTERPOLATED_STREET"]

# Seules colonnes lues dans la base de sinistres.
COLONNES_SIN = ["CLE", "LATITUDE", "LONGITUDE", "GEOCODING_LEVEL", "PERIL"]
# Emprise large de la France métropolitaine : hors de ces bornes, la position
# est fausse (zéro, inversion latitude / longitude, valeur manquante).
LAT_MIN, LAT_MAX, LON_MIN, LON_MAX = 41.0, 51.5, -5.5, 10.0


# --------------------------------------------------------------------------- #
# Lecture                                                                      #
# --------------------------------------------------------------------------- #

def trouver_base(chemin: Path | None) -> Path:
    for c in [chemin] if chemin else SIN_DEFAUT:
        if c.exists():
            return c
    raise SystemExit("Base SIN_EVT_NAT introuvable. Préciser --sin CHEMIN ou la variable "
                     "d'environnement SIN_EVT_NAT.")


def lire_sinistres(chemin: Path, colonnes: list[str]) -> tuple[pd.DataFrame, int]:
    """Colonnes demandées de la base, en texte, doublons de CLE retirés.

    Une CLE répétée est un doublon : seule sa première ligne est gardée.
    `ligne_source` (1 = première ligne de données du fichier) repère chaque
    ligne et sert de clé de jointure entre les scripts. Renvoie aussi le nombre
    de doublons retirés.
    """
    encodage, sep = ouvrir(chemin)
    brut = pd.read_csv(chemin, sep=sep, encoding=encodage, dtype=str, keep_default_na=False,
                       usecols=lambda c: c.strip().upper() in colonnes)
    brut.columns = [c.strip().upper() for c in brut.columns]
    manquantes = [c for c in colonnes if c not in brut.columns]
    if manquantes:
        raise SystemExit(f"Colonnes absentes de {chemin.name} : {', '.join(manquantes)}")
    brut.insert(0, "ligne_source", np.arange(1, len(brut) + 1))
    doublon = brut["CLE"].duplicated(keep="first")
    return brut[~doublon], int(doublon.sum())


def charger_sinistres(chemin: Path) -> tuple[gpd.GeoDataFrame, dict[str, int]]:
    """Sinistres dédoublonnés à position valide, projetés en Lambert-93."""
    brut, doublons = lire_sinistres(chemin, COLONNES_SIN)
    lat, lon = lire_montant(brut["LATITUDE"]), lire_montant(brut["LONGITUDE"])
    valide = lat.between(LAT_MIN, LAT_MAX) & lon.between(LON_MIN, LON_MAX)
    stats = {"lignes": len(brut) + doublons, "doublons_retires": doublons,
             "position_invalide": int((~valide).sum())}

    sin = gpd.GeoDataFrame(brut.loc[valide].drop(columns=["LATITUDE", "LONGITUDE"]),
                           geometry=gpd.points_from_xy(lon[valide], lat[valide]), crs=WGS84)
    return sin.to_crs(LAMBERT93), stats


def contours_grand_est() -> gpd.GeoDataFrame:
    if not CONTOURS.exists():
        CONTOURS.parent.mkdir(parents=True, exist_ok=True)
        print(f"Téléchargement des contours communaux -> {CONTOURS.relative_to(ROOT)}")
        urllib.request.urlretrieve(URL_CONTOURS, CONTOURS)
    return gpd.read_file(CONTOURS)[["codeDepartement", "geometry"]].to_crs(LAMBERT93)


def charger_batiments(dep: str) -> gpd.GeoDataFrame:
    """Emprises des groupes de bâtiments d'un département, dépendances exclues."""
    csv = BDNB / f"dep_{dep}" / "csv"
    groupe = pd.read_csv(csv / "batiment_groupe.csv", sep=";", dtype=str,
                         usecols=["geom_groupe", "batiment_groupe_id", "code_departement_insee",
                                  "code_commune_insee", "libelle_commune_insee"])
    usage = pd.read_csv(csv / "batiment_groupe_synthese_propriete_usage.csv", sep=";",
                        dtype=str, usecols=["batiment_groupe_id", "usage_principal_bdnb_open"])
    bat = groupe.merge(usage, on="batiment_groupe_id", how="left")
    bat["usage_principal_bdnb_open"] = bat["usage_principal_bdnb_open"].fillna(SANS_USAGE)
    bat = bat[bat["usage_principal_bdnb_open"].ne(USAGE_IGNORE) & bat["geom_groupe"].notna()]
    geom = shapely.from_wkt(bat.pop("geom_groupe").to_numpy())
    return gpd.GeoDataFrame(bat, geometry=geom, crs=LAMBERT93)


# --------------------------------------------------------------------------- #
# Appariement                                                                  #
# --------------------------------------------------------------------------- #

def plus_proche(sin: gpd.GeoDataFrame, bat: gpd.GeoDataFrame, dmax: float) -> pd.DataFrame:
    """Bâtiment le plus proche de chaque sinistre à moins de dmax mètres."""
    x0, y0, x1, y1 = bat.total_bounds
    cand = sin.cx[x0 - dmax:x1 + dmax, y0 - dmax:y1 + dmax]
    if cand.empty:
        return pd.DataFrame()
    j = gpd.sjoin_nearest(cand[["geometry"]], bat, how="inner", max_distance=dmax,
                          distance_col="distance_m")
    return pd.DataFrame(j.drop(columns=["geometry", "index_right"]))


def meilleur_rattachement(cands: list[pd.DataFrame]) -> pd.DataFrame:
    """Un bâtiment par sinistre : le plus proche, tous départements confondus.

    Un point proche d'une limite départementale a un candidat de chaque côté ;
    à égalité de distance (emprises superposées), la maison l'emporte.
    """
    c = pd.concat([x for x in cands if not x.empty] or [pd.DataFrame(
        columns=["batiment_groupe_id", "code_departement_insee", "code_commune_insee",
                 "libelle_commune_insee", "usage_principal_bdnb_open", "distance_m"])])
    c["maison"] = c["usage_principal_bdnb_open"].eq(USAGE_MAISON)
    c = c.sort_values(["distance_m", "maison"], ascending=[True, False], kind="stable")
    return c[~c.index.duplicated(keep="first")]


# --------------------------------------------------------------------------- #
# Rapport (effectifs agrégés uniquement)                                       #
# --------------------------------------------------------------------------- #

def statut(sin: pd.DataFrame, best: pd.DataFrame) -> pd.Series:
    s = pd.Series("aucun_batiment", index=sin.index, name="statut")
    autre = best.index[~best["maison"]]
    maison = best[best["maison"]]
    s[autre] = "autre_usage"
    s[maison.index[maison["distance_m"] == 0]] = "maison_emprise"
    s[maison.index[maison["distance_m"] > 0]] = "maison_proximite"
    return s


def rapport(stats: dict[str, int], sin: gpd.GeoDataFrame, st: pd.Series,
            best: pd.DataFrame, dmax: float) -> None:
    n = stats["lignes"]
    print(f"\nlignes lues                 : {n:>9,}")
    print(f"  doublons de CLE retirés   : {stats['doublons_retires']:>9,}")
    print(f"  position invalide         : {stats['position_invalide']:>9,}")
    m = len(sin)
    print(f"  dans le périmètre traité  : {m:>9,}")
    if not m:
        return
    ordre = ["maison_emprise", "maison_proximite", "autre_usage", "aucun_batiment"]
    libelle = {"maison_emprise": "maison, point dans l'emprise",
               "maison_proximite": f"maison, à moins de {dmax:.0f} m",
               "autre_usage": "autre bâtiment (écarté)",
               "aucun_batiment": f"aucun bâtiment à {dmax:.0f} m (écarté)"}
    compte = st.value_counts()
    for k in ordre:
        v = int(compte.get(k, 0))
        print(f"    {libelle[k]:<34}: {v:>9,} ({v / m:6.1%})")

    autres = best.loc[st.index[st.eq("autre_usage")]]
    if len(autres):
        print("  usage des autres bâtiments touchés : " + ", ".join(
            f"{u} {v:,}" for u, v in autres["usage_principal_bdnb_open"].value_counts().items()))

    prox = best.loc[st.index[st.eq("maison_proximite")], "distance_m"]
    if len(prox):
        q = prox.quantile([.5, .9])
        print(f"  distance des rattachements à proximité : médiane {q[.5]:.1f} m, "
              f"p90 {q[.9]:.1f} m")

    for col in ["GEOCODING_LEVEL", "PERIL"]:
        tab = pd.crosstab(sin[col].replace("", "(vide)"), st, normalize="index")
        tab = tab.reindex(columns=ordre, fill_value=0)
        tab.insert(0, "n", sin[col].replace("", "(vide)").value_counts())
        tab = tab.sort_values("n", ascending=False).head(15)
        print(f"\npar {col} (part de chaque statut)")
        print(tab.to_string(formatters={k: "{:.1%}".format for k in ordre}))

    print("\nsinistres rattachés à une maison, par département")
    print(best.loc[best["maison"], "code_departement_insee"].value_counts().sort_index()
          .to_string())


# --------------------------------------------------------------------------- #

def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sin", type=Path, help="chemin de SIN_EVT_NAT.csv")
    p.add_argument("--distance-max", type=float, default=DISTANCE_MAX,
                   help=f"rayon de recherche en mètres (défaut {DISTANCE_MAX:.0f})")
    p.add_argument("--dep", nargs="+", default=GRAND_EST, metavar="CODE",
                   help="départements à traiter (défaut : tout le Grand Est)")
    p.add_argument("--niveaux", nargs="+", default=NIVEAUX_FIABLES, metavar="NIVEAU",
                   help=f"GEOCODING_LEVEL retenus, TOUS pour aucun filtre "
                        f"(défaut : {' '.join(NIVEAUX_FIABLES)})")
    p.add_argument("--sortie", type=Path, default=SORTIE)
    a = p.parse_args()

    chemin = trouver_base(a.sin)
    print(f"Base de sinistres : {chemin}")
    sin, stats = charger_sinistres(chemin)

    # Périmètre : sinistres situés dans une commune des départements traités.
    contours = contours_grand_est()
    contours = contours[contours["codeDepartement"].isin(a.dep)]
    dedans = gpd.sjoin(sin[["geometry"]], contours, how="inner", predicate="within")
    sin = sin.loc[sin.index.isin(dedans.index)]
    if a.niveaux != ["TOUS"]:
        sin = sin[sin["GEOCODING_LEVEL"].isin(a.niveaux)]

    cands = []
    for dep in a.dep:
        print(f"[dep {dep}] lecture des emprises BDNB", flush=True)
        cands.append(plus_proche(sin, charger_batiments(dep), a.distance_max))
    best = meilleur_rattachement(cands)
    st = statut(sin, best)

    garde = best[best["maison"]]
    sortie = sin.loc[garde.index, ["ligne_source", "CLE", "GEOCODING_LEVEL"]].join(
        garde[["batiment_groupe_id", "code_departement_insee", "code_commune_insee",
               "libelle_commune_insee", "distance_m"]])
    sortie.insert(3, "appariement", np.where(sortie["distance_m"] == 0,
                                             "dans_emprise", "proximite"))
    sortie["distance_m"] = sortie["distance_m"].round(1)
    a.sortie.parent.mkdir(parents=True, exist_ok=True)
    sortie.sort_values("ligne_source").to_csv(a.sortie, index=False)

    rapport(stats, sin, st, best, a.distance_max)
    print(f"\n{len(sortie):,} sinistres rattachés à une maison -> {a.sortie}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
