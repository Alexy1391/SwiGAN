#!/usr/bin/env python3
"""Correspondance (commune, maille SAFRAN), reconstruite depuis les adresses BDNB.

`eligibilite_catnat.py` calcule le critère météorologique à la MAILLE. Le régime
CatNat statue à la COMMUNE, et la brochure donne la règle de passage :

    « il suffit que le critère soit rempli pour une maille couvrant une partie
      de la commune pour qu'il soit considéré comme rempli pour toute la
      commune »

Il faut donc savoir quelles mailles couvrent chaque commune. La réponse exacte
demande les contours communaux (ADMIN-EXPRESS de l'IGN, ou la table de
correspondance de Météo-France) ; le dépôt n'en porte aucun. Ce script en
construit une APPROXIMATION avec ce qu'il a : les points d'adresse de la BDNB,
qui portent à la fois un code INSEE et une géométrie.

CE QUE L'APPROXIMATION VAUT, ET DANS QUEL SENS ELLE SE TROMPE
--------------------------------------------------------------
Les adresses couvrent la partie BÂTIE d'une commune, pas son territoire. Une
commune dont le finage déborde sur une maille voisine sans y avoir d'adresse
perd cette maille. L'erreur est donc à sens unique : elle ne peut que RETIRER
des mailles, jamais en ajouter. Comme la règle de passage est permissive — une
maille suffit — l'éligibilité qui en découle est une borne inférieure, dans le
même sens que l'absence du seuil 3 dans `eligibilite_catnat.py`.

L'ordre de grandeur joue en faveur de l'approximation : la maille SAFRAN fait
8 km de côté, la commune française médiane environ 2 km. La plupart des communes
tiennent dans une ou deux mailles, et le bâti suffit à les désigner. Sur les
Ardennes, 447 communes sur 449 sont couvertes, avec 1 à 4 mailles chacune.

LES DEUX PROJECTIONS
--------------------
Elles diffèrent, et c'est le seul point technique du script.

  - la BDNB est en Lambert-93 (EPSG:2154), ellipsoïde GRS80 ;
  - la grille SAFRAN est en Lambert II étendu (x ~ 7e5, y ~ 2,4e6 dans
    `swi_region.parquet`), pas en Lambert-93 : les deux jeux de coordonnées ne
    sont PAS comparables et les superposer donnerait n'importe quoi.

Le passage se fait par les latitudes/longitudes, que `data_with_swi_region_hist.
parquet` porte déjà pour chaque maille — ce qui évite d'avoir à inverser
Lambert II et son changement de datum NTF -> RGF93. Reste l'inverse de
Lambert-93, une conique conforme sécante standard, écrite ici sans dépendance :
RGF93 et WGS84 se confondent à quelques centimètres, aucun changement de datum
n'est requis. L'inverse est vérifiée sur son origine, qui doit retomber
exactement sur 46,5° N 3,0° E.

L'affectation est ensuite la maille dont le CENTRE est le plus proche. Sur une
grille régulière, c'est la maille qui contient le point, aux frontières près.
Un point à plus de 5 657 m (la demi-diagonale d'une maille de 8 km) du centre le
plus proche est hors grille : `--distance-max` l'écarte plutôt que de le
rattacher à une maille qui ne le couvre pas.

Usage :
    python3 correspondance_communes_mailles.py --dep 08 --csv correspondance_08.csv
    python3 eligibilite_catnat.py --correspondance correspondance_08.csv --csv elig.csv
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
SWIGAN = ROOT.parent / "SwiGAN"
GRILLE_LATLON = SWIGAN / "Grand Est" / "data_with_swi_region_hist.parquet"

# Demi-diagonale d'une maille SAFRAN de 8 km : au-delà, le point n'est dans
# aucune maille de la grille.
DEMI_DIAGONALE = 8000 * np.sqrt(2) / 2


# --------------------------------------------------------------------------- #
# Lambert-93 inverse (EPSG:2154, GRS80) — conique conforme sécante             #
# --------------------------------------------------------------------------- #

_A = 6378137.0                     # demi-grand axe GRS80
_F = 1 / 298.257222101             # aplatissement GRS80
_E = np.sqrt(2 * _F - _F * _F)     # première excentricité
_LAT0, _LON0, _LAT1, _LAT2 = np.radians([46.5, 3.0, 44.0, 49.0])
_X0, _Y0 = 700000.0, 6600000.0     # origine de la projection


def _m(phi):
    return np.cos(phi) / np.sqrt(1 - _E**2 * np.sin(phi) ** 2)


def _t(phi):
    return np.tan(np.pi / 4 - phi / 2) / ((1 - _E * np.sin(phi)) / (1 + _E * np.sin(phi))) ** (
        _E / 2
    )


_N = (np.log(_m(_LAT1)) - np.log(_m(_LAT2))) / (np.log(_t(_LAT1)) - np.log(_t(_LAT2)))
_F_C = _m(_LAT1) / (_N * _t(_LAT1) ** _N)
_RHO0 = _A * _F_C * _t(_LAT0) ** _N


def lambert93_vers_latlon(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(x, y) Lambert-93 -> (latitude, longitude) en degrés, RGF93 ~ WGS84."""
    xp, yp = x - _X0, _RHO0 - (y - _Y0)
    rho = np.sign(_N) * np.hypot(xp, yp)
    theta = np.arctan2(xp, yp)
    t = (rho / (_A * _F_C)) ** (1 / _N)
    lon = theta / _N + _LON0
    lat = np.pi / 2 - 2 * np.arctan(t)
    for _ in range(12):  # convergence de l'itération sur la latitude isométrique
        lat = np.pi / 2 - 2 * np.arctan(
            t * ((1 - _E * np.sin(lat)) / (1 + _E * np.sin(lat))) ** (_E / 2)
        )
    return np.degrees(lat), np.degrees(lon)


def _verifier_projection() -> None:
    """L'origine de Lambert-93 doit retomber sur 46,5° N 3,0° E."""
    lat, lon = lambert93_vers_latlon(np.array([_X0]), np.array([_Y0]))
    if abs(lat[0] - 46.5) > 1e-6 or abs(lon[0] - 3.0) > 1e-6:
        raise SystemExit(f"inverse Lambert-93 fausse : origine -> {lat[0]}, {lon[0]}")


# --------------------------------------------------------------------------- #
# Affectation à la maille la plus proche                                       #
# --------------------------------------------------------------------------- #

def charger_grille(chemin: Path) -> pd.DataFrame:
    """Centres des mailles SAFRAN, en latitude/longitude."""
    g = pd.read_parquet(chemin, columns=["maille", "lat", "lon"]).drop_duplicates("maille")
    g["maille"] = g["maille"].astype(int)
    return g.reset_index(drop=True)


def charger_adresses(dep: str) -> pd.DataFrame:
    """Points d'adresse BDNB d'un département, avec leur commune."""
    chemin = ROOT / f"dep_{dep}" / "csv" / "adresse.csv"
    if not chemin.exists():
        raise SystemExit(f"{chemin} introuvable : seul l'export brut d'un département "
                         "porte les géométries.")
    adr = pd.read_csv(chemin, sep=";", usecols=["WKT", "code_commune_insee"],
                      dtype={"code_commune_insee": str})
    xy = adr["WKT"].str.extract(r"POINT \(([-\d.]+) ([-\d.]+)\)")
    adr["x93"] = pd.to_numeric(xy[0], errors="coerce")
    adr["y93"] = pd.to_numeric(xy[1], errors="coerce")
    illisibles = int(adr["x93"].isna().sum())
    if illisibles:
        print(f"  {illisibles:,} adresses sans POINT lisible, écartées")
        adr = adr.dropna(subset=["x93", "y93"])
    adr["code_commune_insee"] = adr["code_commune_insee"].str.zfill(5)
    return adr


def affecter(adr: pd.DataFrame, grille: pd.DataFrame,
             distance_max: float) -> tuple[pd.DataFrame, np.ndarray]:
    """Maille dont le centre est le plus proche, et distance en mètres.

    Projection équirectangulaire locale autour du centre de la zone : sur
    l'emprise d'un département, à ces latitudes, elle est métrique à mieux que
    le pour-cent, très en deçà de la maille de 8 km qu'il s'agit de désigner.
    """
    lat, lon = lambert93_vers_latlon(adr["x93"].values, adr["y93"].values)
    lat_c = np.radians(np.mean(lat))
    px = np.radians(lon) * np.cos(lat_c) * _A
    py = np.radians(lat) * _A
    gx = np.radians(grille["lon"].values) * np.cos(lat_c) * _A
    gy = np.radians(grille["lat"].values) * _A

    proche = np.empty(len(px), dtype=np.int64)
    distance = np.empty(len(px), dtype=np.float64)
    for i in range(0, len(px), 20000):  # par tranches : la matrice complète est inutile
        d2 = (px[i:i + 20000, None] - gx) ** 2 + (py[i:i + 20000, None] - gy) ** 2
        j = d2.argmin(axis=1)
        proche[i:i + 20000] = j
        distance[i:i + 20000] = np.sqrt(d2[np.arange(len(j)), j])

    out = adr[["code_commune_insee"]].copy()
    out["maille"] = grille["maille"].values[proche]
    out["distance_m"] = distance
    hors = distance > distance_max
    if hors.any():
        print(f"  {int(hors.sum()):,} adresses à plus de {distance_max:,.0f} m du centre le "
              f"plus proche, hors grille SAFRAN, écartées")
    return out[~hors], distance


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dep", nargs="+", default=["08"],
                   help="départements à traiter (défaut : 08, le seul export brut conservé)")
    p.add_argument("--grille", type=Path, default=GRILLE_LATLON,
                   help="parquet portant maille/lat/lon (défaut : data_with_swi_region_hist)")
    p.add_argument("--distance-max", type=float, default=DEMI_DIAGONALE,
                   help=f"rejet au-delà de cette distance en m (défaut {DEMI_DIAGONALE:.0f}, "
                        "la demi-diagonale d'une maille)")
    p.add_argument("--min-adresses", type=int, default=1,
                   help="ignorer un couple (commune, maille) sous ce nombre d'adresses ; "
                        "1 par défaut, conformément à la règle « une maille suffit »")
    p.add_argument("--csv", type=Path, help="fichier de sortie")
    a = p.parse_args()

    _verifier_projection()
    grille = charger_grille(a.grille)
    print(f"grille SAFRAN : {len(grille)} mailles, "
          f"lat {grille['lat'].min():.2f}-{grille['lat'].max():.2f}, "
          f"lon {grille['lon'].min():.2f}-{grille['lon'].max():.2f}")

    morceaux = []
    for dep in a.dep:
        print(f"\ndépartement {dep}")
        adr = charger_adresses(dep)
        print(f"  {len(adr):,} adresses, {adr['code_commune_insee'].nunique():,} communes")
        affecte, _ = affecter(adr, grille, a.distance_max)
        morceaux.append(affecte)

    tout = pd.concat(morceaux, ignore_index=True)
    corr = (tout.groupby(["code_commune_insee", "maille"], as_index=False)
                .agg(n_adresses=("distance_m", "size"),
                     distance_moyenne_m=("distance_m", "mean")))
    corr = corr[corr["n_adresses"] >= a.min_adresses]
    corr["part_adresses"] = corr["n_adresses"] / corr.groupby("code_commune_insee")[
        "n_adresses"].transform("sum")
    corr = corr.sort_values(["code_commune_insee", "n_adresses"],
                            ascending=[True, False]).reset_index(drop=True)

    par_commune = corr.groupby("code_commune_insee")["maille"].size()
    perdues = set(tout["code_commune_insee"]) - set(corr["code_commune_insee"])
    print("\n" + "=" * 78)
    print("CORRESPONDANCE (commune, maille)")
    print("=" * 78)
    print(f"  {corr['code_commune_insee'].nunique():,} communes, "
          f"{corr['maille'].nunique()} mailles, {len(corr):,} couples")
    print(f"  mailles par commune : médiane {par_commune.median():.0f}, "
          f"max {par_commune.max()} ; "
          f"{(par_commune == 1).sum():,} communes sur une seule maille")
    if perdues:
        print(f"  {len(perdues)} communes sans aucune maille retenue : {sorted(perdues)[:5]}")
    print("\n  Rappel : le bâti approxime le territoire. L'erreur ne peut que retirer des")
    print("  mailles, donc l'éligibilité qui en découle est une borne inférieure.")

    if a.csv:
        corr.to_csv(a.csv, index=False)
        print(f"\nécrit dans {a.csv}")
    else:
        print("\n  Pour l'écrire : --csv correspondance.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
