#!/usr/bin/env python3
"""Prix au m² des maisons par commune, complété par les communes voisines.

Les ventes sont celles de `prix_m2_departements.py` (une maison seule,
éventuellement avec dépendances), restreintes aux maisons et écrêtées au
p1–p99 du prix au m² de leur DÉPARTEMENT : un écrêtage par commune n'aurait
pas de sens avec trois ventes.

RÈGLE D'AFFECTATION
-------------------
  commune avec au moins 9 ventes   -> sa propre moyenne
  commune avec moins de 9 ventes   -> moyenne pondérée de la commune et de
                                      son voisinage (ci-dessous)
  Moselle, Bas-Rhin, Haut-Rhin     -> prix médian départemental relevé sur
                                      immobilier.notaires.fr (absents de DVF :
                                      Livre foncier)
  Mayotte                          -> aucune valeur (absente de DVF)

Champ : communes des 101 départements au 1er janvier 2026 (contours Etalab),
Paris, Lyon et Marseille découpées en arrondissements ; les collectivités
d'outre-mer sont hors champ. Les codes DVF de communes fusionnées depuis sont
rattachés à la commune actuelle par la table des mouvements de l'Insee.

VOISINAGE
---------
Les voisines sont les communes limitrophes (contours qui se touchent, même en
un point). Si la commune et ses limitrophes totalisent moins de 9 ventes, le
voisinage s'étend aux limitrophes des limitrophes (rang 2), et ainsi de suite.
Les îles sans commune limitrophe prennent les communes les plus proches,
centre à centre, jusqu'à 9 ventes. Le prix du voisinage est la moyenne de
toutes ses ventes : chaque commune y pèse son nombre de ventes.

PONDÉRATION
-----------
On travaille sur y = log(prix au m²), dont l'écart est une erreur relative.
La moyenne de la commune et celle du voisinage estiment la vraie moyenne de la
commune avec les erreurs quadratiques

    MSE_commune   = s2_w(n) / n
    MSE_voisinage = tau2 + bruit du voisinage

s2_w(n) est la variance intra-commune, estimée par classe de taille : la
dispersion passe d'un écart-type d'environ 54 % sous 5 ventes à 39 % au-delà
de 100 (marchés plus homogènes en ville). tau2 est l'écart quadratique moyen
entre la vraie moyenne d'une commune et celle de son voisinage, lu sur
l'ensemble des communes en retirant le bruit des deux moyennes :

    tau2 = moyenne[ (ybar_i - ybar_V)^2 - s2_w(n_i)/n_i - bruit_V ]

Il est estimé pour chaque rang : un voisinage plus large ressemble moins à la
commune. Les deux moyennes sont pondérées par l'inverse de leur erreur :

    prix = Z * moyenne commune + (1 - Z) * moyenne voisinage
    Z    = MSE_voisinage / (MSE_commune + MSE_voisinage)

C'est le facteur de crédibilité de Bühlmann, Z ~ n / (n + k) avec
k = s2_w / MSE_voisinage. Une commune sans vente prend le voisinage seul.

LE SEUIL DE 9 VENTES
--------------------
C'est le nombre de ventes à partir duquel la moyenne d'une commune bat, seule,
celle de sa voisine la plus proche (environ 18 % d'erreur de part et d'autre).
Le diagnostic imprimé refait ce calcul avec le voisinage limitrophe.

Usage : python3 prix_m2_communes.py [--seuil 9]
"""

import argparse
import urllib.request
from pathlib import Path

import duckdb
import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from scipy import sparse
from sklearn.neighbors import BallTree

from prix_m2_departements import OUT, ROOT, mutations

CACHE = ROOT / "cache"
URL_CONTOURS = ("https://etalab-datasets.geo.data.gouv.fr/contours-administratifs/"
                "2026/geojson/communes-100m.geojson.gz")
URL_MOUVEMENTS = "https://www.insee.fr/fr/statistiques/fichier/8740222/v_mvt_commune_2026.csv"

# Prix médian des maisons au m², relevé à la main sur immobilier.notaires.fr.
PRIX_ALSACE_MOSELLE = {"57": 1660, "67": 2180, "68": 2140}
SEUIL = 9
RANG_MAX = 5
CLASSES_TAILLE = [1, 5, 10, 20, 50, 100, np.inf]  # classes de s2_w(n)
PLM = {"75056", "69123", "13055"}  # remplacées par leurs arrondissements
COM = ("975", "977", "978", "984", "986", "987", "988", "989")  # collectivités d'outre-mer


def telecharger(url: str, nom: str) -> Path:
    chemin = CACHE / nom
    if not chemin.exists():
        chemin.parent.mkdir(parents=True, exist_ok=True)
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=300) as r:
            chemin.write_bytes(r.read())
    return chemin


def referentiel() -> gpd.GeoDataFrame:
    """Communes 2026 des départements, Paris, Lyon et Marseille en arrondissements."""
    chemin = telecharger(URL_CONTOURS, "contours/communes-2026-100m.geojson.gz")
    g = gpd.read_file("/vsigzip/" + chemin.as_posix())
    g = g[~g["code"].isin(PLM) & ~g["code"].str.startswith(COM)].reset_index(drop=True)
    g = g.rename(columns={"code": "code_commune_insee", "nom": "nom_commune",
                          "departement": "code_departement"})
    return g[["code_commune_insee", "nom_commune", "code_departement", "geometry"]]


def table_passage(codes: pd.Series, actuels: set) -> dict:
    """Code DVF -> code actuel, en suivant les fusions et changements de code."""
    mvt = pd.read_csv(telecharger(URL_MOUVEMENTS, "v_mvt_commune_2026.csv"), dtype=str)
    mvt = mvt[mvt["TYPECOM_AV"].eq("COM") & mvt["TYPECOM_AP"].isin(["COM", "ARM"])]
    # Pour chaque ancien code, son dernier événement fait foi.
    dernier = (mvt.sort_values("DATE_EFF")
                  .drop_duplicates("COM_AV", keep="last")
                  .set_index("COM_AV")["COM_AP"].to_dict())
    passage = {}
    for code in codes.unique():
        c = code
        for _ in range(10):
            if c in actuels or c not in dernier or dernier[c] == c:
                break
            c = dernier[c]
        passage[code] = c if c in actuels else None
    return passage


def ventes_maisons(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    mutations(con)
    return con.execute("""
        WITH m AS (SELECT * FROM ventes WHERE type_local = 'Maison'),
        bornes AS (
            SELECT dep, quantile_cont(prix_m2, 0.01) AS p01,
                        quantile_cont(prix_m2, 0.99) AS p99
            FROM m GROUP BY dep
        )
        SELECT insee, prix_m2 FROM m JOIN bornes USING (dep)
        WHERE prix_m2 BETWEEN p01 AND p99
    """).df()


def variance_intra(st: pd.DataFrame):
    """s2_w(n) : variance intra-commune poolée par classe de taille."""
    multi = st[st["n"] >= 2]
    classe = pd.cut(multi["n"], CLASSES_TAILLE, right=False)
    poids = multi["n"] - 1
    s2 = ((poids * multi["var_log"]).groupby(classe, observed=False).sum()
          / poids.groupby(classe, observed=False).sum()).values
    return lambda n: s2[np.searchsorted(CLASSES_TAILLE, np.maximum(n, 1), side="right") - 1]


def voisinages(ref: gpd.GeoDataFrame, rang_max: int) -> list:
    """Matrices « au plus r pas de limitrophe en limitrophe », r = 1..rang_max."""
    geoms = ref.geometry.values
    i, j = shapely.STRtree(geoms).query(geoms, predicate="intersects")
    garde = i != j
    a = sparse.csr_matrix((np.ones(garde.sum()), (i[garde], j[garde])),
                          shape=(len(ref), len(ref)))
    rangs = [a]
    for _ in range(rang_max - 1):
        m = ((rangs[-1] + rangs[-1] @ a) > 0).astype(float).tolil()
        m.setdiag(0)
        rangs.append(m.tocsr())
    return rangs


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--seuil", type=int, default=SEUIL,
                   help=f"ventes en dessous desquelles on complète par le voisinage ({SEUIL})")
    args = p.parse_args()
    seuil = args.seuil

    ref = referentiel()
    v = ventes_maisons(duckdb.connect())
    passage = table_passage(v["insee"], set(ref["code_commune_insee"]))
    v["code_commune_insee"] = v["insee"].map(passage)
    perdues = v["code_commune_insee"].isna()
    print(f"{len(v):,} ventes de maisons ; {int(perdues.sum()):,} hors champ ou sans commune "
          f"actuelle ({', '.join(sorted(v.loc[perdues, 'insee'].unique())[:5])}…) écartées")
    v = v[~perdues]
    v["log"] = np.log(v["prix_m2"])

    st = (v.groupby("code_commune_insee")
            .agg(n=("prix_m2", "size"), prix_m2_moyen_commune=("prix_m2", "mean"),
                 prix_m2_median_commune=("prix_m2", "median"),
                 moy_log=("log", "mean"), var_log=("log", "var"))
            .reset_index())
    st = ref.drop(columns="geometry").merge(st, on="code_commune_insee", how="left")
    st["n"] = st["n"].fillna(0).astype(int)

    n = st["n"].values.astype(float)
    s2_w = variance_intra(st)
    bruit_c = np.where(n > 0, s2_w(n) / np.maximum(n, 1), np.inf)   # MSE_commune
    # Sommes par commune, que les matrices de voisinage agrègent.
    somme = np.nan_to_num(n * st["prix_m2_moyen_commune"].values)
    somme_log = np.nan_to_num(n * st["moy_log"].values)
    somme_var = np.where(n > 0, n * s2_w(n), 0.0)

    am = st["code_departement"].isin(PRIX_ALSACE_MOSELLE).values
    deps_dvf = st.loc[st["n"] > 0, "code_departement"].unique()
    absent = ~st["code_departement"].isin(deps_dvf).values & ~am
    a_completer = (n < seuil) & ~am & ~absent

    nb = len(st)
    rang = np.zeros(nb, dtype=int)                      # 0 : pas de voisinage retenu
    n_v, n_com_v, prix_v, mse_v = (np.full(nb, np.nan) for _ in range(4))
    tau2 = {}
    for r, m in enumerate(voisinages(ref, RANG_MAX), start=1):
        N = m @ n
        with np.errstate(divide="ignore", invalid="ignore"):
            y_v = (m @ somme_log) / N
            bruit_v = (m @ somme_var) / N ** 2
        ok = (n >= 1) & (N >= 1)
        ecart = (st["moy_log"].values - y_v) ** 2 - bruit_c - bruit_v
        tau2[r] = max(float(np.mean(ecart[ok])), 0.0)
        prend = a_completer & (rang == 0) & (n + N >= seuil)
        rang[prend] = r
        n_v[prend] = N[prend]
        n_com_v[prend] = (m @ np.ones(nb))[prend]
        prix_v[prend] = (m @ somme)[prend] / N[prend]
        mse_v[prend] = tau2[r] + bruit_v[prend]

    # Îles sans limitrophe (et tout reste) : communes les plus proches.
    reste = np.flatnonzero(a_completer & (rang == 0))
    if len(reste):
        pts = ref.geometry.representative_point()
        latlon = np.radians(np.c_[pts.y.values, pts.x.values])
        avec = np.flatnonzero(n > 0)
        _, idx = BallTree(latlon[avec], metric="haversine").query(latlon[reste], k=60)
        for i, voisines in zip(reste, avec[idx]):
            voisines = voisines[voisines != i]
            cumul = n[i] + np.cumsum(n[voisines])
            sel = voisines[: np.searchsorted(cumul, seuil) + 1]
            N = n[sel].sum()
            rang[i] = -1
            n_v[i], n_com_v[i] = N, len(sel)
            prix_v[i] = somme[sel].sum() / N
            mse_v[i] = tau2[RANG_MAX] + somme_var[sel].sum() / N ** 2

    poids = np.where(a_completer, mse_v / (bruit_c + mse_v), np.nan)
    poids[a_completer & (n == 0)] = 0.0
    prix = np.where(n >= seuil, st["prix_m2_moyen_commune"].values, np.nan)
    prix = np.where(a_completer, poids * np.nan_to_num(st["prix_m2_moyen_commune"].values)
                    + (1 - poids) * prix_v, prix)
    prix[am] = st.loc[am, "code_departement"].map(PRIX_ALSACE_MOSELLE).values

    st["source_prix"] = np.select([am, absent, a_completer], ["notaires_departement", "aucune",
                                  "commune_et_voisinage"], "commune")
    st["voisinage"] = pd.Series(rang).map(
        lambda r: "" if r == 0 else "plus_proches" if r < 0 else
        "limitrophes" if r == 1 else f"rang_{r}").values
    st["n_communes_voisinage"] = n_com_v
    st["n_ventes_voisinage"] = n_v
    st["prix_m2_voisinage"] = prix_v
    st["poids_commune"] = poids
    st["prix_m2_maison"] = prix

    res = st.drop(columns=["moy_log", "var_log"]).rename(columns={"n": "n_ventes"})
    for c in ["prix_m2_moyen_commune", "prix_m2_median_commune", "prix_m2_voisinage",
              "prix_m2_maison"]:
        res[c] = res[c].round()
    res["poids_commune"] = res["poids_commune"].round(3)
    for c in ["n_communes_voisinage", "n_ventes_voisinage"]:
        res[c] = res[c].astype("Int64")
    OUT.mkdir(exist_ok=True)
    res.to_csv(OUT / "prix_m2_maisons_communes.csv", index=False)

    print(f"\n{len(res):,} communes, seuil = {seuil} ventes")
    print(res["source_prix"].value_counts().to_string())
    print("\nÉcart commune / voisinage (racine de tau2) : "
          + ", ".join(f"rang {r} {np.sqrt(t):.1%}" for r, t in tau2.items()))
    print(f"\nCommunes à compléter dont commune + limitrophes < {seuil} ventes :")
    vc = res.loc[a_completer, "voisinage"].value_counts()
    print(vc.drop("limitrophes", errors="ignore").to_string())
    hors = res[a_completer & (rang != 1)].sort_values("n_ventes_voisinage")
    print(hors[["code_commune_insee", "nom_commune", "n_ventes", "voisinage",
                "n_communes_voisinage", "n_ventes_voisinage"]].head(25).to_string(index=False))
    print("\nPoids moyen de la commune selon son nombre de ventes :")
    print(res[a_completer].groupby("n_ventes")["poids_commune"].mean().round(2).to_string())
    lim = a_completer & (rang == 1)
    n_croise = np.arange(1, 1001)
    n_croise = n_croise[s2_w(n_croise) / n_croise <= np.median(mse_v[lim])][0]
    print(f"\nDiagnostic : avec le voisinage limitrophe (erreur médiane "
          f"{np.sqrt(np.median(mse_v[lim])):.1%}), la commune seule le bat dès {n_croise} ventes")
    print(f"-> {OUT / 'prix_m2_maisons_communes.csv'}")


if __name__ == "__main__":
    main()
