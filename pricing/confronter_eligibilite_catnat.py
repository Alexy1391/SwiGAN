#!/usr/bin/env python3
"""Éligibilité météorologique (seuils 1 et 2) sur le SWI observé, confrontée aux arrêtés.

`eligibilite_catnat.py` calcule les seuils 1 et 2 de la brochure Météo-France à
la maille. Ce script les rejoue sur le SWI uniforme OBSERVÉ, pour les
sécheresses 2018 à 2025 et toutes les communes du Grand Est, puis met le
résultat en regard des décisions publiées au Journal officiel
(`output/decisions_catnat_<année>.csv`, produits par `taux_succes_catnat.py`).

TROIS SOURCES, TROIS CHOIX
--------------------------
SWI. `SwiGAN/data_train.parquet` porte le `swi_u` mensuel du Grand Est de 1960
à 2024 (896 mailles). C'est bien le SWI UNIFORME de la brochure, pas le SWI de
SIM2 : sur 2022 il s'écarte de `MENS_SIM2_2022.csv` de 0,09 en médiane, ce
qu'explique le sol et la végétation tenus uniformes. 2025 vient du parquet
d'entrée de l'inférence (`Grand_Est_2022_2025.parquet`), identique au premier
sur 99,8 % des valeurs de 2021-2024 ; deux mailles y manquent.

Communes. Les contours communaux (geo.api.gouv.fr, découpage 2026) remplacent
les adresses BDNB de `correspondance_communes_mailles.py`, qui n'existent plus
que pour les Ardennes. Chaque contour est échantillonné tous les 100 m, plus ses
sommets, et chaque point va à la maille dont le centre est le plus proche dans
une projection équirectangulaire locale. Une maille couvrant une partie de la
commune suffit, comme le veut la brochure. La part de surface de chaque maille
est conservée.

Décisions. Une ligne par demande examinée : commune (nom et département),
période, annexe I ou II, et pour la plupart des arrêtés depuis 2022 la
motivation. Le nom est rapproché du code INSEE du découpage 2026. Une commune
fusionnée depuis la décision ne se rapproche pas, et elle est comptée à part.

CE QUE LA CONFRONTATION PEUT DIRE
---------------------------------
Le critère de la brochure est celui de la circulaire IOME2322937C du 29 avril
2024, appliqué aux sécheresses de 2024 et après. De 2018 à 2023 l'État appliquait
la circulaire INTE1911312C du 10 mai 2019, dont le critère météorologique
diffère. Sur ces années la comparaison mesure l'écart entre les deux règles, pas
la qualité du calcul. Seules 2024 et 2025 valident les seuils 1 et 2, et leurs
motivations disent lequel des trois seuils la commune a franchi.

Seules les communes qui ont déposé une demande figurent dans les arrêtés. Une
commune éligible qui n'a rien demandé n'y apparaît pas.

Usage :
    python3 confronter_eligibilite_catnat.py
    python3 confronter_eligibilite_catnat.py --annees 2018 2025 --csv sortie.csv
"""

import argparse
import json
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

from eligibilite_catnat import (FENETRE_DEFAUT, RANG_5_ANS, RANG_10_ANS, appliquer_seuils,
                                indicateur_annuel, rangs_glissants)

ROOT = Path(__file__).resolve().parent
SWIGAN = ROOT.parent / "SwiGAN"
SWI_1960_2024 = SWIGAN / "data_train.parquet"
SWI_2022_2025 = SWIGAN / "Outputs" / "rcp_8.5" / "Grand Est" / "Grand_Est_2022_2025.parquet"
CONTOURS = ROOT / "cache" / "contours" / "communes_44.geojson"
URL_CONTOURS = ("https://geo.api.gouv.fr/communes?codeRegion=44"
                "&fields=code,nom,codeDepartement&format=geojson&geometry=contour")
COMMUNES = ROOT / "communes-france-2026.parquet"
CORRESPONDANCE = ROOT / "output" / "correspondance_grand_est.csv"
SORTIE = ROOT / "output" / "eligibilite_c12_vs_decisions_grand_est.csv"

R_TERRE = 6371008.8
PAS_ECHANTILLON = 100.0
DEMI_DIAGONALE = 8000 * np.sqrt(2) / 2
PHENOMENE_SECHERESSE = r"cons.cutifs . la s.cheresse"


# --------------------------------------------------------------------------- #
# Correspondance (commune, maille) depuis les contours                         #
# --------------------------------------------------------------------------- #

def dans_anneaux(px: np.ndarray, py: np.ndarray, anneaux: list[np.ndarray]) -> np.ndarray:
    """Pair-impair sur tous les anneaux : les trous et les multipolygones en découlent."""
    dedans = np.zeros(len(px), dtype=bool)
    for a in anneaux:
        x1, y1 = a[:-1, 0], a[:-1, 1]
        x2, y2 = a[1:, 0], a[1:, 1]
        for i in range(0, len(x1), 256):
            s = slice(i, i + 256)
            croise = (y1[s] > py[:, None]) != (y2[s] > py[:, None])
            with np.errstate(divide="ignore", invalid="ignore"):
                xi = (x2[s] - x1[s]) * (py[:, None] - y1[s]) / (y2[s] - y1[s]) + x1[s]
            dedans ^= (croise & (px[:, None] < xi)).sum(axis=1) % 2 == 1
    return dedans


def correspondance_contours(contours: Path, grille: pd.DataFrame) -> pd.DataFrame:
    """Mailles couvrant chaque commune, et la part de surface de chacune."""
    feats = json.loads(contours.read_text())["features"]
    g_lat, g_lon = grille["lat"].values, grille["lon"].values
    lignes = []
    for f in feats:
        geom = f["geometry"]
        polys = [geom["coordinates"]] if geom["type"] == "Polygon" else geom["coordinates"]
        anneaux_ll = [np.asarray(r, dtype=float) for poly in polys for r in poly]
        tous = np.vstack(anneaux_ll)
        lat0 = np.radians(tous[:, 1].mean())
        lon0 = tous[:, 0].mean()

        def metrique(lon, lat):
            return (np.radians(lon - lon0) * np.cos(lat0) * R_TERRE,
                    np.radians(lat - tous[:, 1].mean()) * R_TERRE)

        anneaux = [np.column_stack(metrique(r[:, 0], r[:, 1])) for r in anneaux_ll]
        sommets = np.vstack(anneaux)
        xmin, ymin = sommets.min(axis=0)
        xmax, ymax = sommets.max(axis=0)
        gx, gy = np.meshgrid(np.arange(xmin + PAS_ECHANTILLON / 2, xmax, PAS_ECHANTILLON),
                             np.arange(ymin + PAS_ECHANTILLON / 2, ymax, PAS_ECHANTILLON))
        gx, gy = gx.ravel(), gy.ravel()
        dedans = dans_anneaux(gx, gy, anneaux) if len(gx) else np.zeros(0, dtype=bool)
        # Les points intérieurs pèsent la surface ; les sommets n'ajoutent que des
        # mailles touchées par une frange plus fine que le pas.
        px = np.concatenate([gx[dedans], sommets[:, 0]])
        py = np.concatenate([gy[dedans], sommets[:, 1]])
        poids = np.concatenate([np.ones(int(dedans.sum())), np.zeros(len(sommets))])

        cx, cy = metrique(g_lon, g_lat)
        proches = ((cx > xmin - 12000) & (cx < xmax + 12000)
                   & (cy > ymin - 12000) & (cy < ymax + 12000))
        if not proches.any():
            continue
        cx, cy, mailles = cx[proches], cy[proches], grille["maille"].values[proches]
        d2 = (px[:, None] - cx) ** 2 + (py[:, None] - cy) ** 2
        j = d2.argmin(axis=1)
        sur_grille = np.sqrt(d2[np.arange(len(j)), j]) <= DEMI_DIAGONALE
        df = pd.DataFrame({"maille": mailles[j[sur_grille]], "poids": poids[sur_grille]})
        par = df.groupby("maille")["poids"].sum()
        total = poids.sum()
        for m, w in par.items():
            lignes.append((f["properties"]["code"], int(m),
                           w / total if total else np.nan))
    return pd.DataFrame(lignes, columns=["code_commune_insee", "maille", "part_surface"])


# --------------------------------------------------------------------------- #
# SWI observé 1960-2025 et seuils 1-2                                          #
# --------------------------------------------------------------------------- #

def charger_swi_observe() -> tuple[pd.DataFrame, pd.DataFrame]:
    """SWI uniforme mensuel par maille, et la grille (maille, lat, lon)."""
    a = pd.read_parquet(SWI_1960_2024, columns=["maille", "year", "month", "x", "y",
                                                "lat", "lon", "swi_u"])
    a["maille"] = a["maille"].astype(int)
    a["swi"] = a["swi_u"].astype(str).str.replace(",", ".", regex=False).astype(float)
    grille = a[["maille", "lat", "lon"]].drop_duplicates("maille").reset_index(drop=True)
    cles = a[["maille", "x", "y"]].drop_duplicates()

    b = pd.read_parquet(SWI_2022_2025, columns=["year", "month", "x", "y", "swi_u"])
    b = b[b["year"] > a["year"].max()].merge(cles, on=["x", "y"], how="inner")
    b["swi"] = b["swi_u"].astype(float)
    swi = pd.concat([a[["maille", "year", "month", "swi"]],
                     b[["maille", "year", "month", "swi"]]], ignore_index=True)
    return swi.rename(columns={"year": "annee", "month": "mois"}), grille


def seuils_par_maille(swi: pd.DataFrame) -> pd.DataFrame:
    ann = indicateur_annuel(swi, ["maille"])
    cube = ann.pivot(index="maille", columns="annee", values="swi_min")
    rangs = rangs_glissants(cube.values.astype(np.float32), FENETRE_DEFAUT)
    res = appliquer_seuils(rangs, RANG_10_ANS, RANG_5_ANS)
    idx = pd.MultiIndex.from_product([cube.index, cube.columns], names=["maille", "annee"])
    return pd.DataFrame({"rang": rangs.ravel(), "seuil1": res["seuil1"].ravel(),
                         "seuil2": res["seuil2"].ravel()}, index=idx).reset_index()


# --------------------------------------------------------------------------- #
# Décisions                                                                    #
# --------------------------------------------------------------------------- #

def cle_nom(nom: str) -> str:
    """« Férée (La) » et « La Férée » donnent la même clé."""
    m = re.fullmatch(r"(.*) \((L'|La|Le|Les|L’)\)", nom.strip())
    if m:
        nom = f"{m.group(2)} {m.group(1)}"
    nom = unicodedata.normalize("NFKD", nom).encode("ascii", "ignore").decode().lower()
    nom = re.sub(r"\bsaint\b", "st", re.sub(r"\bsainte\b", "ste", nom))
    return re.sub(r"[^a-z0-9]", "", nom)


def seuil_officiel(motivation: str) -> str | None:
    """Le seuil que la motivation dit franchi, circulaire IOME2322937C seulement."""
    m = re.search(r"crit[èe]re m[ée]t[ée]orologique est satisfait\s*:([^.]*)", motivation)
    if not m:
        return None
    t = m.group(1)
    if "limitrophe" in t:
        return "3"
    if "succession anormale" in t:
        return "2"
    if "période de retour" in t or "periode de retour" in t:
        return "1"
    return None


def meteo_officiel(ligne: pd.Series) -> str | None:
    """« satisfait », « pas satisfait », ou None quand l'arrêté ne le dit pas."""
    c = ligne["critere_meteorologique"]
    if isinstance(c, str) and c:
        return c
    mot = ligne["motivation"] if isinstance(ligne["motivation"], str) else ""
    if re.search(r"crit[èe]res m[ée]t[ée]orologiques et g[ée]otechniques.*sont r[ée]unis", mot):
        return "satisfait"
    if re.search(r"crit[èe]re m[ée]t[ée]orologique.*n'est pas satisfait", mot):
        return "pas satisfait"
    # Une reconnaissance suppose le critère météorologique rempli ; un rejet sans
    # motivation peut venir de l'un ou l'autre critère.
    return "satisfait" if ligne["reconnue"] == "True" else None


def charger_decisions(communes: pd.DataFrame, annees: tuple[int, int]) -> pd.DataFrame:
    fr = [pd.read_csv(f, sep=";", dtype=str)
          for f in sorted((ROOT / "output").glob("decisions_catnat_20*.csv"))]
    d = pd.concat(fr, ignore_index=True)
    d = d[d["phenomene"].str.contains(PHENOMENE_SECHERESSE, na=False, regex=True)].copy()
    d["annee"] = pd.to_datetime(d["debut"], dayfirst=True, errors="coerce").dt.year
    d = d[d["departement"].isin(communes["dep_nom"].unique()) & d["annee"].between(*annees)]
    d = d.drop_duplicates(["texte", "departement", "commune", "debut", "fin", "reconnue"])

    d["cle"] = d["commune"].map(cle_nom)
    ref = communes.assign(cle=communes["nom_standard"].map(cle_nom))
    d = d.merge(ref[["dep_nom", "cle", "code_insee"]], left_on=["departement", "cle"],
                right_on=["dep_nom", "cle"], how="left")
    d["meteo"] = d.apply(meteo_officiel, axis=1)
    d["seuil_officiel"] = d["motivation"].fillna("").map(seuil_officiel)
    return d


def decisions_par_commune(d: pd.DataFrame) -> pd.DataFrame:
    """Une ligne par (commune, année) : plusieurs périodes ou un réexamen se cumulent."""
    d = d.dropna(subset=["code_insee"])

    def meteo(s):
        if (s == "satisfait").any():
            return "satisfait"
        return "pas satisfait" if (s == "pas satisfait").any() else "inconnu"

    return (d.groupby(["code_insee", "annee"])
             .agg(reconnue=("reconnue", lambda s: (s == "True").any()),
                  meteo_officiel=("meteo", meteo),
                  seuil_officiel=("seuil_officiel",
                                  lambda s: ",".join(sorted(set(s.dropna()))) or None),
                  n_lignes=("reconnue", "size"))
             .reset_index()
             .rename(columns={"code_insee": "code_commune_insee"}))


# --------------------------------------------------------------------------- #
# Affichage                                                                    #
# --------------------------------------------------------------------------- #

def afficher(tab: pd.DataFrame, n_communes: int) -> None:
    print("\n" + "=" * 96)
    print("SEUILS 1-2 SUR LE SWI OBSERVÉ, CONTRE LE CRITÈRE MÉTÉOROLOGIQUE DES ARRÊTÉS")
    print("=" * 96)
    print(f"  {n_communes:,} communes du Grand Est. « demandes » : communes ayant une décision ;")
    print("  « off. » : critère météorologique officiel connu. Rappel accord / sensibilité / "
          "précision\n  calculés sur les demandes au critère connu.\n")
    print(f"  {'année':>5} {'circ.':>5} {'élig.':>6} {'S1':>6} {'S2':>6} {'demandes':>9} "
          f"{'off.':>5} {'off.sat':>7} {'calc&off':>8} {'accord':>7} {'sensib.':>7} "
          f"{'précis.':>7}")
    for an, g in tab.groupby("annee"):
        dem = g[g["demande"]]
        k = dem[dem["meteo_officiel"] != "inconnu"]
        off = k["meteo_officiel"] == "satisfait"
        calc = k["eligible_c12"]
        circ = "2024" if an >= 2024 else "2019"
        sens = (calc & off).sum() / off.sum() if off.sum() else np.nan
        prec = (calc & off).sum() / calc.sum() if calc.sum() else np.nan
        print(f"  {an:>5} {circ:>5} {g['eligible_c12'].sum():>6,} {g['seuil1'].sum():>6,} "
              f"{g['seuil2'].sum():>6,} {len(dem):>9,} {len(k):>5,} {off.sum():>7,} "
              f"{(calc & off).sum():>8,} {(calc == off).mean():>7.1%} {sens:>7.1%} "
              f"{prec:>7.1%}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--annees", type=int, nargs=2, default=[2018, 2025],
                   metavar=("DEBUT", "FIN"))
    p.add_argument("--recalculer-correspondance", action="store_true",
                   help=f"reconstruire {CORRESPONDANCE.name} depuis les contours")
    p.add_argument("--csv", type=Path, default=SORTIE)
    a = p.parse_args()

    swi, grille = charger_swi_observe()
    print(f"SWI observé : {swi['maille'].nunique()} mailles, "
          f"{swi['annee'].min()}-{swi['annee'].max()} ; "
          f"{swi.loc[swi['annee'] == swi['annee'].max(), 'maille'].nunique()} mailles "
          f"en {swi['annee'].max()}")

    if a.recalculer_correspondance or not CORRESPONDANCE.exists():
        if not CONTOURS.exists():
            raise SystemExit(f"{CONTOURS} introuvable. Le télécharger :\n"
                             f"  curl -o '{CONTOURS}' '{URL_CONTOURS}'")
        corr = correspondance_contours(CONTOURS, grille)
        corr.to_csv(CORRESPONDANCE, index=False)
        print(f"correspondance écrite dans {CORRESPONDANCE}")
    corr = pd.read_csv(CORRESPONDANCE, dtype={"code_commune_insee": str})
    par = corr.groupby("code_commune_insee").size()
    print(f"correspondance : {len(par):,} communes, {corr['maille'].nunique()} mailles ; "
          f"mailles par commune médiane {par.median():.0f}, max {par.max()}")

    mailles = seuils_par_maille(swi)
    mailles = mailles[mailles["annee"].between(*a.annees)]
    com = (corr.merge(mailles, on="maille")
               .assign(part_anormale=lambda df: df["part_surface"]
                       * (df["seuil1"] | df["seuil2"]))
               .groupby(["code_commune_insee", "annee"])
               .agg(seuil1=("seuil1", "any"), seuil2=("seuil2", "any"),
                    rang_min=("rang", "min"), part_anormale=("part_anormale", "sum"),
                    n_mailles=("maille", "size"))
               .reset_index())
    com["eligible_c12"] = com["seuil1"] | com["seuil2"]

    communes = pd.read_parquet(COMMUNES, columns=["code_insee", "nom_standard", "dep_code",
                                                  "dep_nom", "reg_code"])
    communes = communes[communes["reg_code"] == "44"]
    d = charger_decisions(communes, tuple(a.annees))
    perdues = d[d["code_insee"].isna()]
    print(f"décisions sécheresse Grand Est {a.annees[0]}-{a.annees[1]} : {len(d):,} lignes ; "
          f"{len(perdues)} sans code INSEE 2026 (communes fusionnées depuis), écartées")
    dec = decisions_par_commune(d)

    tab = (com.merge(communes.rename(columns={"code_insee": "code_commune_insee",
                                              "nom_standard": "commune"})
                     [["code_commune_insee", "commune", "dep_code"]],
                     on="code_commune_insee", how="left")
              .merge(dec, on=["code_commune_insee", "annee"], how="left"))
    tab["demande"] = tab["n_lignes"].notna()
    tab["meteo_officiel"] = tab["meteo_officiel"].fillna("inconnu")
    tab["reconnue"] = tab["reconnue"].astype("boolean")
    sans_maille = set(dec["code_commune_insee"]) - set(com["code_commune_insee"])
    if sans_maille:
        print(f"  {len(sans_maille)} communes ayant une décision sans maille : "
              f"{sorted(sans_maille)[:5]}")

    afficher(tab, com["code_commune_insee"].nunique())

    # 2024-2025 : la motivation dit quel seuil est franchi.
    rec = tab[tab["annee"] >= 2024]
    rec = rec[rec["demande"]]
    if len(rec):
        print("\n  Circulaire 2024 : seuil calculé contre seuil officiel, demandes seulement")
        calc = np.select([rec["seuil1"], rec["seuil2"]], ["S1", "S2 seul"], "aucun")
        off = rec["seuil_officiel"].fillna(
            rec["meteo_officiel"].map({"pas satisfait": "non", "satisfait": "?"})).fillna("?")
        print(pd.crosstab([rec["annee"], pd.Series(calc, index=rec.index, name="calculé")],
                          off.rename("officiel")).to_string().replace("\n", "\n    "))

    cols = ["code_commune_insee", "commune", "dep_code", "annee", "eligible_c12", "seuil1",
            "seuil2", "rang_min", "n_mailles", "part_anormale", "demande", "reconnue",
            "meteo_officiel", "seuil_officiel"]
    tab = tab[cols].sort_values(["annee", "code_commune_insee"])
    tab.to_csv(a.csv, index=False)
    print(f"\nécrit dans {a.csv} ({len(tab):,} lignes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
