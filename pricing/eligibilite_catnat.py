#!/usr/bin/env python3
"""Section 6.1 : du SWI à l'éligibilité CatNat — seuils 1 et 2.

Le maillon qui manquait entre SwiGAN et `cout_subsidence_futur.py`. La section
6.1 de l'article ne définit aucun critère : elle rejoue celui du régime CatNat,
en note 11, avec le SWI généré par SwiGAN en entrée. Ce script fait la même
chose, et n'invente rien de plus.

    « The first step consists of aggregating SWI values for each SAFRAN pixel
      from monthly to yearly values by considering the driest month. Then,
      return periods of these yearly SWI values are used to identify abnormal
      pixels for that year. Finally, the pixels within the commune of interest,
      as well as the characteristics of surrounding communes, are analyzed to
      determine eligibility. »                                    (section 6.1)

LE CRITÈRE, TEL QUE LA BROCHURE MÉTÉO-FRANCE L'ÉCRIT
----------------------------------------------------
L'indicateur est le SWI « uniforme » de SAFRAN-ISBA-MODCOU, grille de 8 km, sol
et végétation tenus uniformes sur toute la France. L'indicateur mensuel est la
moyenne des SWI quotidiens ; l'indicateur annuel d'une maille est le PLUS FAIBLE
des douze indicateurs mensuels — le mois le plus sec.

La période de retour est un RANG dans une fenêtre glissante de 30 ans : l'année
étudiée et les 29 précédentes. 2024 se compare donc à 1995-2024, et la référence
dérive avec le climat au lieu d'être figée sur une normale.

    rang 1 à 3 sur 30   ->   période de retour >= 10 ans
    rang 1 à 6 sur 30   ->   période de retour >=  5 ans

Trois seuils alternatifs, examinés tour à tour ; il suffit que l'un soit atteint.

    seuil 1  sécheresse annuelle   : période de retour >= 10 ans (rang 1-3)
    seuil 2  succession anormale   : période de retour >= 5 ans l'année étudiée
                                     ET dans au moins 2 des 4 années précédentes,
                                     sur la MÊME maille (rangs 1-6)
    seuil 3  communes limitrophes  : la commune à >= 5 ans, et voisine d'au moins
                                     une commune satisfaisant le seuil 1 ou 2

CE QUI EST IMPLÉMENTÉ ICI, ET CE QUI NE L'EST PAS
--------------------------------------------------
Les seuils 1 et 2 le sont : ils sont locaux à la maille et ne demandent que le
SWI. Le seuil 3 ne l'est pas : il demande un graphe d'adjacence des communes,
donc les contours communaux, que le dépôt ne porte pas. C'est aussi lui que la
figure 12 tient à part sous le nom de « potentially eligible », les communes
« not officially eligible [...] but surrounded by eligible communes ». Les
comptages produits ici sont donc une BORNE INFÉRIEURE de l'éligibilité : le
seuil 3 ne peut qu'en ajouter.

Le passage pixel -> commune est délibérément permissif dans la brochure : « il
suffit que le critère soit rempli pour une maille couvrant une partie de la
commune pour qu'il soit considéré comme rempli pour toute la commune ». Une
commune est donc éligible dès qu'UNE de ses mailles l'est. Ce passage demande
une correspondance (commune, maille) que ce script prend en entrée
(`--correspondance`) et que `correspondance_communes_mailles.py` construit pour
le département 08. Sans elle, la sortie reste à la maille.

Le filtre argile ne se pose pas ici. La commission interministérielle croise le
critère météorologique avec les zones d'argile du BRGM, mais c'est le maillon
suivant : c'est le comptage x_c de la section 6.2, déjà dans
`cout_subsidence_futur.py`.

LA FENÊTRE MIXTE, ET LE PIÈGE QU'ELLE TEND
-------------------------------------------
Une trajectoire 2025-2050 ne fait que 26 ans : sa fenêtre de 30 ans déborde
forcément sur l'observé. C'est ce que le régime ferait du vrai SWI, et c'est le
défaut ici (`--reference mixte`) : l'historique de `swi_region.parquet` est
préfixé à chaque trajectoire, et 2025 se compare bien à 1996-2025.

Mais le rang est une comparaison, et une comparaison entre deux populations qui
ne sortent pas de la même source. Un biais du générateur — même constant — se
lit alors comme un signal climatique : un SWI généré systématiquement plus sec
que l'observé classe presque toute année future au rang 1. `utils/bias_correction.py`
et `utils/climatology_anchor.py`, côté SwiGAN, mesurent précisément ce décalage
et concluent qu'il est réel. Le script affiche donc le diagnostic sur les années
de recouvrement, et refuse de le passer sous silence.

`--reference generee` range chaque année générée parmi les seules années
générées de sa propre trajectoire. Le rang devient invariant à toute
transformation monotone du SWI, donc insensible au biais — mais la référence
n'est plus le climat observé, c'est le climat du modèle. La fenêtre est alors
bornée par la longueur de la trajectoire (`--fenetre`), et les rangs seuils sont
mis à l'échelle : W/10 et W/5, soit 3 et 6 pour W = 30.

SORTIE
------
Avec `--correspondance`, exactement la table qu'attend `cout_subsidence_futur.py` :

    code_commune_insee,annee,trajectoire,eligible

Sans elle, la même chose à la maille, avec le détail des seuils :

    maille,annee,trajectoire,seuil1,seuil2,eligible

Usage :
    python3 eligibilite_catnat.py                       # historique 1960-2024
    python3 eligibilite_catnat.py --csv elig_mailles.csv
    python3 eligibilite_catnat.py --correspondance corr_08.csv --csv elig_08.csv
    python3 eligibilite_catnat.py \
        --swi-genere "../SwiGAN/Outputs/inference_grand_est_rcp85.npy" \
        --grille "../SwiGAN/Outputs/rcp_8.5/Grand Est/Grand_Est_2022_2025.parquet" \
        --correspondance corr_08.csv --csv elig_rcp85.csv
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
SWIGAN = ROOT.parent / "SwiGAN"
SWI_HIST_DEFAUT = SWIGAN / "Grand Est" / "swi_region.parquet"

# La fenêtre de la brochure, et les deux rangs qui y valent 10 et 5 ans.
FENETRE_DEFAUT = 30
RANG_10_ANS = 3
RANG_5_ANS = 6
# Seuil 2 : période de retour >= 5 ans dans au moins 2 des 4 années précédentes.
ANTECEDENTS = 4
ANTECEDENTS_REQUIS = 2


# --------------------------------------------------------------------------- #
# Lecture du SWI                                                              #
# --------------------------------------------------------------------------- #

def _swi_numerique(s: pd.Series) -> pd.Series:
    """`swi_u` est stocké en texte à virgule décimale dans tous les parquets.

    Le test porte sur le dtype et non sur `object` : pandas 3 rend les colonnes
    texte d'un parquet en `str` adossé à Arrow, jamais en `object`.
    """
    if pd.api.types.is_numeric_dtype(s):
        return s.astype(float)
    return s.astype(str).str.replace(",", ".", regex=False).astype(float)


def charger_swi_historique(chemin: Path) -> pd.DataFrame:
    """SWI mensuel observé, une ligne par (maille, année, mois)."""
    df = pd.read_parquet(chemin, columns=["maille", "year", "month", "swi_u", "x", "y"])
    df["swi"] = _swi_numerique(df["swi_u"])
    df["maille"] = df["maille"].astype(int)
    return df.rename(columns={"year": "annee", "month": "mois"})[
        ["maille", "annee", "mois", "swi", "x", "y"]
    ]


def charger_swi_genere(chemin: Path, grille: Path | None) -> pd.DataFrame:
    """SWI mensuel généré, une ligne par (trajectoire, maille, année, mois).

    Deux formats. Un tableau long (parquet/CSV) portant déjà `maille` ou
    `x`/`y`, `annee`/`year`, `mois`/`month`, un SWI et éventuellement
    `trajectoire`. Ou le `.npy` que produit `swigan/inference.py`, de forme
    (trajectoires, pas de temps, 1, hauteur, largeur) : il ne porte aucune
    coordonnée, et `--grille` doit alors fournir le parquet d'entrée de
    l'inférence, dont on relit la disposition des pixels et le calendrier.
    """
    if chemin.suffix.lower() != ".npy":
        return _charger_swi_genere_long(chemin)
    if grille is None:
        raise SystemExit(
            f"{chemin.name} est une grille sans coordonnées : --grille est requis.\n"
            "  Passer le parquet d'entrée de l'inférence, p. ex.\n"
            "  '../SwiGAN/Outputs/rcp_8.5/Grand Est/Grand_Est_2022_2025.parquet'"
        )
    return _charger_swi_genere_npy(chemin, grille)


def _charger_swi_genere_long(chemin: Path) -> pd.DataFrame:
    lire = pd.read_parquet if chemin.suffix.lower() in (".parquet", ".pq") else pd.read_csv
    brut = lire(chemin)
    dispo = {c.lower(): c for c in brut.columns}
    alias = {
        "annee": ["annee", "année", "year", "an"],
        "mois": ["mois", "month"],
        "swi": ["swi", "swi_u", "swi_uniforme", "valeur"],
        "maille": ["maille", "pixel", "cell", "id_maille"],
        "trajectoire": ["trajectoire", "trajectory", "traj", "membre", "member", "simulation"],
    }
    ren = {}
    for cible, noms in alias.items():
        trouve = next((dispo[n] for n in noms if n in dispo), None)
        if trouve is not None:
            ren[trouve] = cible
    df = brut.rename(columns=ren)

    if "maille" not in df.columns:
        if not {"x", "y"}.issubset(df.columns):
            raise SystemExit(f"{chemin} : ni `maille` ni (`x`, `y`) ; colonnes lues "
                             f"{list(brut.columns)}")
        cles = charger_swi_historique(SWI_HIST_DEFAUT)[["maille", "x", "y"]].drop_duplicates()
        df = df.merge(cles, on=["x", "y"], how="inner")
    manquantes = [c for c in ("annee", "mois", "swi") if c not in df.columns]
    if manquantes:
        raise SystemExit(f"{chemin} : colonnes introuvables {manquantes} ; colonnes lues "
                         f"{list(brut.columns)}")
    if "trajectoire" not in df.columns:
        df["trajectoire"] = 0
    df["swi"] = _swi_numerique(df["swi"])
    return df[["trajectoire", "maille", "annee", "mois", "swi"]]


def _charger_swi_genere_npy(chemin: Path, grille: Path) -> pd.DataFrame:
    """Rendre au `.npy` d'inférence ses mailles et son calendrier.

    `utils.preprocessing.dataframe_to_rasters` trie par (année, mois, y, x) puis
    remet à plat en (hauteur, largeur) : la ligne i d'une carte est le i-ème y
    croissant, la colonne j le j-ème x croissant. La reconstruction est donc
    exacte, à condition de reprendre la grille COMPLÉTÉE par
    `fill_all_missing_pixels`, qui ajoute les pixels manquants du produit
    cartésien. Ces pixels de remplissage sont masqués à l'inférence et n'existent
    pas dans SAFRAN : on ne garde que les (x, y) réellement présents dans le
    parquet, ce qui les élimine sans jamais les confondre avec un SWI nul.
    """
    cartes = np.load(chemin)
    if cartes.ndim == 5:
        cartes = cartes[:, :, 0]
    if cartes.ndim != 4:
        raise SystemExit(f"{chemin} : forme {cartes.shape} inattendue ; "
                         "attendu (trajectoires, pas, [1,] hauteur, largeur)")
    n_traj, n_pas, hauteur, largeur = cartes.shape

    ref = pd.read_parquet(grille, columns=["year", "month", "x", "y"])
    xs, ys = np.sort(ref["x"].unique()), np.sort(ref["y"].unique())
    if (len(ys), len(xs)) != (hauteur, largeur):
        raise SystemExit(f"grille incohérente : le .npy est {hauteur}x{largeur}, "
                         f"{grille.name} porte {len(ys)} y x {len(xs)} x")

    # Pixels réels de la grille complétée, et leur maille SAFRAN.
    yy, xx = np.meshgrid(ys, xs, indexing="ij")
    pixels = pd.DataFrame({"y": yy.ravel(), "x": xx.ravel()})
    pixels["rang_plat"] = np.arange(len(pixels))
    reels = ref[["x", "y"]].drop_duplicates()
    cles = charger_swi_historique(SWI_HIST_DEFAUT)[["maille", "x", "y"]].drop_duplicates()
    pixels = pixels.merge(reels, on=["x", "y"]).merge(cles, on=["x", "y"])
    absentes = len(reels) - len(pixels)
    if absentes:
        print(f"attention : {absentes} pixels de la grille d'inférence sans maille SAFRAN, "
              "ignorés")

    # Le calendrier : l'inférence génère tous les mois à partir de sa date de
    # départ, donc les n_pas derniers mois du parquet d'entrée.
    calendrier = (ref[["year", "month"]].drop_duplicates()
                  .sort_values(["year", "month"]).tail(n_pas))
    if len(calendrier) != n_pas:
        raise SystemExit(f"{grille.name} ne porte que {len(calendrier)} mois pour "
                         f"{n_pas} pas générés")

    plat = cartes.reshape(n_traj, n_pas, -1)[:, :, pixels["rang_plat"].values]
    idx = pd.MultiIndex.from_product(
        [range(n_traj), list(zip(calendrier["year"], calendrier["month"])),
         pixels["maille"].tolist()],
        names=["trajectoire", "date", "maille"])
    df = pd.DataFrame({"swi": plat.astype(np.float32).ravel()}, index=idx).reset_index()
    df[["annee", "mois"]] = pd.DataFrame(df["date"].tolist(), index=df.index)
    return df[["trajectoire", "maille", "annee", "mois", "swi"]]


# --------------------------------------------------------------------------- #
# Étape 1 — mois le plus sec                                                   #
# --------------------------------------------------------------------------- #

def indicateur_annuel(df: pd.DataFrame, cle: list[str]) -> pd.DataFrame:
    """Minimum des douze indicateurs mensuels, par maille et par année.

    Une année incomplète n'a pas de « mois le plus sec » : son minimum porterait
    sur les seuls mois présents et serait mécaniquement trop humide si l'été
    manque. Elle est écartée.
    """
    grp = df.groupby([*cle, "annee"])["swi"]
    ann = grp.agg(["min", "size"]).rename(columns={"min": "swi_min", "size": "n_mois"})
    incompletes = int((ann["n_mois"] < 12).sum())
    if incompletes:
        annees = sorted(ann.index.get_level_values("annee")[ann["n_mois"] < 12].unique())
        print(f"attention : {incompletes:,} couples (maille, année) sans 12 mois, écartés "
              f"(années {annees[:5]}{'...' if len(annees) > 5 else ''})")
        ann = ann[ann["n_mois"] == 12]
    return ann["swi_min"].reset_index()


# --------------------------------------------------------------------------- #
# Étape 2 — période de retour, et les seuils 1 et 2                            #
# --------------------------------------------------------------------------- #

def rangs_glissants(annuel: np.ndarray, fenetre: int) -> np.ndarray:
    """Rang de chaque année parmi elle-même et les `fenetre - 1` précédentes.

    `annuel` est de forme (..., années), années en ordre croissant. Le rang 1 va
    à l'année la plus sèche. Les ex aequo reçoivent le meilleur rang, ce qui est
    le sens permissif du régime.

    Le rang reste NaN partout où il n'est pas défini, et il faut l'écrire
    explicitement : une comparaison avec NaN est fausse, donc un indicateur
    manquant ne serait battu par aucune année de sa fenêtre et sortirait au
    rang 1 — la valeur la plus sèche possible. Deux cas le produisent, et tous
    deux doivent donner NaN : la fenêtre incomplète du début de l'historique, et
    une maille sans SWI sur l'année ou sur une année de sa fenêtre (une maille
    observée absente de la grille générée, par exemple).
    """
    n = annuel.shape[-1]
    rangs = np.full(annuel.shape, np.nan, dtype=np.float32)
    for i in range(fenetre - 1, n):
        win = annuel[..., i + 1 - fenetre: i + 1]
        courant = annuel[..., [i]]
        defini = ~np.isnan(win).any(axis=-1) & ~np.isnan(courant[..., 0])
        rang = (win < courant).sum(axis=-1) + 1
        rangs[..., i] = np.where(defini, rang, np.nan)
    return rangs


def appliquer_seuils(rangs: np.ndarray, rang_10: int, rang_5: int) -> dict[str, np.ndarray]:
    """Seuils 1 et 2 de la brochure. Le seuil 3 demanderait l'adjacence."""
    connu = ~np.isnan(rangs)
    seuil1 = connu & (rangs <= rang_10)
    rp5 = connu & (rangs <= rang_5)

    # Nombre d'années à >= 5 ans parmi les ANTECEDENTS précédentes, même maille.
    precedentes = np.zeros(rp5.shape, dtype=np.int16)
    for k in range(1, ANTECEDENTS + 1):
        decale = np.zeros_like(rp5)
        decale[..., k:] = rp5[..., :-k]
        precedentes += decale
    seuil2 = rp5 & (precedentes >= ANTECEDENTS_REQUIS)
    return {"seuil1": seuil1, "seuil2": seuil2, "anormal": seuil1 | seuil2}


def rangs_seuils(fenetre: int) -> tuple[int, int]:
    """Les rangs valant 10 et 5 ans de période de retour pour une fenêtre donnée."""
    if fenetre == FENETRE_DEFAUT:
        return RANG_10_ANS, RANG_5_ANS
    return max(1, round(fenetre / 10)), max(1, round(fenetre / 5))


# --------------------------------------------------------------------------- #
# Assemblage : historique préfixé, une matrice (trajectoire, maille, année)     #
# --------------------------------------------------------------------------- #

def matrice_annuelle(hist: pd.DataFrame, genere: pd.DataFrame | None,
                     reference: str) -> tuple[np.ndarray, list[int], list[int], list[int]]:
    """Cube (trajectoire, maille, année) des indicateurs annuels.

    En référence `mixte`, l'historique observé est préfixé à chaque trajectoire :
    la fenêtre de 30 ans d'une année de projection déborde sur l'observé, comme
    elle le ferait au régime. Une année générée qui recouvre une année observée
    remplace celle-ci — dans la ligne de temps d'une trajectoire, c'est le SWI
    généré qui fait foi.

    En référence `generee`, seules les années générées entrent, et chaque
    trajectoire est classée contre elle-même.
    """
    ann_h = indicateur_annuel(hist, ["maille"])
    mailles = sorted(ann_h["maille"].unique())
    idx_maille = {m: i for i, m in enumerate(mailles)}

    if genere is None:
        annees = sorted(ann_h["annee"].unique())
        cube = np.full((1, len(mailles), len(annees)), np.nan, dtype=np.float32)
        cube[0,
             ann_h["maille"].map(idx_maille).values,
             ann_h["annee"].map({a: i for i, a in enumerate(annees)}).values] = ann_h["swi_min"]
        return cube, [0], mailles, annees

    ann_g = indicateur_annuel(genere, ["trajectoire", "maille"])
    trajectoires = sorted(ann_g["trajectoire"].unique())
    annees_g = sorted(ann_g["annee"].unique())
    annees_h = sorted(ann_h["annee"].unique())

    if reference == "generee":
        annees = annees_g
        cube = np.full((len(trajectoires), len(mailles), len(annees)), np.nan, dtype=np.float32)
    else:
        # L'historique s'arrête là où la génération commence.
        annees = [a for a in annees_h if a < annees_g[0]] + annees_g
        cube = np.full((len(trajectoires), len(mailles), len(annees)), np.nan, dtype=np.float32)
        pos = {a: i for i, a in enumerate(annees)}
        socle = np.full((len(mailles), len(annees)), np.nan, dtype=np.float32)
        garde = ann_h[ann_h["annee"].isin(pos)]
        socle[garde["maille"].map(idx_maille).values,
              garde["annee"].map(pos).values] = garde["swi_min"]
        cube[:] = socle

    pos = {a: i for i, a in enumerate(annees)}
    idx_traj = {t: i for i, t in enumerate(trajectoires)}

    # Une maille observée mais absente de la grille générée n'a pas de
    # projection : ses années de projection doivent rester vides. Sans cela le
    # socle observé y survivrait au remplissage, et la maille serait jugée sur
    # de l'observé au milieu d'une trajectoire générée.
    if reference != "generee":
        sans_projection = ~np.isin(mailles, ann_g["maille"].unique())
        if sans_projection.any():
            debut = pos[annees_g[0]]
            cube[:, sans_projection, debut:] = np.nan

    cube[ann_g["trajectoire"].map(idx_traj).values,
         ann_g["maille"].map(idx_maille).values,
         ann_g["annee"].map(pos).values] = ann_g["swi_min"]
    return cube, trajectoires, mailles, annees


# --------------------------------------------------------------------------- #
# Diagnostic de biais — la fenêtre mixte compare deux populations              #
# --------------------------------------------------------------------------- #

def diagnostic_biais(hist: pd.DataFrame, genere: pd.DataFrame, reference: str) -> None:
    """Décalage des minima annuels générés contre les observés, sur le recouvrement.

    En fenêtre mixte, un rang oppose des années observées à des années générées.
    Tout décalage systématique entre les deux se lit alors comme un signal
    climatique. Le seul endroit où il se mesure honnêtement est le recouvrement.
    """
    ann_h = indicateur_annuel(hist, ["maille"])
    ann_g = indicateur_annuel(genere, ["trajectoire", "maille"])
    communes = sorted(set(ann_g["annee"]) & set(ann_h["annee"]))
    print("\n" + "=" * 78)
    print("DIAGNOSTIC DE BIAIS — minima annuels générés contre observés")
    print("=" * 78)
    if not communes:
        print("  aucune année de recouvrement : le décalage n'est pas mesurable ici.")
        print("  La fenêtre mixte oppose pourtant bien deux populations — cf. docstring,")
        print("  et utils/bias_correction.py, utils/climatology_anchor.py côté SwiGAN.")
        return

    ref = ann_h[ann_h["annee"].isin(communes)].set_index(["maille", "annee"])["swi_min"]
    gen = ann_g[ann_g["annee"].isin(communes)]
    ecart = gen["swi_min"].values - ref.reindex(
        pd.MultiIndex.from_arrays([gen["maille"], gen["annee"]])).values
    print(f"  recouvrement : {communes[0]}-{communes[-1]} "
          f"({len(communes)} an(s), {ann_g['trajectoire'].nunique()} trajectoire(s))")
    print(f"  écart moyen généré - observé : {np.nanmean(ecart):+.3f} SWI "
          f"(écart-type {np.nanstd(ecart):.3f})")
    par_an = pd.Series(ecart).groupby(gen["annee"].values).mean()
    print("  par année : " + ", ".join(f"{a} {v:+.3f}" for a, v in par_an.items()))
    sens = "plus SEC" if np.nanmean(ecart) < 0 else "plus HUMIDE"
    print(f"\n  Le SWI généré est en moyenne {sens} que l'observé.")
    if reference == "mixte":
        print("  En fenêtre mixte ce décalage entre directement dans le rang, et donc dans")
        print("  l'éligibilité : il n'est pas distinguable d'une dérive climatique. Corriger")
        print("  en amont (utils/climatology_anchor.py), ou passer --reference generee, dont")
        print("  le rang est invariant à toute transformation monotone du SWI.")
    else:
        print("  --reference generee range chaque année générée contre les seules années")
        print("  générées : le rang est invariant à toute transformation monotone du SWI, et")
        print("  ce décalage n'entre donc pas dans l'éligibilité. Il reste affiché parce")
        print("  qu'il dit ce que vaut la trajectoire, pas ce que vaut le rang.")


# --------------------------------------------------------------------------- #
# Étape 3, partielle — du pixel à la commune                                   #
# --------------------------------------------------------------------------- #

def charger_correspondance(chemin: Path) -> pd.DataFrame:
    """Table (commune, maille). Plusieurs mailles par commune sont attendues."""
    lire = pd.read_parquet if chemin.suffix.lower() in (".parquet", ".pq") else pd.read_csv
    brut = lire(chemin)
    dispo = {c.lower(): c for c in brut.columns}
    alias = {"code_commune_insee": ["code_commune_insee", "code_insee", "insee_com", "commune"],
             "maille": ["maille", "pixel", "cell", "id_maille"]}
    ren = {}
    for cible, noms in alias.items():
        trouve = next((dispo[n] for n in noms if n in dispo), None)
        if trouve is not None:
            ren[trouve] = cible
    df = brut.rename(columns=ren)
    manquantes = [c for c in alias if c not in df.columns]
    if manquantes:
        raise SystemExit(f"{chemin} : colonnes introuvables {manquantes} ; colonnes lues "
                         f"{list(brut.columns)}")
    df["code_commune_insee"] = (df["code_commune_insee"].astype(str)
                                .str.replace(r"\.0$", "", regex=True).str.zfill(5))
    df["maille"] = df["maille"].astype(int)
    return df[["code_commune_insee", "maille"]].drop_duplicates()


def eligibilite_communale(mailles_anormales: pd.DataFrame,
                          correspondance: pd.DataFrame) -> pd.DataFrame:
    """Une commune est éligible dès qu'UNE de ses mailles l'est.

    C'est la règle explicite de la brochure : « il suffit que le critère soit
    rempli pour une maille couvrant une partie de la commune pour qu'il soit
    considéré comme rempli pour toute la commune ».
    """
    df = mailles_anormales.merge(correspondance, on="maille", how="inner")
    return (df.groupby(["code_commune_insee", "annee", "trajectoire"], as_index=False)
              .agg(seuil1=("seuil1", "any"), seuil2=("seuil2", "any"))
              .assign(eligible=True))


# --------------------------------------------------------------------------- #
# Affichage                                                                    #
# --------------------------------------------------------------------------- #

def afficher_par_annee(res: dict[str, np.ndarray], annees: list[int], n_mailles: int,
                       n_traj: int, titre: str) -> None:
    """Comptage annuel des mailles anormales, moyenné sur les trajectoires."""
    print("\n" + "=" * 78)
    print(titre)
    print("=" * 78)
    print(f"  {'année':>6s} {'seuil 1':>9s} {'seuil 2':>9s} {'anormal':>9s} {'part':>7s}")
    valides = ~np.isnan(res["rangs"]).all(axis=(0, 1))
    for j, an in enumerate(annees):
        if not valides[j]:
            continue
        s1 = res["seuil1"][:, :, j].sum() / n_traj
        s2 = res["seuil2"][:, :, j].sum() / n_traj
        an_ = res["anormal"][:, :, j].sum() / n_traj
        print(f"  {an:6d} {s1:9.1f} {s2:9.1f} {an_:9.1f} {an_ / n_mailles:6.1%}")
    print(f"\n  {n_mailles} mailles SAFRAN, {n_traj} trajectoire(s) ; comptages moyennés")
    print("  sur les trajectoires. Les années dont la fenêtre glissante est incomplète")
    print("  sont omises : elles n'ont pas de période de retour.")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--swi-historique", type=Path, default=SWI_HIST_DEFAUT,
                   help="parquet du SWI mensuel observé (défaut : Grand Est/swi_region.parquet)")
    p.add_argument("--swi-genere", type=Path,
                   help="trajectoires générées : .npy d'inférence, ou tableau long")
    p.add_argument("--grille", type=Path,
                   help="parquet d'entrée de l'inférence, requis avec un .npy")
    p.add_argument("--reference", choices=["mixte", "generee"], default="mixte",
                   help="population de rang : observé + généré (défaut), ou généré seul")
    p.add_argument("--fenetre", type=int, default=FENETRE_DEFAUT,
                   help=f"années de la fenêtre glissante (défaut {FENETRE_DEFAUT})")
    p.add_argument("--correspondance", type=Path,
                   help="table (code_commune_insee, maille) ; sinon sortie à la maille")
    p.add_argument("--annees", type=int, nargs=2, metavar=("DEBUT", "FIN"),
                   help="restreindre la sortie à une période")
    p.add_argument("--csv", type=Path, help="fichier de sortie")
    a = p.parse_args()

    rang_10, rang_5 = rangs_seuils(a.fenetre)
    print(f"fenêtre glissante : {a.fenetre} ans ; rangs seuils {rang_10} (>= 10 ans) "
          f"et {rang_5} (>= 5 ans)")
    if a.fenetre < 10:
        print(f"attention : une fenêtre de {a.fenetre} ans ne peut pas exprimer une période "
              "de retour de 10\n  ans ; les deux seuils s'y confondent et le seuil 1 perd "
              "son sens. Résultat indicatif.")
    print("seuils 1 et 2 de la brochure CatNat ; le seuil 3 (communes limitrophes) "
          "demande l'adjacence\n  et n'est pas calculé : les comptages sont une borne "
          "inférieure.")

    hist = charger_swi_historique(a.swi_historique)
    print(f"\nSWI observé : {hist['maille'].nunique()} mailles, "
          f"{hist['annee'].min()}-{hist['annee'].max()}")

    genere = None
    if a.swi_genere:
        genere = charger_swi_genere(a.swi_genere, a.grille)
        print(f"SWI généré  : {genere['trajectoire'].nunique()} trajectoire(s), "
              f"{genere['maille'].nunique()} mailles, "
              f"{genere['annee'].min()}-{genere['annee'].max()}")
        orphelines = sorted(set(hist["maille"]) - set(genere["maille"]))
        if orphelines:
            print(f"  {len(orphelines)} mailles observées absentes de la grille générée, "
                  f"sans projection (ex. {orphelines[:3]})")
        diagnostic_biais(hist, genere, a.reference)

    cube, trajectoires, mailles, annees = matrice_annuelle(hist, genere, a.reference)
    if a.reference == "mixte" and genere is not None:
        print(f"\nfenêtre mixte : {annees[0]}-{annees[-1]}, historique préfixé à chaque "
              "trajectoire")

    rangs = rangs_glissants(cube, a.fenetre)
    res = appliquer_seuils(rangs, rang_10, rang_5)
    res["rangs"] = rangs

    titre = ("ÉLIGIBILITÉ MÉTÉOROLOGIQUE — mailles anormales par année"
             if genere is not None else
             "ÉLIGIBILITÉ MÉTÉOROLOGIQUE SUR L'OBSERVÉ — mailles anormales par année")
    afficher_par_annee(res, annees, len(mailles), len(trajectoires), titre)

    # Table longue des mailles anormales.
    it, im, ia = np.nonzero(res["anormal"])
    sortie = pd.DataFrame({
        "trajectoire": np.asarray(trajectoires)[it],
        "maille": np.asarray(mailles)[im],
        "annee": np.asarray(annees)[ia],
        "seuil1": res["seuil1"][it, im, ia],
        "seuil2": res["seuil2"][it, im, ia],
    })
    if a.annees:
        sortie = sortie[sortie["annee"].between(*a.annees)]

    if a.correspondance:
        corr = charger_correspondance(a.correspondance)
        couvertes = corr["maille"].isin(mailles)
        print(f"\ncorrespondance : {corr['code_commune_insee'].nunique():,} communes, "
              f"{corr['maille'].nunique()} mailles, {len(corr):,} couples ; "
              f"{(~couvertes).sum()} couples hors grille SAFRAN")
        sortie = eligibilite_communale(sortie, corr[couvertes])
        sortie = sortie[["code_commune_insee", "annee", "trajectoire", "eligible",
                         "seuil1", "seuil2"]]
        print(f"éligibilité communale : {len(sortie):,} lignes "
              f"(commune, année, trajectoire), "
              f"{sortie['code_commune_insee'].nunique():,} communes distinctes")
        if len(sortie):
            par_an = sortie.groupby("annee")["code_commune_insee"].nunique()
            print(f"  communes éligibles par année : médiane {par_an.median():.0f}, "
                  f"max {par_an.max()} en {par_an.idxmax()}")
    else:
        sortie["eligible"] = True
        sortie = sortie[["maille", "annee", "trajectoire", "eligible", "seuil1", "seuil2"]]
        print(f"\nsortie à la maille : {len(sortie):,} lignes (maille, année, trajectoire)")
        print("  Pour la table qu'attend cout_subsidence_futur.py : --correspondance.")

    if a.csv:
        sortie.to_csv(a.csv, index=False)
        print(f"\nécrit dans {a.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
