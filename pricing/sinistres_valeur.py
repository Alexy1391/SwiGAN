#!/usr/bin/env python3
"""Sinistres événements naturels (SIN_EVT_NAT) rapprochés de la valeur des maisons.

Lit la base de sinistres de la Chaire Allianz, en tire le coût de chaque
sinistre, et le met en regard de la valeur des maisons de sa commune telle que
la portent les extractions BDNB (`output/maisons_dep*.csv`).

SCHÉMA INCONNU, DONC DÉTECTÉ
----------------------------
Le schéma de SIN_EVT_NAT.csv n'est pas documenté dans le dépôt. Encodage
(UTF-8 ou Windows-1252), séparateur et virgule décimale sont détectés ; chaque
colonne utile est cherchée par alias, puis confirmée sur son contenu. Commencer
par :

    python3 sinistres_valeur.py --inspecter

qui affiche les colonnes lues, la correspondance retenue et ce qu'elle permet,
sans rien écrire. Une correspondance fausse se corrige par --col :

    python3 sinistres_valeur.py --col montant=CHARGE_TOTALE --col commune=CODE_INSEE
    python3 sinistres_valeur.py --col montant=MT_REGLE+MT_RESERVE   # somme de colonnes

Quand un même sinistre occupe plusieurs lignes (une par garantie ou par
règlement), ses montants sont sommés. C'est faux si les lignes sont des
photographies successives de la même charge : --inspecter signale le cas.

À QUEL NIVEAU LE RAPPROCHEMENT EST POSSIBLE
-------------------------------------------
Les fichiers maisons ne portent pas de coordonnées, seulement le code INSEE de
la commune et l'IRIS. Le rapprochement se fait donc **par commune** : chaque
sinistre reçoit la valeur médiane des maisons vendues de sa commune. Si la base
de sinistres ne porte qu'un code postal, la commune est retrouvée par
(département, nom). Un rapprochement maison par maison demanderait une adresse
ou des coordonnées côté sinistres, et les géométries BDNB que seul dep_08/
conserve.

La valeur est la valeur DVF (`dvf_valeur_fonciere`), qui n'existe que dans
`maisons_dep08.csv` : les neuf autres départements ont été extraits avant l'ajout
de la table DVF. Hors Ardennes, les sinistres sont lus et agrégés, mais sans
valeur en regard — NaN, pas zéro. Une médiane sur moins de --min-ventes ventes
est écartée : sous 5 transactions, l'erreur communale dépasse 20 % (README).

RÉSERVES DE LECTURE
-------------------
- La valeur DVF est une valeur vénale, terrain compris, pas un coût de
  reconstruction : montant / valeur est un taux de dommage relatif, pas un taux
  de destruction au sens assurantiel. Si la base porte un capital assuré, le
  rapport montant / capital est calculé à côté.
- Les sinistres ne couvrent que le portefeuille Allianz. Le nombre de sinistres
  rapporté au parc BDNB mêle sinistralité et part de marché ; le coût moyen et le
  rapport montant / valeur, eux, n'en dépendent pas.
- Les ventes DVF datent de 2014-2025, en euros courants. --annees restreint les
  sinistres à une fenêtre comparable.

Usage : python3 sinistres_valeur.py [--sin CHEMIN] [--inspecter] [--peril secheresse]
"""

import argparse
import re
import unicodedata
from pathlib import Path

import duckdb
import pandas as pd

ROOT = Path(__file__).resolve().parent
SORTIE = ROOT / "output"
# Lecteur réseau sous Windows ; à défaut, une copie déposée dans l'espace de travail.
SIN_CANDIDATS = [Path(r"P:\Chaire allianz\SIN_EVT_NAT.csv"),
                 ROOT.parent / "SIN_EVT_NAT.csv", ROOT / "SIN_EVT_NAT.csv"]

# Mêmes bornes que modele_valeur.py : une maison seule, prix plausible.
VALEUR_MIN, VALEUR_MAX = 5_000, 1_000_000
ECHANTILLON = 50_000

# Rôle -> alias, du plus spécifique au plus générique. Un alias reconnaît une
# colonne dont le nom normalisé contient ses mots, contigus et dans l'ordre :
# « cout » reconnaît COUT_SINISTRE, pas COUTURE. L'ordre des rôles compte, une
# colonne prise par un rôle ne l'est plus par les suivants.
ALIAS = {
    "commune": ["code_commune_insee", "code_insee", "codeinsee", "insee", "code_commune",
                "codecommune", "depcom", "codgeo", "cog"],
    "code_postal": ["code_postal", "codepostal", "cp", "cpostal"],
    "montant": ["charge_totale", "charge_sinistre", "charge", "cout_total", "cout_sinistre", "cout",
                "montant_total", "montant_sinistre", "montant", "mt", "total_regle", "regle",
                "paye", "indemnite"],
    "valeur_assuree": ["capital_assure", "valeur_assuree", "somme_assuree", "capital", "smp"],
    "date": ["date_survenance", "survenance", "date_sinistre", "date_evenement", "date_evt",
             "date_sin", "date_declaration", "date", "annee_survenance", "annee", "exercice"],
    "peril": ["type_evenement", "evenement", "type_evt", "evt", "peril", "type_sinistre",
              "nature_sinistre", "cause", "garantie", "nature"],
    "type_bien": ["type_bien", "nature_bien", "type_habitation", "type_logement", "type_risque",
                  "categorie_bien", "bien", "habitation"],
    "libelle_commune": ["libelle_commune", "nom_commune", "commune", "ville", "localite"],
    "id": ["num_sinistre", "numero_sinistre", "no_sinistre", "n_sinistre", "id_sinistre",
           "ref_sinistre", "num_sin", "numero", "num_dossier", "dossier", "reference"],
    "latitude": ["latitude", "lat"],
    "longitude": ["longitude", "lon", "lng"],
    "adresse": ["adresse", "voie", "rue"],
}
# Mots qui disqualifient une colonne pour un rôle. Une réserve seule n'est pas
# un coût, un numéro de contrat n'identifie pas un sinistre.
EXCLUS = {
    "montant": {"franchise", "capital", "prime", "cotisation", "assure", "assuree", "recours",
                "reserve", "smp", "valeur"},
    "valeur_assuree": {"franchise", "prime", "cotisation"},
    "id": {"contrat", "police", "client", "assure", "adherent", "souscripteur", "date"},
}
NUMERIQUES = {"montant", "valeur_assuree", "latitude", "longitude"}
MOTIF_CODE = r"\d{5}|2[AB]\d{3}"


# --------------------------------------------------------------------------- #
# Lecture et conversion                                                        #
# --------------------------------------------------------------------------- #

def _cle(nom: str) -> str:
    """Nom normalisé : minuscules, sans accents, mots séparés par « _ »."""
    s = unicodedata.normalize("NFKD", str(nom)).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")


def _reconnait(alias: str, colonne: str) -> bool:
    a, c = alias.split("_"), _cle(colonne).split("_")
    return any(c[i:i + len(a)] == a for i in range(len(c) - len(a) + 1))


def lire_montant(v: pd.Series) -> pd.Series:
    """Texte -> nombre ; espaces, « € » et séparateur de milliers ignorés.

    Le séparateur décimal est le dernier des deux qui apparaît : « 1.234,56 » et
    « 1,234.56 » donnent tous deux 1234.56. Seul « 1,234 » reste ambigu ; il est
    lu à la française, 1.234.
    """
    # Insécables passés en caractères, non en « \u » : le moteur regex de pyarrow le refuse.
    t = v.astype(str).str.replace("[\\s\u00a0\u202f€]", "", regex=True)
    virgule = t.str.rfind(",") > t.str.rfind(".")
    t = t.where(~virgule, t.str.replace(".", "", regex=False).str.replace(",", ".", regex=False))
    t = t.where(virgule, t.str.replace(",", "", regex=False))
    return pd.to_numeric(t, errors="coerce")


def lire_annee(v: pd.Series) -> pd.Series:
    """Année d'une date en texte : « 15/07/2018 », « 2018-07-15 », « 20180715 »,
    « 2018 ». Un millésime à quatre chiffres suffit, le reste de la date n'est
    pas lu ; une année à deux chiffres passe par pandas."""
    v = v.astype(str).str.strip()
    a = pd.to_numeric(v.str.extract(r"((?:19|20)\d{2})", expand=False), errors="coerce")
    reste = a.isna() & v.ne("")
    if reste.any():
        a[reste] = pd.to_datetime(v[reste], errors="coerce", dayfirst=True, format="mixed").dt.year
    return a.astype("Int64")


def lire_code(v: pd.Series) -> pd.Series:
    """Code commune ou postal sur 5 caractères ; « 8105.0 » (export tableur) -> 08105."""
    return v.astype(str).str.strip().str.upper().str.replace(r"\.0$", "", regex=True).str.zfill(5)


def _valide(role: str, v: pd.Series) -> bool:
    """La colonne a-t-elle le contenu attendu pour ce rôle ?"""
    v = v[v.str.strip() != ""]
    if v.empty:
        return False
    if role in ("commune", "code_postal"):
        return lire_code(v).str.fullmatch(MOTIF_CODE).mean() > .8
    if role in NUMERIQUES:
        return lire_montant(v).notna().mean() > .8
    if role == "date":
        return lire_annee(v).notna().mean() > .8
    if role == "id":
        return v.nunique() > .1 * len(v) and lire_annee(v).notna().mean() < .5
    return lire_montant(v).isna().mean() > .8      # libellés : du texte, pas des nombres


def ouvrir(chemin: Path) -> tuple[str, str]:
    """Encodage et séparateur, d'après le premier Mo du fichier."""
    with open(chemin, "rb") as f:
        brut = f.read(1 << 20)
    try:
        texte, encodage = brut.decode("utf-8-sig"), "utf-8-sig"
    except UnicodeDecodeError as e:
        if e.start >= len(brut) - 3:       # caractère coupé par la fin du bloc lu
            texte, encodage = brut[:e.start].decode("utf-8-sig"), "utf-8-sig"
        else:
            texte, encodage = brut.decode("cp1252", errors="replace"), "cp1252"
    # L'en-tête ne contient pas de virgule décimale : c'est lui qui tranche.
    entete = texte.splitlines()[0]
    return encodage, max(";\t|,", key=entete.count)


def lire(chemin: Path, encodage: str, sep: str, **kw) -> pd.DataFrame:
    """Lecture brute, tout en texte : les codes gardent leur zéro initial et les
    montants leur virgule, convertis ensuite rôle par rôle."""
    return pd.read_csv(chemin, sep=sep, encoding=encodage, dtype=str,
                       keep_default_na=False, **kw)


def detecter(ech: pd.DataFrame, forces: dict[str, list[str]]) -> dict[str, list[str]]:
    """Rôle -> colonne(s) source. Les --col imposés passent d'abord ; chaque
    autre rôle prend la première colonne libre qu'un alias reconnaît et que son
    contenu confirme."""
    roles, prises = {}, set()
    for role, cols in forces.items():
        absentes = [c for c in cols if c not in ech.columns]
        if absentes:
            raise SystemExit(f"--col {role} : colonnes absentes {absentes}\n"
                             f"  colonnes lues : {list(ech.columns)}")
        roles[role] = cols
        prises.update(cols)
    for role, alias in ALIAS.items():
        if role in roles:
            continue
        for a in alias:
            col = next((c for c in ech.columns
                        if c not in prises and _reconnait(a, c)
                        and not EXCLUS.get(role, set()) & set(_cle(c).split("_"))
                        and _valide(role, ech[c])), None)
            if col is not None:
                roles[role] = [col]
                prises.add(col)
                break
    return roles


def convertir(brut: pd.DataFrame, roles: dict[str, list[str]]) -> pd.DataFrame:
    """Une colonne par rôle, typée. Un montant sur plusieurs colonnes est leur
    somme, NaN seulement si toutes sont vides."""
    df = pd.DataFrame(index=brut.index)
    for role, cols in roles.items():
        if role in NUMERIQUES:
            df[role] = pd.concat([lire_montant(brut[c]) for c in cols], axis=1).sum(axis=1, min_count=1)
        elif role == "date":
            df["annee"] = lire_annee(brut[cols[0]])
        elif role in ("commune", "code_postal"):
            df[role] = lire_code(brut[cols[0]])
        else:
            df[role] = brut[cols[0]].str.strip()
    return df


# --------------------------------------------------------------------------- #
# Parc BDNB et rattachement communal                                           #
# --------------------------------------------------------------------------- #

def communes_bdnb() -> pd.DataFrame:
    """Une ligne par commune extraite : effectif du parc et valeur DVF des
    maisons vendues seules. Les départements sans colonnes DVF donnent NaN."""
    motif = (SORTIE / "maisons_dep*.csv").as_posix()
    source = (f"read_csv('{motif}', header=true, union_by_name=true, "
              f"types={{'code_commune_insee': 'VARCHAR', 'code_departement_insee': 'VARCHAR'}})")
    con = duckdb.connect()
    colonnes = {r[0] for r in con.execute(f"DESCRIBE SELECT * FROM {source}").fetchall()}
    if "dvf_valeur_fonciere" in colonnes:
        vente = (f"dvf_mutation_simple AND dvf_valeur_fonciere "
                 f"BETWEEN {VALEUR_MIN} AND {VALEUR_MAX}")
        dvf = f"""count(*) FILTER ({vente}) AS n_ventes,
                  median(dvf_valeur_fonciere) FILTER ({vente}) AS valeur_mediane,
                  avg(dvf_valeur_fonciere) FILTER ({vente}) AS valeur_moyenne,
                  median(dvf_prix_m2) FILTER ({vente}) AS prix_m2_median,"""
    else:
        dvf = """0 AS n_ventes, NULL::DOUBLE AS valeur_mediane,
                 NULL::DOUBLE AS valeur_moyenne, NULL::DOUBLE AS prix_m2_median,"""
    return con.execute(f"""
        SELECT code_commune_insee,
               any_value(libelle_commune_insee) AS libelle_commune_insee,
               any_value(code_departement_insee) AS code_departement_insee,
               count(*) AS n_maisons,
               {dvf}
               median(surface_sol_m2) AS emprise_mediane_m2,
               median(annee_construction) AS annee_construction_mediane
        FROM {source}
        GROUP BY code_commune_insee
    """).df()


def nom_cle(s: pd.Series) -> pd.Series:
    """Nom de commune comparable d'une source à l'autre : sans accents, casse ni
    ponctuation ; SAINT(E) abrégé, article entre parenthèses remis en tête
    (« CHESNE (LE) » -> « LE CHESNE »), CEDEX retiré."""
    t = s.astype(str).str.replace(r"\bCEDEX\b.*$", "", regex=True, flags=re.I)
    t = t.str.replace(r"^(.*?)\s*\((LE|LA|LES|L')\)\s*$", r"\2 \1", regex=True, flags=re.I)
    t = t.map(_cle).str.replace("_", " ").str.upper()
    t = t.str.replace(r"\bSAINTE\b", "STE", regex=True).str.replace(r"\bSAINT\b", "ST", regex=True)
    return t.str.replace(" ", "", regex=False)


def rattacher(df: pd.DataFrame, ref: pd.DataFrame) -> tuple[pd.Series, str]:
    """Code INSEE de chaque sinistre, et comment il a été obtenu.

    Un code postal se fait souvent passer pour un code commune : 08000 est le
    code postal de Charleville-Mézières, dont le code INSEE est 08105. Que le
    code existe dans la BDNB ne prouve rien — les codes postaux des Ardennes,
    08000 à 08460 de dix en dix, sont presque tous aussi des codes INSEE, d'une
    autre commune. C'est le nom qui tranche : un code INSEE désigne une commune
    du même nom que celle du sinistre. Sinon le code est relu comme code
    postal, et la commune retrouvée par (département, nom).
    """
    connus = set(ref["code_commune_insee"])
    deps = set(ref["code_departement_insee"])
    cp = df.get("code_postal")
    if "commune" in df:
        code = df["commune"]
        perimetre = code.str[:2].isin(deps)
        connu = code.isin(connus)
        if "libelle_commune" in df and connu.any():
            noms = dict(zip(ref["code_commune_insee"], nom_cle(ref["libelle_commune_insee"])))
            accord = (code[connu].map(noms) == nom_cle(df.loc[connu, "libelle_commune"])).mean()
            if accord >= .5:
                return code, f"code INSEE (nom concordant pour {accord:.1%} des codes reconnus)"
            cp = code
        else:
            taux = connu[perimetre].mean() if perimetre.any() else 0.0
            # Un code postal finit presque toujours par 0, un code INSEE une fois sur dix.
            if perimetre.any() and code[perimetre].str.endswith("0").mean() > .5:
                print("ATTENTION : la plupart des codes finissent par 0 — probablement des "
                      "codes postaux, que rien ne permet de vérifier sans nom de commune.")
            return code, f"code INSEE ({taux:.1%} des codes du périmètre reconnus, non vérifié par le nom)"
    if cp is None:
        raise SystemExit("Aucune colonne de commune ni de code postal reconnue : "
                         "préciser --col commune=... ou --col code_postal=...")
    if "libelle_commune" not in df:
        raise SystemExit("Code postal sans nom de commune : le rattachement à la commune "
                         "est impossible. Préciser --col libelle_commune=...")
    cle = ref["code_departement_insee"] + "|" + nom_cle(ref["libelle_commune_insee"])
    univoque = ~cle.duplicated(keep=False)          # homonymes d'un même département écartés
    table = dict(zip(cle[univoque], ref.loc[univoque, "code_commune_insee"]))
    code = (cp.str[:2] + "|" + nom_cle(df["libelle_commune"])).map(table)
    return code, "code postal + nom de commune"


# --------------------------------------------------------------------------- #
# Rapports                                                                     #
# --------------------------------------------------------------------------- #

def _n(x) -> str:
    return f"{x:,.0f}".replace(",", " ")


def _apercu(v: pd.Series, k: int = 4) -> str:
    vals = v[v.str.strip() != ""].drop_duplicates().head(k).tolist()
    return " | ".join(vals)


def inspecter(chemin, encodage, sep, ech, roles, ref):
    print(f"{chemin}\n  encodage {encodage}, séparateur « {sep} », "
          f"{_n(len(ech))} lignes examinées, {len(ech.columns)} colonnes\n")
    print("Colonnes lues :", ", ".join(ech.columns), "\n")

    print("Correspondance retenue (à corriger par --col role=COLONNE) :")
    for role in ALIAS:
        cols = roles.get(role)
        if cols:
            print(f"  {role:16s} {'+'.join(cols):28s} ex. {_apercu(ech[cols[0]])}")
        else:
            print(f"  {role:16s} —")
    libres = [c for c in ech.columns if c not in {x for cs in roles.values() for x in cs}
              and _valide("montant", ech[c]) and not _valide("commune", ech[c])]
    if libres:
        print(f"\nAutres colonnes numériques, à sommer au montant si besoin : {', '.join(libres)}")

    df = convertir(ech, roles)
    for role in ("peril", "type_bien"):
        if role in df:
            print(f"\nValeurs de {role} :")
            print(df[role].value_counts().head(12).to_string())
    if "montant" in df:
        m = df["montant"]
        print(f"\nMontant lisible sur {m.notna().mean():.1%} des lignes ; "
              f"nul {(m == 0).mean():.1%}, négatif {(m < 0).mean():.1%}, "
              f"médiane des positifs {_n(m[m > 0].median())} €")
    if "annee" in df:
        print(f"Années : {df['annee'].min()} – {df['annee'].max()}")
    if "id" in df:
        n_id = df["id"].nunique()
        print(f"Identifiant : {_n(len(df))} lignes pour {_n(n_id)} sinistres distincts"
              + (" -> plusieurs lignes par sinistre, montants sommés par défaut."
                 " Vérifier qu'il s'agit de garanties ou de règlements, pas de"
                 " photographies successives de la charge." if n_id < len(df) else ""))
    else:
        print("Aucun identifiant de sinistre : chaque ligne est tenue pour un sinistre.")

    print("\nRapprochement avec les maisons BDNB :")
    if "commune" in df or "code_postal" in df:
        code, methode = rattacher(df, ref)
        perimetre = code.str[:2].isin(set(ref["code_departement_insee"]))
        print(f"  par commune : oui, via {methode}")
        print(f"  sinistres dans les {ref['code_departement_insee'].nunique()} départements "
              f"extraits : {perimetre.mean():.1%} ; commune retrouvée : "
              f"{code.isin(set(ref['code_commune_insee'])).mean():.1%}")
    else:
        print("  par commune : non, aucune colonne de commune reconnue")
    geo = [r for r in ("latitude", "longitude", "adresse") if r in roles]
    print("  par bâtiment : " + (f"colonnes {', '.join(geo)} présentes — à développer sur dep_08/"
                                 if geo else "non, ni adresse ni coordonnées"))
    avec_dvf = ref[ref["n_ventes"] > 0]
    print(f"  valeur DVF disponible pour {len(avec_dvf)} communes "
          f"(départements {', '.join(sorted(avec_dvf['code_departement_insee'].unique()))})")


def rapport(sin: pd.DataFrame, n_lignes: int, methode: str, min_sinistres: int):
    paye = sin["montant"] > 0
    avec_valeur = sin["valeur_mediane_commune"].notna()
    print(f"\n{_n(n_lignes)} lignes -> {_n(len(sin))} sinistres ; montant lisible "
          f"{sin['montant'].notna().mean():.1%}, positif {paye.mean():.1%}")
    print(f"Rattachement : {methode}")
    print(f"  dans le parc BDNB extrait : {_n(sin['n_maisons'].notna().sum())} "
          f"({sin['n_maisons'].notna().mean():.1%})")
    print(f"  avec valeur DVF communale : {_n(avec_valeur.sum())} ({avec_valeur.mean():.1%})")

    p = sin[paye]
    if "peril" in sin and sin["peril"].ne("").any():
        par = p.groupby("peril").agg(sinistres_payes=("montant", "size"),
                                     cout_median=("montant", "median"),
                                     valeur_mediane=("valeur_mediane_commune", "median"),
                                     rapport_median=("rapport_montant_valeur", "median"))
        print("\nSinistres payés par péril :")
        print(par.sort_values("sinistres_payes", ascending=False).head(10)
              .round({"cout_median": 0, "valeur_mediane": 0, "rapport_median": 4}).to_string())

    pv = p[p["valeur_mediane_commune"].notna()]
    if len(pv) < 20:
        print("\nTrop peu de sinistres payés avec valeur communale pour la mise en regard "
              f"({len(pv)}) : la valeur DVF n'existe que pour les Ardennes.")
        return
    q = pv["rapport_montant_valeur"].quantile([.25, .5, .75])
    print(f"\nMontant / valeur médiane communale, sinistres payés ({_n(len(pv))}) : "
          f"médiane {q[.5]:.2%}, quartiles {q[.25]:.2%} – {q[.75]:.2%}")

    # La valeur communale classe-t-elle les coûts ? Quintiles de valeur au
    # niveau du sinistre ; les égalités (une valeur par commune) fusionnent
    # éventuellement des classes.
    classe = pd.qcut(pv["valeur_mediane_commune"], 5, duplicates="drop")
    tab = pv.groupby(classe, observed=True).agg(
        sinistres=("montant", "size"), communes=("code_commune_insee", "nunique"),
        cout_median=("montant", "median"), rapport_median=("rapport_montant_valeur", "median"))
    print("\nPar classe de valeur médiane communale :")
    print(tab.round({"cout_median": 0, "rapport_median": 4}).to_string())

    com = (pv.groupby("code_commune_insee")
             .agg(n=("montant", "size"), cout=("montant", "median"),
                  valeur=("valeur_mediane_commune", "first")))
    com = com[com["n"] >= min_sinistres]
    if len(com) >= 10:
        rho = com["cout"].rank().corr(com["valeur"].rank())
        print(f"\nCorrélation de rang, coût médian contre valeur médiane, sur {len(com)} "
              f"communes à {min_sinistres} sinistres payés ou plus : ρ = {rho:+.2f}")


# --------------------------------------------------------------------------- #

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--sin", type=Path, default=None,
                    help="base de sinistres (défaut : P:\\Chaire allianz\\SIN_EVT_NAT.csv, "
                         "puis une copie à côté de pricing/)")
    ap.add_argument("--inspecter", action="store_true",
                    help="afficher colonnes et correspondance détectée, sans rien écrire")
    ap.add_argument("--col", action="append", default=[], metavar="ROLE=COLONNE[+COLONNE]",
                    help=f"imposer une colonne ; rôles : {', '.join(ALIAS)}")
    ap.add_argument("--bien", default="maison|individuel|pavillon",
                    help="regex sur le type de bien, s'il est reconnu ('' : pas de filtre)")
    ap.add_argument("--peril", default=None,
                    help="regex sur le péril, sans accents ni casse (ex. secheresse)")
    ap.add_argument("--annees", type=int, nargs=2, default=None, metavar=("DEBUT", "FIN"),
                    help="années de survenance retenues, bornes comprises")
    ap.add_argument("--min-ventes", type=int, default=5,
                    help="ventes DVF minimales pour qu'une valeur communale soit retenue")
    ap.add_argument("--min-sinistres", type=int, default=5,
                    help="sinistres payés minimaux par commune pour la corrélation")
    args = ap.parse_args()

    chemin = args.sin or next((c for c in SIN_CANDIDATS if c.exists()), None)
    if chemin is None or not chemin.exists():
        raise SystemExit("Base de sinistres introuvable. Chemins essayés :\n  "
                         + "\n  ".join(str(c) for c in ([args.sin] if args.sin else SIN_CANDIDATS))
                         + "\nLe lecteur P: n'existe que sous Windows : lancer le script sur un "
                           "poste qui le monte, ou passer --sin vers une copie.")

    forces = {}
    for spec in args.col:
        role, _, cols = spec.partition("=")
        if role not in ALIAS or not cols:
            raise SystemExit(f"--col {spec!r} : attendu ROLE=COLONNE, ROLE parmi {list(ALIAS)}")
        forces[role] = cols.split("+")

    encodage, sep = ouvrir(chemin)
    ech = lire(chemin, encodage, sep, nrows=ECHANTILLON)
    roles = detecter(ech, forces)
    ref = communes_bdnb()

    if args.inspecter:
        inspecter(chemin, encodage, sep, ech, roles, ref)
        return 0

    if "montant" not in roles:
        raise SystemExit("Aucune colonne de montant reconnue : lancer --inspecter, "
                         "puis --col montant=...")
    brut = lire(chemin, encodage, sep, usecols=sorted({c for cs in roles.values() for c in cs}))
    df = convertir(brut, roles)
    n_lignes = len(df)
    for role in ("peril", "type_bien"):
        if role not in df:
            df[role] = ""

    if args.bien and roles.get("type_bien"):
        garde = df["type_bien"].map(_cle).str.contains(args.bien, regex=True)
        if not garde.any():
            raise SystemExit(f"--bien {args.bien!r} n'élimine que... tout. Valeurs de "
                             f"{roles['type_bien'][0]} :\n"
                             + df["type_bien"].value_counts().head(12).to_string()
                             + "\nAjuster --bien, ou --bien '' pour ne pas filtrer.")
        print(f"Type de bien ({roles['type_bien'][0]} ~ {args.bien!r}) : "
              f"{_n(garde.sum())} lignes gardées sur {_n(len(df))}")
        df = df[garde]
    elif args.bien:
        print("Aucune colonne de type de bien reconnue : maisons et appartements restent mêlés.")
    if args.peril:
        if not roles.get("peril"):
            raise SystemExit("--peril demandé, mais aucune colonne de péril reconnue : --col peril=...")
        df = df[df["peril"].map(_cle).str.contains(args.peril, regex=True)]
    if args.annees:
        if "annee" not in df:
            raise SystemExit("--annees demandé, mais aucune colonne de date reconnue : --col date=...")
        df = df[df["annee"].between(*args.annees)]
    if df.empty:
        raise SystemExit("Aucun sinistre après filtrage.")

    # Plusieurs lignes par sinistre : montants sommés, le reste pris sur la
    # première ligne. Le capital assuré se répète, il ne se somme pas.
    if "id" in df:
        g = df.groupby("id", sort=False)
        df = g.first().assign(montant=g["montant"].sum(min_count=1)).reset_index()

    df["code_commune_insee"], methode = rattacher(df, ref)
    sin = df.merge(ref, on="code_commune_insee", how="left")
    sin["valeur_mediane_commune"] = sin["valeur_mediane"].where(sin["n_ventes"] >= args.min_ventes)
    sin["rapport_montant_valeur"] = sin["montant"] / sin["valeur_mediane_commune"]
    if "valeur_assuree" in sin:
        capital = sin["valeur_assuree"].where(sin["valeur_assuree"] > 0)
        sin["rapport_montant_capital"] = sin["montant"] / capital

    rapport(sin, n_lignes, methode, args.min_sinistres)

    paye = sin["montant"].where(sin["montant"] > 0)
    com = (sin.assign(montant_paye=paye,
                      rapport_paye=sin["rapport_montant_valeur"].where(paye.notna()))
              .groupby(["code_commune_insee", "peril"], dropna=False)
              .agg(n_sinistres=("montant", "size"), n_payes=("montant_paye", "count"),
                   cout_total=("montant", "sum"), cout_moyen_paye=("montant_paye", "mean"),
                   cout_median_paye=("montant_paye", "median"),
                   rapport_median_montant_valeur=("rapport_paye", "median"))
              .reset_index()
              .merge(ref, on="code_commune_insee", how="left"))
    # Rapporté au parc entier, pas au portefeuille : à lire en relatif entre communes.
    com["sinistres_pour_1000_maisons"] = 1000 * com["n_sinistres"] / com["n_maisons"]

    SORTIE.mkdir(exist_ok=True)
    for nom, tab in (("sinistres_evt_nat.csv", sin), ("sinistres_valeur_commune.csv", com)):
        tab.to_csv(SORTIE / nom, index=False)
        print(f"-> {SORTIE / nom} ({_n(len(tab))} lignes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
