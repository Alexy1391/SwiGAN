#!/usr/bin/env python3
"""Extraction des maisons individuelles depuis les exports CSV de la BDNB.

Pour chaque département disponible sous ``pricing/dep_<code>/csv/``, on
reconstitue une ligne par maison individuelle avec ses caractéristiques
structurantes pour un modèle de prix : nombre de niveaux, emprise au sol,
année de construction.

Source des variables (cf. documentation.xlsx / metadonnees_colonne.csv) :
  - usage_principal_bdnb_open  <- batiment_groupe_synthese_propriete_usage
        Indicateur d'usage corrigé (les bâtiments publics classés à tort en
        résidentiel individuel par les Fichiers Fonciers y sont requalifiés
        en tertiaire). C'est lui qui définit le périmètre « maison ».
  - nb_niveau, annee_construction, nb_log, mat_mur_txt, mat_toit_txt
        <- batiment_groupe_ffo_bat (Fichiers Fonciers)
  - s_geom_groupe              <- batiment_groupe (emprise au sol du groupe,
        géométrie Lambert-93 donc en m²) + rattachement commune / IRIS
  - hauteur, s_geom_cstr       <- batiment_construction (BD TOPO IGN),
        agrégés au niveau du groupe de bâtiments

Usage :
    python3 extract_maisons.py                # tous les départements trouvés
    python3 extract_maisons.py --dep 08       # un département
    python3 extract_maisons.py --dep 08 51 57 # plusieurs
    python3 extract_maisons.py --aggregate-only  # réassembler l'agrégat régional

Le fichier régional `output/maisons_grand_est.csv` est reconstruit à chaque
exécution depuis les CSV départementaux présents dans `output/` : les exports
bruts n'ont donc pas besoin de coexister sur le disque.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "output"

# Départements de la région Grand Est (INSEE).
GRAND_EST = ["08", "10", "51", "52", "54", "55", "57", "67", "68", "88"]

USAGE_MAISON = "Résidentiel individuel"

# Bornes de plausibilité pour signaler (sans supprimer) les valeurs douteuses.
ANNEE_MIN, ANNEE_MAX = 1800, 2026
SURFACE_MIN, SURFACE_MAX = 10.0, 2000.0

# Colonnes des CSV volumineux : les déclarer explicitement permet à DuckDB de
# ne pas matérialiser les colonnes de géométrie WKT (plusieurs centaines de Mo).
COLS_BATIMENT_GROUPE = {
    "geom_groupe": "VARCHAR",
    "batiment_groupe_id": "VARCHAR",
    "code_departement_insee": "VARCHAR",
    "s_geom_groupe": "DOUBLE",
    "code_iris": "VARCHAR",
    "code_commune_insee": "VARCHAR",
    "libelle_commune_insee": "VARCHAR",
    "code_epci_insee": "VARCHAR",
    "contient_fictive_geom_groupe": "INTEGER",
}

COLS_BATIMENT_CONSTRUCTION = {
    "WKT": "VARCHAR",
    "batiment_construction_id": "VARCHAR",
    "batiment_groupe_id": "VARCHAR",
    "code_departement_insee": "VARCHAR",
    "code_iris": "VARCHAR",
    "code_commune_insee": "VARCHAR",
    "s_geom_cstr": "DOUBLE",
    "hauteur": "DOUBLE",
    "altitude_sol": "DOUBLE",
    "fictive_hauteur_cstr": "INTEGER",
    "fictive_geom_cstr": "INTEGER",
}


def _read_csv(path: Path, columns: dict[str, str] | None = None) -> str:
    """Expression SQL de lecture d'un CSV BDNB (séparateur ';', en-tête)."""
    opts = ["delim=';'", "header=true"]
    if columns:
        spec = ", ".join(f"'{k}': '{v}'" for k, v in columns.items())
        opts.append(f"columns={{{spec}}}")
    return f"read_csv('{path.as_posix()}', {', '.join(opts)})"


REQUIRED_TABLES = [
    "batiment_groupe_synthese_propriete_usage",
    "batiment_groupe_ffo_bat",
    "batiment_groupe",
    "batiment_construction",
]

# Tables jointes si présentes dans l'export, ignorées sinon : elles portent
# l'aléa et la valeur, indispensables à un modèle de coût de sinistre mais pas
# à la description du parc.
OPTIONAL_TABLES = [
    "batiment_groupe_risques",                  # aléa retrait-gonflement des argiles
    "batiment_groupe_dvf_open_representatif",   # valeur foncière (transactions DVF)
]


def discover_departements(root: Path) -> list[str]:
    """Codes des départements dont l'export CSV est présent sur le disque."""
    found = []
    for d in sorted(root.glob("dep_*/csv")):
        code = d.parent.name.removeprefix("dep_")
        if all((d / f"{t}.csv").exists() for t in REQUIRED_TABLES):
            found.append(code)
    return found


def extract_departement(con: duckdb.DuckDBPyConnection, code: str) -> str:
    """Construit la table ``maisons_<code>`` et renvoie son nom."""
    csv_dir = ROOT / f"dep_{code}" / "csv"
    missing = [t for t in REQUIRED_TABLES if not (csv_dir / f"{t}.csv").exists()]
    if missing:
        raise FileNotFoundError(f"dep {code}: tables manquantes {missing}")

    usage = _read_csv(csv_dir / "batiment_groupe_synthese_propriete_usage.csv")
    ffo = _read_csv(csv_dir / "batiment_groupe_ffo_bat.csv")
    groupe = _read_csv(csv_dir / "batiment_groupe.csv", COLS_BATIMENT_GROUPE)
    cstr = _read_csv(csv_dir / "batiment_construction.csv", COLS_BATIMENT_CONSTRUCTION)

    # Aléa argile (BRGM/Géorisques) : le retrait-gonflement des argiles est le
    # mécanisme même de la subsidence. Absent de l'export -> colonne nulle.
    risq_csv = csv_dir / "batiment_groupe_risques.csv"
    if risq_csv.exists():
        risq = f"(SELECT batiment_groupe_id, alea_argile FROM {_read_csv(risq_csv)})"
    else:
        risq = "(SELECT NULL AS batiment_groupe_id, NULL AS alea_argile WHERE false)"

    # DVF : valeur de la mutation représentative rattachée au groupe. On ne
    # retient comme prix exploitable que les mutations portant sur une maison
    # unique et aucun autre local - sinon la valeur mélange plusieurs biens.
    dvf_csv = csv_dir / "batiment_groupe_dvf_open_representatif.csv"
    if dvf_csv.exists():
        dvf = f"""(SELECT batiment_groupe_id, valeur_fonciere, prix_m2_local,
                          surface_bati_mutee_residencielle_individuelle AS surface_bati,
                          date_mutation, nb_maison_mutee_mutation, nb_locaux_mutee_mutation
                   FROM {_read_csv(dvf_csv)})"""
    else:
        dvf = """(SELECT NULL AS batiment_groupe_id, NULL AS valeur_fonciere,
                         NULL AS prix_m2_local, NULL AS surface_bati, NULL AS date_mutation,
                         NULL AS nb_maison_mutee_mutation, NULL AS nb_locaux_mutee_mutation
                  WHERE false)"""

    table = f"maisons_{code}"
    con.execute(f"DROP TABLE IF EXISTS {table}")
    con.execute(f"""
    CREATE TABLE {table} AS
    WITH maisons AS (
        -- Périmètre : groupes de bâtiments à usage principal de maison.
        SELECT batiment_groupe_id
        FROM {usage}
        WHERE usage_principal_bdnb_open = '{USAGE_MAISON}'
    ),
    constructions AS (
        -- Un groupe peut réunir plusieurs emprises physiques (corps de
        -- bâtiment, appentis) : on agrège hauteur et surface au sol.
        SELECT batiment_groupe_id,
               count(*)                                  AS nb_constructions,
               round(sum(s_geom_cstr), 1)                AS s_sol_constructions_m2,
               round(max(hauteur), 1)                    AS hauteur_max_m,
               round(sum(s_geom_cstr * hauteur) / nullif(sum(s_geom_cstr), 0), 1)
                                                         AS hauteur_ponderee_m,
               max(fictive_geom_cstr)                    AS geom_fictive,
               max(fictive_hauteur_cstr)                 AS hauteur_fictive
        FROM {cstr}
        GROUP BY batiment_groupe_id
    )
    SELECT
        m.batiment_groupe_id,
        g.code_departement_insee,
        g.code_commune_insee,
        g.libelle_commune_insee,
        g.code_iris,
        g.code_epci_insee,

        -- Variables cibles
        f.nb_niveau                                      AS nb_niveau,
        f.annee_construction                             AS annee_construction,
        round(g.s_geom_groupe, 1)                        AS surface_sol_m2,

        -- Emprise au sol vue par la BD TOPO (contrôle croisé de surface_sol_m2)
        c.s_sol_constructions_m2,
        c.nb_constructions,
        c.hauteur_max_m,
        c.hauteur_ponderee_m,

        -- Surface de plancher approchée = emprise au sol x nombre de niveaux.
        -- Estimation géométrique, à ne pas confondre avec la surface habitable.
        round(g.s_geom_groupe * f.nb_niveau, 1)          AS surface_plancher_estimee_m2,

        f.nb_log                                         AS nb_logements,
        f.mat_mur_txt                                    AS materiau_mur,
        f.mat_toit_txt                                   AS materiau_toit,
        f.usage_niveau_1_txt                             AS usage_ffo,

        -- Aléa retrait-gonflement des argiles (Faible / Moyen / Fort).
        -- C'est l'aléa qui conditionne l'éligibilité d'un bâtiment à
        -- indemnisation une fois la commune reconnue en état de catastrophe
        -- naturelle sécheresse.
        r.alea_argile                                    AS alea_argile,

        -- Valeur foncière observée (DVF). Ne couvre que les groupes ayant
        -- connu une mutation ; `dvf_mutation_simple` isole les transactions
        -- portant sur une maison seule, les seules dont le prix au m² est
        -- directement interprétable.
        v.valeur_fonciere                                AS dvf_valeur_fonciere,
        v.prix_m2_local                                  AS dvf_prix_m2,
        v.surface_bati                                   AS dvf_surface_bati_m2,
        v.date_mutation                                  AS dvf_date_mutation,
        (v.nb_maison_mutee_mutation = 1
            AND v.nb_locaux_mutee_mutation = 1)          AS dvf_mutation_simple,

        -- Indicateurs qualité : à filtrer selon l'usage aval.
        coalesce(g.contient_fictive_geom_groupe, 0) = 1
            OR coalesce(c.geom_fictive, 0) = 1           AS geometrie_fictive,
        coalesce(c.hauteur_fictive, 0) = 1               AS hauteur_fictive,
        (f.annee_construction IS NOT NULL
            AND f.annee_construction BETWEEN {ANNEE_MIN} AND {ANNEE_MAX})
            AND (g.s_geom_groupe BETWEEN {SURFACE_MIN} AND {SURFACE_MAX})
            AND f.nb_niveau BETWEEN 1 AND 4              AS plausible
    FROM maisons m
    JOIN {groupe} g USING (batiment_groupe_id)
    LEFT JOIN {ffo} f USING (batiment_groupe_id)
    LEFT JOIN constructions c USING (batiment_groupe_id)
    LEFT JOIN {risq} r USING (batiment_groupe_id)
    LEFT JOIN {dvf} v USING (batiment_groupe_id)
    """)
    return table


def report(con: duckdb.DuckDBPyConnection, table: str) -> None:
    """Contrôle qualité résumé d'une extraction départementale."""
    row = con.execute(f"""
        SELECT count(*)                                             AS maisons,
               count(nb_niveau)                                     AS a_nb_niveau,
               count(annee_construction)                            AS a_annee,
               count(surface_sol_m2)                                AS a_surface,
               count(*) FILTER (WHERE plausible)                    AS plausibles,
               median(nb_niveau)                                    AS med_niveau,
               median(annee_construction)                           AS med_annee,
               median(surface_sol_m2)                               AS med_surface
        FROM {table}
    """).fetchone()
    n = row[0]
    print(f"    maisons                : {n:>7,}")
    if not n:
        print("    (aucune maison extraite)")
        return
    pct = lambda v: f"{v:>7,} ({v / n:6.2%})"
    print(f"    nb_niveau renseigné    : {pct(row[1])}")
    print(f"    année renseignée       : {pct(row[2])}")
    print(f"    surface renseignée     : {pct(row[3])}")
    print(f"    lignes plausibles      : {pct(row[4])}")
    med = lambda v: "n/a" if v is None else f"{v:.0f}"
    print(f"    médianes               : {med(row[5])} niveaux, "
          f"{med(row[6])}, {med(row[7])} m²")


def aggregate(con: duckdb.DuckDBPyConnection, output_dir: Path) -> None:
    """Assemble le fichier régional depuis les CSV départementaux déjà extraits.

    L'agrégat se reconstruit à partir de ``output/``, pas des exports bruts :
    les exports BDNB étant volumineux, ils sont typiquement chargés puis
    supprimés un département à la fois, et ne coexistent jamais sur le disque.
    """
    parts = sorted(output_dir.glob("maisons_dep*.csv"))
    if len(parts) < 2:
        return
    # UNION ALL BY NAME : les exports ne sont pas tous extraits avec la même
    # version du script (des colonnes comme `alea_argile` ou les champs DVF ont
    # été ajoutées en cours de route). L'appariement par nom aligne ce qui
    # existe et met à NULL ce qui manque, au lieu d'échouer sur le décalage.
    union = " UNION ALL BY NAME ".join(
        f"SELECT * FROM read_csv('{f.as_posix()}', header=true)" for f in parts)
    out = output_dir / "maisons_grand_est.csv"
    con.execute(f"""COPY ({union} ORDER BY code_departement_insee,
                          code_commune_insee, batiment_groupe_id)
                    TO '{out}' (HEADER, DELIMITER ',')""")
    schemas = {}
    for f in parts:
        cols = {d[0] for d in con.execute(
            f"SELECT * FROM read_csv('{f.as_posix()}', header=true) LIMIT 0").description}
        schemas[f.name] = cols
    full = max(schemas.values(), key=len)
    stale = sorted(n for n, c in schemas.items() if c != full)

    rows = con.execute(f"""SELECT code_departement_insee, count(*) n
                           FROM ({union}) GROUP BY 1 ORDER BY 1""").fetchall()
    total = sum(r[1] for r in rows)
    couverts = ", ".join(r[0] for r in rows)
    manquants = [d for d in GRAND_EST if d not in {r[0] for r in rows}]
    print(f"\n[agrégat] {total:,} maisons sur {len(rows)} départements ({couverts})"
          f" -> {_display(out)}")
    for code, n in rows:
        print(f"    dep {code} : {n:>8,}")
    if manquants:
        print(f"    reste à traiter : {', '.join(manquants)}")
    if stale:
        missing = sorted(full - min(schemas.values(), key=len))
        print(f"    schéma périmé ({len(stale)} fichiers) : colonnes à NULL "
              f"{', '.join(missing)}")
        print(f"      -> relancer l'extraction de ces départements : "
              f"{", ".join(n[11:13] for n in stale)}")


def _display(path: Path) -> str:
    """Chemin relatif au projet quand c'est possible, absolu sinon."""
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dep", nargs="+", metavar="CODE",
                        help="codes département à traiter (défaut : tous ceux présents)")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--aggregate-only", action="store_true",
                        help="réassembler output/maisons_grand_est.csv depuis les "
                             "CSV départementaux déjà extraits, sans relire d'export")
    args = parser.parse_args()

    con_agg = duckdb.connect()
    if args.aggregate_only:
        aggregate(con_agg, args.output_dir)
        return 0

    available = discover_departements(ROOT)
    if not available:
        print(f"Aucun export BDNB trouvé sous {ROOT}/dep_*/csv/", file=sys.stderr)
        return 1

    deps = args.dep or available
    unknown = [d for d in deps if d not in available]
    if unknown:
        print(f"Export absent pour : {', '.join(unknown)} "
              f"(disponibles : {', '.join(available)})", file=sys.stderr)
        return 1

    args.output_dir.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    con.execute("SET preserve_insertion_order = false")

    for code in deps:
        print(f"[dep {code}]")
        table = extract_departement(con, code)
        report(con, table)
        out = args.output_dir / f"maisons_dep{code}.csv"
        con.execute(f"COPY (SELECT * FROM {table} ORDER BY code_commune_insee, "
                    f"batiment_groupe_id) TO '{out}' (HEADER, DELIMITER ',')")
        print(f"    -> {_display(out)}")
        con.execute(f"DROP TABLE {table}")

    aggregate(con, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
