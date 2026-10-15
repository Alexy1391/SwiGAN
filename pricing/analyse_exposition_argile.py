#!/usr/bin/env python3
"""Déplacement d'exposition : comptage communal vs aléa argile par bâtiment.

Compare, sur un département, l'exposition à la subsidence telle qu'un comptage
au niveau de la commune la restitue (l'approche de la section 6.2 de l'article)
et telle que l'aléa retrait-gonflement des argiles par bâtiment de la BDNB la
restitue.

Trois définitions communales sont confrontées à la définition par bâtiment :

    A0  toutes les maisons de la commune, sans filtre d'aléa
    A1  toutes les maisons des communes majoritairement en aléa Moyen/Fort
    A2  toutes les maisons des communes contenant de l'aléa Moyen/Fort
    B   les seules maisons dont l'aléa propre est Moyen ou Fort

Usage : python3 analyse_exposition_argile.py [--dep 08]
"""

import argparse
from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parent
ALEA_EXPOSE = ("Moyen", "Fort")


def prepare(con: duckdb.DuckDBPyConnection, csv: Path) -> None:
    """Charge les maisons et impute l'aléa manquant par la classe majoritaire."""
    con.execute(f"""
    CREATE OR REPLACE TABLE maisons AS
    WITH src AS (SELECT * FROM read_csv('{csv.as_posix()}', header=true)),
    -- 7 % des bâtiments n'ont pas d'aléa (hors emprise des cartes BRGM).
    -- On leur affecte la classe majoritaire de leur commune plutôt que de les
    -- écarter : les exclure biaiserait l'exposition à la baisse.
    majo AS (
        SELECT code_commune_insee, alea_argile,
               row_number() OVER (PARTITION BY code_commune_insee
                                  ORDER BY count(*) DESC, alea_argile) AS rk
        FROM src WHERE alea_argile IS NOT NULL
        GROUP BY 1, 2
    )
    SELECT s.*,
           coalesce(s.alea_argile, j.alea_argile)       AS alea,
           s.alea_argile IS NULL                        AS alea_impute
    FROM src s
    LEFT JOIN (SELECT code_commune_insee, alea_argile FROM majo WHERE rk = 1) j
           USING (code_commune_insee)
    """)

    # Valeur imputée : médiane communale des mutations portant sur une maison
    # seule, à défaut médiane départementale. On ne modélise pas le prix ici —
    # l'objectif est de porter le gradient spatial de valeur, pas de prédire un
    # prix individuel.
    con.execute("""
    CREATE OR REPLACE TABLE valeurs AS
    WITH obs AS (
        SELECT code_commune_insee, dvf_valeur_fonciere v
        FROM maisons
        WHERE dvf_mutation_simple AND dvf_valeur_fonciere BETWEEN 5000 AND 2000000
    ),
    parc AS (SELECT median(v) med FROM obs)
    SELECT code_commune_insee,
           median(v)          AS valeur_commune,
           count(*)           AS n_transactions
    FROM obs GROUP BY 1
    """)
    con.execute("""
    CREATE OR REPLACE TABLE m AS
    SELECT m.*,
           coalesce(v.valeur_commune,
                    (SELECT median(dvf_valeur_fonciere) FROM maisons
                     WHERE dvf_mutation_simple
                       AND dvf_valeur_fonciere BETWEEN 5000 AND 2000000)) AS valeur_est
    FROM maisons m LEFT JOIN valeurs v USING (code_commune_insee)
    """)


def commune_table(con: duckdb.DuckDBPyConnection) -> None:
    con.execute(f"""
    CREATE OR REPLACE TABLE com AS
    SELECT code_commune_insee                                   AS cc,
           any_value(libelle_commune_insee)                     AS lib,
           count(*)                                             AS n_total,
           count(*) FILTER (WHERE alea IN {ALEA_EXPOSE})        AS n_mf,
           count(*) FILTER (WHERE alea = 'Fort')                AS n_fort,
           sum(valeur_est)                                      AS v_total,
           sum(valeur_est) FILTER (WHERE alea IN {ALEA_EXPOSE}) AS v_mf
    FROM m GROUP BY 1
    """)


def displacement(con, num, denom_a, denom_b, label):
    """Part d'exposition déplacée entre communes entre deux définitions."""
    r = con.execute(f"""
    WITH a AS (SELECT cc, {denom_a} AS e FROM com),
         b AS (SELECT cc, {denom_b} AS e FROM com),
         ta AS (SELECT sum(e) t FROM a), tb AS (SELECT sum(e) t FROM b)
    SELECT (SELECT t FROM ta), (SELECT t FROM tb),
           sum(abs(a.e / (SELECT t FROM ta) - b.e / (SELECT t FROM tb))) / 2 * 100
    FROM a JOIN b USING (cc)""").fetchone()
    print(f"  {label:46s} {r[0]:>12,.0f} -> {r[1]:>12,.0f}  "
          f"({r[1]/r[0]-1:+7.1%})   déplacé {r[2]:5.1f} %")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dep", default="08")
    a = ap.parse_args()
    csv = ROOT / "output" / f"maisons_dep{a.dep}.csv"
    if not csv.exists():
        print(f"absent : {csv}"); return 1

    con = duckdb.connect()
    prepare(con, csv)
    commune_table(con)

    n, ni = con.execute("SELECT count(*), count(*) FILTER (WHERE alea_impute) FROM m").fetchone()
    print(f"\ndép {a.dep} — {n:,} maisons ({ni:,} aléas imputés par la classe majoritaire "
          f"communale, {ni/n:.1%})\n")

    print("EXPOSITION EN NOMBRE DE MAISONS")
    displacement(con, None, "n_total", "n_mf", "A0  toutes les maisons")
    displacement(con, None, "CASE WHEN n_mf*1.0/n_total>0.5 THEN n_total ELSE 0 END", "n_mf",
                 "A1  communes majoritairement argileuses")
    displacement(con, None, "CASE WHEN n_mf>0 THEN n_total ELSE 0 END", "n_mf",
                 "A2  communes contenant de l'argile")

    print("\nEXPOSITION EN VALEUR (€)")
    displacement(con, None, "v_total", "v_mf", "A0  toutes les maisons")
    displacement(con, None, "CASE WHEN n_mf*1.0/n_total>0.5 THEN v_total ELSE 0 END", "v_mf",
                 "A1  communes majoritairement argileuses")
    displacement(con, None, "CASE WHEN n_mf>0 THEN v_total ELSE 0 END", "v_mf",
                 "A2  communes contenant de l'argile")

    print("\nPROFIL DU BÂTI EXPOSÉ (aléa Moyen/Fort) vs non exposé")
    print(con.execute(f"""
      SELECT CASE WHEN alea IN {ALEA_EXPOSE} THEN 'exposé' ELSE 'non exposé' END AS groupe,
             count(*) maisons, round(median(surface_sol_m2),0) emprise_med,
             median(annee_construction) annee_med,
             round(count(*) FILTER (WHERE annee_construction<1976)*100.0/count(*),1) pct_av1976,
             round(median(valeur_est),0) valeur_med
      FROM m GROUP BY 1 ORDER BY 1""").df().to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
