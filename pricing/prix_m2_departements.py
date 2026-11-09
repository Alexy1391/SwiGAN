#!/usr/bin/env python3
"""Prix au m² des logements par département, depuis les fichiers DVF bruts.

Source : pricing/dvf/ValeursFoncieres-*.txt (format DGFiP, séparateur « | »).
Une ligne DVF n'est ni une vente ni un local : c'est le produit cartésien
(local × nature de culture) d'une parcelle d'une disposition. D'où :

Mutation    pas d'identifiant dans le fichier brut -> clé composite
            (date, valeur foncière, département, commune, n° de disposition).
Locaux      un même local est répété une fois par nature de culture de sa
            parcelle. L'identifiant local étant masqué, on compte les locaux
            d'un groupe (parcelle, type, surface, pièces) comme
            nb lignes / nb natures de culture distinctes de la parcelle,
            ce qui préserve les locaux réellement identiques.
Prix        une seule valeur foncière par mutation (elle est répétée sur
            chaque ligne).

Seules les mutations de type « Vente » contenant exactement un logement
(maison ou appartement), éventuellement avec des dépendances, et aucun local
commercial sont retenues : le prix d'une vente multi-logements ne se ventile
pas entre les biens. La VEFA (neuf) est exclue.

Prix au m² = valeur foncière / surface réelle bâtie du logement. Pour une
maison, la valeur inclut le terrain (convention usuelle des indices notariaux).

Les extrêmes sont écrêtés par département × type (p1–p99 du prix au m²) avant
de calculer moyenne et médiane.

Absents de DVF : Bas-Rhin (67), Haut-Rhin (68), Moselle (57) — régime du Livre
foncier — et Mayotte (976).

Usage : python3 prix_m2_departements.py [--par-annee]
"""

import argparse
from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parent
SRC = (ROOT / "dvf" / "ValeursFoncieres-*.txt").as_posix()
OUT = ROOT / "output"


def mutations(con: duckdb.DuckDBPyConnection) -> None:
    """Table `ventes` : une ligne par vente d'un logement unique."""
    con.execute(f"""
        CREATE TEMP TABLE lignes AS
        SELECT "Date mutation"                                   AS date_mut,
               "Valeur fonciere"                                 AS valeur_txt,
               "Code departement"                                AS dep,
               "Code commune"                                    AS com,
               "No disposition"                                  AS dispo,
               coalesce("Prefixe de section", '') || '|' || coalesce("Section", '')
                   || '|' || coalesce("No plan", '')             AS parcelle,
               "Code type local"                                 AS type_local,
               try_cast("Surface reelle bati" AS DOUBLE)         AS surface,
               "Nombre pieces principales"                       AS pieces,
               "Nature culture"                                  AS nc
        FROM read_csv('{SRC}', delim='|', header=true, all_varchar=true)
        WHERE "Nature mutation" = 'Vente'
    """)
    cle = "date_mut, valeur_txt, dep, com, dispo"
    con.execute(f"""
        CREATE TEMP TABLE ventes AS
        WITH nat AS (       -- natures de culture distinctes par parcelle
            SELECT {cle}, parcelle, greatest(count(DISTINCT nc), 1) AS n_nc
            FROM lignes GROUP BY ALL
        ),
        locaux AS (         -- un groupe = des locaux indiscernables
            SELECT {cle}, parcelle, type_local, surface, pieces,
                   count(*) AS n_lignes
            FROM lignes WHERE type_local IS NOT NULL GROUP BY ALL
        ),
        locaux_n AS (
            SELECT l.*, round(n_lignes / n_nc) AS n_locaux
            FROM locaux l JOIN nat USING ({cle}, parcelle)
        ),
        mut AS (
            SELECT {cle},
                   sum(n_locaux) FILTER (type_local IN ('1', '2')) AS n_logements,
                   sum(n_locaux) FILTER (type_local = '4')         AS n_commerces,
                   any_value(type_local) FILTER (type_local IN ('1', '2')) AS type_local,
                   sum(n_locaux * surface) FILTER (type_local IN ('1', '2')) AS surface
            FROM locaux_n GROUP BY ALL
        )
        SELECT dep,
               -- Code INSEE : DVF code la commune sur 3 chiffres, sauf en
               -- outre-mer où le département en prend 3 et la commune 2.
               CASE WHEN length(dep) = 3 THEN dep || lpad(com, 2, '0')
                    ELSE dep || lpad(com, 3, '0') END                AS insee,
               year(strptime(date_mut, '%d/%m/%Y'))                  AS annee,
               CASE type_local WHEN '1' THEN 'Maison' ELSE 'Appartement' END AS type_local,
               replace(valeur_txt, ',', '.')::DOUBLE                 AS valeur,
               surface,
               valeur / surface                                      AS prix_m2
        FROM mut
        WHERE n_logements = 1 AND coalesce(n_commerces, 0) = 0
          AND valeur_txt IS NOT NULL
    """)
    con.execute("""
        DELETE FROM ventes
        WHERE NOT (valeur >= 1000 AND surface BETWEEN 9 AND 1000)
    """)


def agreger(con: duckdb.DuckDBPyConnection, groupes: str) -> "pd.DataFrame":
    """Moyenne et médiane du prix au m² après écrêtage p1–p99 par groupe."""
    return con.execute(f"""
        WITH bornes AS (
            SELECT {groupes},
                   quantile_cont(prix_m2, 0.01) AS p01,
                   quantile_cont(prix_m2, 0.99) AS p99
            FROM ventes GROUP BY ALL
        )
        SELECT {groupes},
               count(*)                          AS n_ventes,
               round(avg(prix_m2))               AS prix_m2_moyen,
               round(median(prix_m2))            AS prix_m2_median,
               round(quantile_cont(prix_m2, 0.25)) AS prix_m2_q1,
               round(quantile_cont(prix_m2, 0.75)) AS prix_m2_q3,
               round(median(surface))            AS surface_mediane
        FROM ventes JOIN bornes USING ({groupes})
        WHERE prix_m2 BETWEEN p01 AND p99
        GROUP BY ALL ORDER BY ALL
    """).df()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--par-annee", action="store_true",
                   help="ajoute une table département × type × année")
    args = p.parse_args()

    con = duckdb.connect()
    mutations(con)
    n = con.execute("SELECT count(*) FROM ventes").fetchone()[0]
    print(f"{n:,} ventes d'un logement unique retenues")

    OUT.mkdir(exist_ok=True)
    # Tous logements confondus : chaque vente est dupliquée sous le type
    # « Ensemble », écrêté et agrégé comme un type à part entière.
    con.execute("INSERT INTO ventes SELECT * REPLACE ('Ensemble' AS type_local) FROM ventes")
    res = agreger(con, "dep, type_local")
    res.to_csv(OUT / "prix_m2_departements.csv", index=False)
    print(f"-> {OUT / 'prix_m2_departements.csv'}")

    if args.par_annee:
        res_a = agreger(con, "dep, type_local, annee")
        res_a.to_csv(OUT / "prix_m2_departements_annee.csv", index=False)
        print(f"-> {OUT / 'prix_m2_departements_annee.csv'}")

    print(res.pivot(index="dep", columns="type_local", values="prix_m2_moyen")
             .astype("Int64").to_string())


if __name__ == "__main__":
    main()
