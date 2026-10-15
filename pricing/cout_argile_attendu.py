#!/usr/bin/env python3
"""Coût attendu de la subsidence par commune, via la probabilité de reconnaissance.

Chaîne complète, dans l'ordre :

    1.  p_k        probabilité qu'une maison d'aléa argile k (Faible / Moyen /
                   Fort / Inconnu) déclenche une reconnaissance CatNat, ajustée
                   par maximum de vraisemblance sur les commune-années éligibles
                   du département (Ardennes : 1 784 observations, 77 reconnues)

    2.  Λ_c,t  =  λ_t · Σ_k p_k · n_ck            intensité de la commune c

    3.  P(reconnue | éligible)  =  1 − exp(−Λ_c,t)

    4.  coût  =  E[nombre de sinistres] × coût moyen d'un sinistre (16 500 € par
                 défaut, moyenne France)

L'étape 4 est le maillon faible et le script l'expose sous deux lectures, parce
que l'ajustement ne voit qu'un **binaire communal** (reconnue ou non) et ne peut
donc pas identifier le nombre de sinistres à l'intérieur d'une commune reconnue :

    --sinistres modele   N = Λ_c,t.  C'est ce que le modèle implique
                         littéralement, soit ≈ 1 sinistre par commune reconnue.
                         C'est un PLANCHER, pas une estimation.

    --sinistres taux     N = P(reconnue) × τ × n_c, avec τ le taux de maisons
                         sinistrées dans une commune reconnue, calibré HORS
                         modèle (défaut 2,25 %, valeur implicite de l'équation
                         6.1 de l'article : 371,5 €/maison ÷ 16 500 €).

--demande beta sépare la DEMANDE du maire de son succès (voir section 2 bis) :
P(reconnue) = E[θ_c | historique] × (1 − exp(−Λ)), θ_c ~ Bêta(mκ, (1−m)κ),
avec m = 15 à 30 % (littérature, non identifiable sans les demandes rejetées)
et κ estimé sur les reconnaissances répétées. Une demande passée relève la
probabilité de redéposer : (mκ + A)/(κ + E).

Sortie : output/cout_argile_<dep>.csv (une ligne par commune-année éligible),
         output/cout_argile_<dep>_beta.csv et, avec --eligibilite-futur,
         output/projection_argile_<dep>.csv

Usage : python3 cout_argile_attendu.py [--dep 08] [--sinistres taux]
                                       [--cout-moyen 16500] [--taux 0.0225]
                                       [--simulations 20000]
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import logsumexp, roots_jacobi
from scipy.stats import chi2, norm, rankdata

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "output"

CLASSES = ["Faible", "Moyen", "Fort", "Inconnu"]
ANNEES_FIT = [2018, 2019, 2020, 2022]      # années sous le régime de critère actuel
ANNEE_REF = 2022                            # λ = 1 par convention : « année type 2022 »

COUT_MOYEN_DEFAUT = 16_500.0                # € par sinistre, moyenne France
TAUX_DEFAUT = 0.0225                        # 371,5 € / 16 500 € (équation 6.1)
SIGMA_LOG_DEFAUT = 0.9                      # dispersion de la sévérité (lognormale)

TAUX_DEMANDE_DEFAUT = [0.15, 0.225, 0.30]   # part des maires qui déposent (littérature)
NOEUDS = 64                                 # quadrature de Gauss-Jacobi sur θ
KAPPA_MIN, KAPPA_MAX = 0.02, 500.0          # au-delà, Gauss-Jacobi déborde (2^(α+β))


# --------------------------------------------------------------------------
# 1. Assemblage : une ligne par commune-année éligible
# --------------------------------------------------------------------------
def charge(dep: str) -> pd.DataFrame:
    """Croise parc de maisons, éligibilité SWI et reconnaissances CatNat."""
    m = pd.read_csv(OUT / f"maisons_dep{dep}.csv", low_memory=False,
                    dtype={"code_commune_insee": str})
    m = m[m["usage_ffo"] == "Résidentiel individuel"]
    # L'aléa manquant garde sa propre classe : l'imputer par la classe
    # majoritaire de la commune (ce que fait cout_subsidence_futur.py) lui
    # prêterait un p_k que l'ajustement exclut à 95 %.
    m["cls"] = m["alea_argile"].fillna("Inconnu")
    n = (m.pivot_table(index="code_commune_insee", columns="cls",
                       values="batiment_groupe_id", aggfunc="count")
           .reindex(columns=CLASSES).fillna(0.0))

    e = pd.read_csv(OUT / f"eligibilite_{dep}_historique.csv",
                    dtype={"code_commune_insee": str})
    e = e[e["eligible"] & e["annee"].isin(ANNEES_FIT)].copy()

    c = pd.read_csv(OUT / "catnat_secheresse_argiles.csv", sep=";", dtype=str)
    c = c[c["departement"] == dep].drop_duplicates(
        subset=["code_geographique", "type_catastrophe",
                "date_debut_corrigee", "date_fin_corrigee"])
    c["annee"] = pd.to_datetime(c["date_debut_corrigee"]).dt.year
    reconnues = set(zip(c["code_geographique"], c["annee"]))

    e["reconnue"] = [int((k, a) in reconnues)
                     for k, a in zip(e["code_commune_insee"], e["annee"])]
    e = e.join(n, on="code_commune_insee")
    e[CLASSES] = e[CLASSES].fillna(0.0)
    e["n_maisons"] = e[CLASSES].sum(axis=1)

    # Historique de la commune avant l'année courante, sur les années du régime
    # actuel. Une reconnaissance est une demande certaine ; une année éligible
    # sans reconnaissance reste ambiguë (pas de demande, ou demande rejetée).
    e = e.sort_values(["code_commune_insee", "annee"]).reset_index(drop=True)
    g = e.groupby("code_commune_insee")
    e["E_prec"] = g.cumcount()
    e["A_prec"] = g["reconnue"].cumsum() - e["reconnue"]
    return e


# --------------------------------------------------------------------------
# 2. Ajustement de p_k et λ_t
# --------------------------------------------------------------------------
def _negll(theta, X, y, D, ref, ncls):
    p = np.exp(theta[:ncls])
    lam = np.ones(D.shape[1])
    lam[[i for i in range(D.shape[1]) if i != ref]] = np.exp(theta[ncls:])
    base = X[:, :ncls] if ncls > 1 else X.sum(1, keepdims=True)
    Lam = (D @ lam) * (base @ p)
    P = np.clip(-np.expm1(-Lam), 1e-12, 1 - 1e-12)
    return -(y * np.log(P) + (1 - y) * np.log1p(-P)).sum()


def _annees(e: pd.DataFrame):
    """Années, indice de l'année de référence (λ = 1) et indicatrices d'année."""
    annees = sorted(e["annee"].unique())
    # hors 2022 (validation en avant), la dernière année sert de référence
    ref = annees.index(ANNEE_REF) if ANNEE_REF in annees else len(annees) - 1
    D = np.column_stack([(e["annee"] == a).to_numpy(float) for a in annees])
    return annees, ref, D


def ajuste(e: pd.DataFrame) -> dict:
    """Maximum de vraisemblance sur le lien complementary log-log sans constante."""
    X = e[CLASSES].to_numpy(float)
    y = e["reconnue"].to_numpy(float)
    annees, ref, D = _annees(e)

    def run(ncls):
        t0 = np.r_[np.full(ncls, np.log(3e-4)), np.zeros(len(annees) - 1)]
        return minimize(_negll, t0, args=(X, y, D, ref, ncls), method="BFGS",
                        options={"maxiter": 20000, "gtol": 1e-10})

    plein, nul = run(len(CLASSES)), run(1)
    lam = np.ones(len(annees))
    lam[[i for i in range(len(annees)) if i != ref]] = np.exp(plein.x[len(CLASSES):])
    se = np.sqrt(np.diag(plein.hess_inv))
    z = norm.ppf(0.975)
    # La classe Inconnu est au bord (p -> 0) : son erreur-type de Wald diverge.
    # L'intervalle de profil vaut [~0 ; 1,8e-4], soit Faible ou moins.
    with np.errstate(over="ignore"):
        return {
            "p": np.exp(plein.x[:len(CLASSES)]),
            "p_bas": np.exp(plein.x[:len(CLASSES)] - z * se[:len(CLASSES)]),
            "p_haut": np.exp(plein.x[:len(CLASSES)] + z * se[:len(CLASSES)]),
            "lam": dict(zip(annees, lam)),
            "negll": plein.fun,
            "lr": 2 * (nul.fun - plein.fun),
            "ddl": len(CLASSES) - 1,
        }


# --------------------------------------------------------------------------
# 2 bis. Demande du maire : prior bêta mis à jour par l'historique
# --------------------------------------------------------------------------
#
#   θ_c ~ Bêta(mκ, (1−m)κ)           propension du maire à déposer une demande
#   F_ct | θ_c ~ Bernoulli(θ_c)       il dépose, l'année t où la commune est éligible
#   R_ct | F_ct = 1 ~ Bernoulli(g_ct) la demande aboutit, g = 1 − exp(−λ_t Σ_k p_k n_ck)
#
# Sachant A demandes sur E années éligibles passées, la probabilité de déposer
# vaut (mκ + A) / (κ + E) (urne de Pólya) : chaque demande la relève. Les
# demandes ne sont pas observées (catnat.csv ne liste que les reconnaissances),
# d'où la vraisemblance intégrée sur θ_c par commune. m se confond avec
# l'échelle de λ·p et n'est pas identifiable : il est fixé (15–30 %,
# littérature). κ est estimé sur les reconnaissances répétées. κ → ∞ rend le
# modèle indépendant, P = m·g, soit le modèle ci-dessus à l'échelle près.

def noeuds_beta(a: float, b: float, n: int = NOEUDS):
    """Nœuds θ_j et poids (somme 1) de la quadrature de Gauss-Jacobi pour Bêta(a, b)."""
    # roots_jacobi pondère par (1−x)^α (1+x)^β sur [−1, 1] ; θ = (1+x)/2
    x, w = roots_jacobi(n, b - 1.0, a - 1.0)
    return (1.0 + x) / 2.0, w / w.sum()


def _succes(theta, X, D, ref):
    """g_ct = 1 − exp(−λ_t Σ_k p_k n_ck) pour des paramètres log (p_k, λ_t≠ref)."""
    ncls, nan = X.shape[1], D.shape[1]
    lam = np.ones(nan)
    lam[[i for i in range(nan) if i != ref]] = np.exp(theta[ncls:ncls + nan - 1])
    return -np.expm1(-(D @ lam) * (X @ np.exp(theta[:ncls])))


def _log_vrais_noeuds(g, y, idx, ncom, th):
    """log P(historique de la commune | θ_j) : matrice communes × nœuds."""
    q = np.clip(g[:, None] * th[None, :], 1e-300, 1 - 1e-12)
    ll = np.where(y[:, None] == 1, np.log(q), np.log1p(-q))
    out = np.zeros((ncom, len(th)))
    np.add.at(out, idx, ll)
    return out


def _kappa(t):
    return float(np.exp(np.clip(t, np.log(KAPPA_MIN), np.log(KAPPA_MAX))))


def _negll_beta(theta, X, y, D, ref, idx, ncom, m, kappa):
    g = _succes(theta, X, D, ref)
    if kappa is None:
        kappa = _kappa(theta[-1])
    if np.isinf(kappa):                        # pas d'effet d'historique : P = m g
        P = np.clip(m * g, 1e-12, 1 - 1e-12)
        return -(y * np.log(P) + (1 - y) * np.log1p(-P)).sum()
    th, w = noeuds_beta(m * kappa, (1 - m) * kappa)
    L = _log_vrais_noeuds(g, y, idx, ncom, th) + np.log(np.maximum(w, 1e-300))
    return -logsumexp(L, axis=1).sum()


def ajuste_beta(e: pd.DataFrame, m: float, kappa: float | None = None,
                depart: dict | None = None) -> dict:
    """Maximum de vraisemblance de (p_k, λ_t, κ) à m fixé ; κ fixé si fourni."""
    X = e[CLASSES].to_numpy(float)
    y = e["reconnue"].to_numpy(float)
    annees, ref, D = _annees(e)
    idx, communes = pd.factorize(e["code_commune_insee"])
    if depart is None:                         # départ : modèle indépendant, p / m
        f0 = ajuste(e)
        lam0 = [np.log(f0["lam"][a]) for i, a in enumerate(annees) if i != ref]
        depart = {"x": np.r_[np.log(np.maximum(f0["p"], 1e-9) / m), lam0,
                             np.log(10.0)]}
    t0 = depart["x"] if kappa is None else depart["x"][:len(CLASSES) + len(annees) - 1]
    res = minimize(_negll_beta, t0, method="BFGS",
                   args=(X, y, D, ref, idx, len(communes), m, kappa),
                   options={"maxiter": 20000, "gtol": 1e-8})
    k = _kappa(res.x[-1]) if kappa is None else kappa
    lam = np.ones(len(annees))
    lam[[i for i in range(len(annees)) if i != ref]] = np.exp(
        res.x[len(CLASSES):len(CLASSES) + len(annees) - 1])
    return {"m": m, "kappa": k, "p": np.exp(res.x[:len(CLASSES)]),
            "lam": dict(zip(annees, lam)), "negll": res.fun,
            "x": res.x if kappa is None else np.r_[res.x, np.log(min(k, KAPPA_MAX))]}


def profil_kappa(e: pd.DataFrame, m: float, fit: dict,
                 grille=(0.1, 0.3, 1, 3, 10, 30, 100, 500)) -> pd.DataFrame:
    """Vraisemblance profilée de κ (p_k et λ_t réajustés à chaque κ)."""
    lignes = [(k, ajuste_beta(e, m, kappa=k, depart=fit)["negll"])
              for k in (*grille, np.inf)]
    t = pd.DataFrame(lignes, columns=["kappa", "negll"])
    t["dev"] = 2 * (t["negll"] - fit["negll"])
    return t


def posterieur(e: pd.DataFrame, fit: dict, e_cible: pd.DataFrame | None = None):
    """Loi a posteriori de θ_c sur les nœuds.

    Renvoie (θ_j, poids finaux communes × nœuds, index des communes, et, par
    ligne de e_cible (défaut e), E[θ_c | historique strictement antérieur]).
    """
    X = e[CLASSES].to_numpy(float)
    y = e["reconnue"].to_numpy(float)
    g = e["annee"].map(fit["lam"]).to_numpy(float) * (X @ fit["p"])
    g = -np.expm1(-g)
    k = min(fit["kappa"], KAPPA_MAX)
    th, w = noeuds_beta(fit["m"] * k, (1 - fit["m"]) * k)
    lw = np.log(np.maximum(w, 1e-300))
    q = np.clip(g[:, None] * th[None, :], 1e-300, 1 - 1e-12)
    ll = pd.DataFrame(np.where(y[:, None] == 1, np.log(q), np.log1p(-q)))
    ll["c"], ll["a"] = e["code_commune_insee"].to_numpy(), e["annee"].to_numpy()

    total = ll.groupby("c").sum(numeric_only=True).drop(columns="a")
    post = total.to_numpy() + lw
    post = np.exp(post - logsumexp(post, axis=1, keepdims=True))
    communes = total.index

    cible = e if e_cible is None else e_cible
    # historique strictement antérieur : somme des lignes de la commune avant a
    avant = {c: (grp["a"].to_numpy(), grp.drop(columns=["c", "a"]).to_numpy())
             for c, grp in ll.groupby("c")}
    lp = []
    for c, a in zip(cible["code_commune_insee"], cible["annee"]):
        s = lw.copy()
        if c in avant:
            ans, mat = avant[c]
            s = s + mat[ans < a].sum(axis=0)
        s = np.exp(s - logsumexp(s))
        lp.append(s @ th)
    return th, pd.DataFrame(post, index=communes), np.array(lp)


def auc(score, y):
    """Aire sous la courbe ROC (statistique de Mann-Whitney)."""
    y = np.asarray(y, bool)
    r = rankdata(score)
    n1, n0 = y.sum(), (~y).sum()
    return (r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0) if n1 and n0 else np.nan


def valide_en_avant(e: pd.DataFrame, m: float) -> pd.DataFrame:
    """Prédit l'année cible avec les seules années antérieures, avec et sans historique.

    λ de l'année cible est inconnu : l'AUC (classement intra-année) n'en dépend
    presque pas ; le score logarithmique est calculé après calage de λ_cible
    sur l'effectif observé, identiquement pour les deux modèles.
    """
    lignes = []
    for cible in ANNEES_FIT[2:]:                 # 2020 et 2022
        app, test = e[e["annee"] < cible], e[e["annee"] == cible]
        X, y = test[CLASSES].to_numpy(float), test["reconnue"].to_numpy(float)
        f0 = ajuste(app)
        fb = ajuste_beta(app, m)
        _, _, prior = posterieur(app, fb, test)
        scores = {"indépendant": (np.ones(len(test)), X @ f0["p"]),
                  "historique": (prior, X @ fb["p"])}
        for nom, (pi, base) in scores.items():
            def P(lam):
                return np.clip(pi * -np.expm1(-lam * base), 1e-12, 1 - 1e-12)
            # λ_cible calé pour que l'effectif attendu égale l'observé
            lo, hi = 1e-4, 1e3
            for _ in range(200):
                mid = np.sqrt(lo * hi)
                lo, hi = (mid, hi) if P(mid).sum() < y.sum() else (lo, mid)
            p = P(lo)
            lignes.append({"cible": cible, "modele": nom, "reconnues": int(y.sum()),
                           "auc": auc(p, y),
                           "log_score": (y * np.log(p) + (1 - y) * np.log1p(-p)).sum()})
    return pd.DataFrame(lignes)


# --------------------------------------------------------------------------
# 3-4. Probabilité de reconnaissance, sinistres, coût
# --------------------------------------------------------------------------
def chaine(e: pd.DataFrame, fit: dict, lecture: str, cout_moyen: float,
           taux: float, p_depot: np.ndarray | None = None) -> pd.DataFrame:
    """Ajoute Λ, P(reconnue), l'espérance de sinistres et le coût attendu.

    p_depot : E[θ_c | historique antérieur] (modèle « beta ») ; absent, la
    demande n'est pas séparée et P(reconnue) = 1 − exp(−Λ) comme avant.
    """
    r = e.copy()
    lam_t = r["annee"].map(fit["lam"]).to_numpy(float)
    r["lambda_annee"] = lam_t
    r["Lambda"] = lam_t * (r[CLASSES].to_numpy(float) @ fit["p"])
    r["p_reconnue"] = -np.expm1(-r["Lambda"])           # 1 − exp(−Λ)
    if p_depot is not None:
        r["p_succes"] = r["p_reconnue"]
        r["p_depot"] = p_depot
        r["p_reconnue"] = r["p_depot"] * r["p_succes"]

    if lecture == "modele":
        r["n_sinistres"] = r["Lambda"]
    else:
        r["n_sinistres"] = r["p_reconnue"] * taux * r["n_maisons"]
    r["cout_attendu"] = r["n_sinistres"] * cout_moyen
    return r


def simule(r: pd.DataFrame, lecture: str, cout_moyen: float, taux: float,
           sigma_log: float, tirages: int, graine: int = 0) -> np.ndarray:
    """Distribution du coût annuel par Monte-Carlo (reconnaissance × fréquence × sévérité)."""
    rng = np.random.default_rng(graine)
    p_rec = r["p_reconnue"].to_numpy()
    # sévérité lognormale de moyenne cout_moyen -> mu = log(moyenne) - sigma²/2
    mu = np.log(cout_moyen) - sigma_log ** 2 / 2
    totaux = np.empty(tirages)
    for s in range(tirages):
        rec = rng.random(len(r)) < p_rec                # la commune est-elle reconnue ?
        if not rec.any():
            totaux[s] = 0.0
            continue
        if lecture == "modele":
            # sachant reconnue, Poisson tronquée à >= 1
            lam_c = r["Lambda"].to_numpy()[rec]
            k = 1 + rng.poisson(lam_c)
        else:
            k = rng.poisson(taux * r["n_maisons"].to_numpy()[rec])
        n = int(k.sum())
        totaux[s] = rng.lognormal(mu, sigma_log, n).sum() if n else 0.0
    return totaux


def projette(communes: pd.DataFrame, fit: dict, post: tuple | None,
             elig: pd.DataFrame, cout_moyen: float, taux: float, sigma_log: float,
             tirages: int, graine: int = 0) -> tuple[pd.DataFrame, np.ndarray]:
    """Coût annuel simulé sur des trajectoires d'éligibilité futures.

    communes : n_ck et n_maisons par code INSEE. post : (θ_j, poids communes ×
    nœuds) pour le modèle « beta », None pour le modèle indépendant. λ = 1
    (année type 2022). θ_c est tiré une fois par trajectoire et par tirage :
    les années d'une trajectoire partagent la propension du maire, ce qui
    revient à mettre à jour l'urne (mκ + A)/(κ + E) après chaque demande simulée.
    Renvoie une ligne par (trajectoire, année) et tous les coûts annuels tirés.
    """
    rng = np.random.default_rng(graine)
    g = -np.expm1(-(communes[CLASSES].to_numpy(float) @ fit["p"]))
    n = communes["n_maisons"].to_numpy(float)
    code = {c: i for i, c in enumerate(communes.index)}
    if post is not None:
        th, w = post
        cdf = np.cumsum(w.reindex(communes.index).to_numpy(), axis=1)
    mu = np.log(cout_moyen) - sigma_log ** 2 / 2
    lignes, tous = [], []
    for traj, gt in elig.groupby("trajectoire"):
        theta = np.ones((tirages, len(communes)))
        if post is not None:
            u = rng.random((tirages, len(communes)))
            for i in range(len(communes)):
                j = np.searchsorted(cdf[i], u[:, i] * cdf[i, -1])
                theta[:, i] = th[np.minimum(j, len(th) - 1)]
        deja = np.zeros((tirages, len(communes)), bool)
        for an, ga in gt.sort_values("annee").groupby("annee"):
            ic = np.array([code[c] for c in ga["commune"] if c in code], dtype=int)
            F = rng.random((tirages, len(ic))) < theta[:, ic]           # le maire dépose
            R = F & (rng.random((tirages, len(ic))) < g[ic])           # la demande aboutit
            k = rng.poisson(taux * n[ic], size=R.shape) * R
            nk = k.sum(axis=1)
            sev = rng.lognormal(mu, sigma_log, int(nk.sum()))
            cout = np.bincount(np.repeat(np.arange(tirages), nk), weights=sev,
                               minlength=tirages)
            recid = (R & deja[:, ic]).sum(axis=1)
            deja[:, ic] |= R
            tous.append(cout)
            lignes.append({"trajectoire": traj, "annee": an, "eligibles": len(ic),
                           "reconnues": R.sum(axis=1).mean(),
                           "deja_reconnues": recid.mean(),
                           "cout_moyen": cout.mean(),
                           "cout_p99": np.percentile(cout, 99)})
    return pd.DataFrame(lignes), np.concatenate(tous) if tous else np.zeros(0)


# --------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dep", default="08")
    ap.add_argument("--sinistres", choices=["modele", "taux"], default="taux",
                    help="lecture du nombre de sinistres (voir docstring)")
    ap.add_argument("--cout-moyen", type=float, default=COUT_MOYEN_DEFAUT)
    ap.add_argument("--taux", type=float, default=TAUX_DEFAUT,
                    help="part des maisons sinistrées dans une commune reconnue")
    ap.add_argument("--sigma-log", type=float, default=SIGMA_LOG_DEFAUT)
    ap.add_argument("--simulations", type=int, default=20_000)
    ap.add_argument("--demande", choices=["independante", "beta"], default="independante",
                    help="« beta » sépare la demande du maire, prior bêta mis à jour "
                         "par l'historique de la commune")
    ap.add_argument("--taux-demande", type=float, nargs="+", default=TAUX_DEMANDE_DEFAUT,
                    help="m, part des maires qui déposent (la valeur médiane sert "
                         "à la chaîne détaillée)")
    ap.add_argument("--kappa", type=float, default=None,
                    help="fixe la concentration κ du prior au lieu de l'estimer")
    ap.add_argument("--eligibilite-futur", type=Path, default=None,
                    help="table d'éligibilité projetée (trajectoire, annee, commune)")
    ap.add_argument("--tirages-futur", type=int, default=2_000,
                    help="tirages Monte-Carlo par trajectoire projetée")
    a = ap.parse_args()

    e = charge(a.dep)
    fit = ajuste(e)

    print(f"Ajustement — {len(e)} commune-années éligibles, "
          f"{int(e['reconnue'].sum())} reconnues")
    print(f"{'aléa':9} {'maisons':>9} {'p (année type 2022)':>22} {'IC 95 %':>24}")
    for i, cl in enumerate(CLASSES):
        print(f"{cl:9} {e[cl].sum() / len(fit['lam']):9,.0f} "
              f"{fit['p'][i]:22.2e} "
              f"[{fit['p_bas'][i]:8.2e}, {fit['p_haut'][i]:8.2e}]")
    print(f"λ : " + ", ".join(f"{k} {v:.2f}" for k, v in fit["lam"].items()))
    print(f"LR classe unique = {fit['lr']:.1f} / {fit['ddl']} ddl "
          f"(p = {chi2.sf(fit['lr'], fit['ddl']):.1e})")

    fits_beta = {}
    if a.demande == "beta":
        print("\nDemande du maire — θ_c ~ Bêta(mκ, (1−m)κ), "
              "P(déposer | A demandes sur E années) = (mκ + A)/(κ + E)")
        rep = e[e["A_prec"] > 0]
        print(f"  déjà reconnues avant : {int(rep['reconnue'].sum())} reconnaissances "
              f"sur {len(rep)} commune-années ; modèle indépendant : "
              f"{chaine(rep, fit, a.sinistres, a.cout_moyen, a.taux)['p_reconnue'].sum():.1f}")
        print(f"{'m':>6} {'κ':>8} {'IC 95 % profil':>18} {'LR histoire':>12} {'p':>8}   "
              + "  ".join(f"p_{cl[:3]}×1e4" for cl in CLASSES))
        for m in a.taux_demande:
            f = ajuste_beta(e, m, kappa=a.kappa)
            pr = profil_kappa(e, m, f) if a.kappa is None else None
            if pr is not None:
                dans = pr.loc[pr["dev"] <= chi2.ppf(0.95, 1), "kappa"]
                ic = f"[{dans.min():.2g} ; {dans.max():.2g}]"
                # κ = ∞ est au bord : loi ½χ²(0) + ½χ²(1)
                lr = max(pr["dev"].iloc[-1], 0.0)
                pv = 0.5 * chi2.sf(lr, 1)
            else:
                ic, lr, pv = "fixé", np.nan, np.nan
            fits_beta[m] = (f, pr)
            print(f"{m:6.3f} {f['kappa']:8.2f} {ic:>18} {lr:12.2f} {pv:8.3f}   "
                  + "  ".join(f"{v * 1e4:10.2f}" for v in f["p"]))
        if a.kappa is None:
            m0 = sorted(a.taux_demande)[len(a.taux_demande) // 2]
            print(f"\nProfil de κ (m = {m0}) : "
                  + ", ".join(f"κ={k:g} Δdev {d:.2f}"
                              for k, d in zip(fits_beta[m0][1]["kappa"],
                                              fits_beta[m0][1]["dev"])))
        m0 = sorted(a.taux_demande)[len(a.taux_demande) // 2]
        fb = fits_beta[m0][0]
        _, _, p_dep = posterieur(e, fb)

        print(f"\nValidation en avant (m = {m0}) : années antérieures seules")
        v = valide_en_avant(e, m0)
        print(v.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
        fit_chaine = fb
    else:
        p_dep, fit_chaine = None, fit

    r = chaine(e, fit_chaine, a.sinistres, a.cout_moyen, a.taux, p_depot=p_dep)
    lib = ("N = Λ" if a.sinistres == "modele"
           else f"N = P(reconnue) × {a.taux:.2%} × maisons")
    print(f"\nChaîne — lecture « {a.sinistres} » ({lib}), "
          f"coût moyen {a.cout_moyen:,.0f} €\n")
    print(f"{'année':>6} {'λ':>5} {'communes':>9} {'reconnues':>10} {'attendues':>10} "
          f"{'sinistres':>10} {'coût attendu':>14}")
    for an, g in r.groupby("annee"):
        print(f"{an:>6} {g['lambda_annee'].iloc[0]:5.2f} {len(g):9d} "
              f"{int(g['reconnue'].sum()):10d} {g['p_reconnue'].sum():10.1f} "
              f"{g['n_sinistres'].sum():10.1f} "
              f"{g['cout_attendu'].sum() / 1e6:11.2f} M€")

    # Pour mémoire : la section 6.2 facture TOUTES les communes éligibles, sans
    # filtre de reconnaissance. Même τ, même coût moyen, sans l'étape 3.
    brut = r.groupby("annee")["n_maisons"].sum() * a.taux * a.cout_moyen
    fin = r.groupby("annee")["cout_attendu"].sum()
    print("\nPour mémoire — lecture section 6.2 (toutes les communes éligibles) :")
    for an in fin.index:
        print(f"{an:>6} {brut[an] / 1e6:8.2f} M€  soit "
              f"{brut[an] / fin[an]:5.1f}× le coût ci-dessus")
    if a.demande == "beta":
        # correction de la section 6.2 par la seule littérature : × m
        print("  corrigée par la part des maires qui déposent (× m) : "
              + "  ".join(f"m={m:.3f} → {brut.mean() * m / 1e6:.2f} M€"
                          for m in sorted(a.taux_demande)))
        print("\nReconnaissances chez les communes déjà reconnues (observé / attendu) :")
        for an, g in r[r["A_prec"] > 0].groupby("annee"):
            print(f"{an:>6} {int(g['reconnue'].sum()):4d} / {g['p_reconnue'].sum():5.1f}"
                  f"   (P(déposer) moyen {g['p_depot'].mean():.3f}, autres communes "
                  f"{r.loc[(r['annee'] == an) & (r['A_prec'] == 0), 'p_depot'].mean():.3f})")

    print(f"\nDistribution du coût annuel ({a.simulations:,} tirages, "
          f"sévérité lognormale σ={a.sigma_log})")
    print(f"{'année':>6} {'moyenne':>11} {'P50':>11} {'P90':>11} {'P99':>11}")
    for an, g in r.groupby("annee"):
        t = simule(g, a.sinistres, a.cout_moyen, a.taux, a.sigma_log, a.simulations)
        q = np.percentile(t, [50, 90, 99])
        print(f"{an:>6} {t.mean() / 1e6:8.2f} M€ {q[0] / 1e6:8.2f} M€ "
              f"{q[1] / 1e6:8.2f} M€ {q[2] / 1e6:8.2f} M€")

    suffixe = "_beta" if a.demande == "beta" else ""
    dest = OUT / f"cout_argile_{a.dep}{suffixe}.csv"
    cols = ["code_commune_insee", "annee", *CLASSES, "n_maisons", "lambda_annee",
            "Lambda", "p_reconnue", "n_sinistres", "cout_attendu", "reconnue"]
    if a.demande == "beta":
        cols += ["E_prec", "A_prec", "p_depot", "p_succes"]
    r[cols].to_csv(dest, index=False)
    print(f"\n-> {dest.relative_to(ROOT)}  ({len(r)} lignes)")

    if a.eligibilite_futur is not None:
        from cout_subsidence_futur import charger_eligibilite
        elig = charger_eligibilite(a.eligibilite_futur, (1900, 2200))
        communes = e.groupby("code_commune_insee")[[*CLASSES, "n_maisons"]].first()
        print(f"\nProjection — {a.eligibilite_futur.name} : {elig['trajectoire'].nunique()} "
              f"trajectoires, {elig['annee'].min()}–{elig['annee'].max()}, λ = 1 "
              f"(année type 2022), {a.tirages_futur:,} tirages par trajectoire")
        modeles = {"indépendant": (fit, None)}
        for m, (f, _) in sorted(fits_beta.items()):
            th_m, w_m, _ = posterieur(e, f)
            modeles[f"historique m={m:.3f}"] = (f, (th_m, w_m))
        # « dont déjà » : part des reconnaissances d'une commune déjà reconnue
        # plus tôt dans la même trajectoire
        print(f"{'modèle':>22} {'reconnues/an':>13} {'dont déjà':>10} "
              f"{'coût moyen':>11} {'P99 annuel':>11}")
        sorties = []
        for nom, (f, pst) in modeles.items():
            t, couts = projette(communes, f, pst, elig, a.cout_moyen, a.taux,
                                a.sigma_log, a.tirages_futur)
            t.insert(0, "modele", nom)
            sorties.append(t)
            print(f"{nom:>22} {t['reconnues'].mean():13.2f} "
                  f"{t['deja_reconnues'].sum() / max(t['reconnues'].sum(), 1e-12):10.1%} "
                  f"{couts.mean() / 1e6:8.2f} M€ {np.percentile(couts, 99) / 1e6:8.2f} M€")
        dest = OUT / f"projection_argile_{a.dep}.csv"
        pd.concat(sorties).to_csv(dest, index=False)
        print(f"-> {dest.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
