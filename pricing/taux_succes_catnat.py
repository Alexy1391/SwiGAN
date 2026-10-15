#!/usr/bin/env python3
"""Part des demandes CatNat reconnues, lue dans les arrêtés publiés au Journal officiel.

Un arrêté « portant reconnaissance de l'état de catastrophe naturelle » range
chaque demande examinée — une commune, un phénomène, une période — dans l'une de
ses deux annexes (article 1) :

    annexe I    communes reconnues en état de catastrophe naturelle
    annexe II   communes non reconnues, demande rejetée

La part reconnue, I / (I + II), estime P(succès | demande déposée), la
probabilité g de la figure 5 (`--demande beta`). Elle est donnée par phénomène :
seule la ligne « Mouvements de terrain différentiels consécutifs à la sécheresse
et à la réhydratation des sols » concerne la subsidence. Le libellé « Mouvements
de terrains (hors sécheresse géotechnique) » contient aussi le mot sécheresse ;
un filtre sur ce seul mot mélangerait les deux.

DÉCISION ET CRITÈRES
--------------------
La décision est lue sur l'annexe. Pour la sécheresse, la colonne « Motivations »
détaille en plus chaque critère de la circulaire IOME2322937C : « Le critère
météorologique est satisfait » ou « n'est pas satisfait ». Une ligne de l'annexe I
doit avoir tous ses critères satisfaits, une ligne de l'annexe II au moins un non
satisfait ; le script le vérifie et signale les écarts. Les autres phénomènes
(inondations, mouvements de terrain, vents) ne portent pas ce libellé : compter
les lignes « satisfait » ne les classerait pas.

Jusqu'en avril 2022 les annexes sont en paragraphes, sans tableau ni motivation
(`decisions_paragraphes`). La décision se lit toujours sur l'annexe, mais les
colonnes de critères restent vides : l'arrêté ne dit pas quels critères ont été
appliqués. D'après les motivations publiées depuis, la circulaire INTE1911312C du
10 mai 2019 vaut pour les sécheresses de 2018 à 2023 et la circulaire
IOME2322937C pour celles de 2024 et après ; les sécheresses de 2017 et avant,
soit les arrêtés de 2018 et du début de 2019, relèvent des critères antérieurs.

SOURCE DES TEXTES
-----------------
Légifrance répond 403 aux requêtes automatiques (contrôle anti-robot). Le script
lit les mêmes textes dans les flux open data de la DILA
(https://echanges.dila.gouv.fr/OPENDATA/JORF/), au format XML : une archive
tar.gz par mise à jour, deux par jour environ, depuis le 13 juillet 2025. Les
textes antérieurs ne sont que dans l'archive globale (Freemium_jorf_global du
13 juillet 2025, 1,6 Go), lue seulement par `--annee` (voir plus bas).

Une URL Légifrance ne donne que l'identifiant JORFTEXT. Les identifiants
croissant avec le temps, une recherche dichotomique sur le plus grand
identifiant de chaque archive situe le texte en une dizaine de téléchargements
(0,1 à 7 Mo chacun), puis les archives voisines sont parcourues. Quand un texte
figure dans plusieurs archives, la plus récente l'emporte. Le contenu des
archives consultées est indexé dans `cache/jorf/index_flux.json`, et les XML du
texte sont conservés dans `cache/jorf/textes/` : un second appel ne télécharge
rien.

RECHERCHE PAR ANNÉE
-------------------
`--annee 2026` parcourt toutes les archives publiées à partir du 1er janvier de
l'année (environ 500 archives et 1,4 Go pour 2026, quelques minutes) et retient
les textes dont le titre complet est exactement « Arrêté du <date> portant
reconnaissance de l'état de catastrophe naturelle », signés dans l'année. Les
arrêtés modificatifs et les rectificatifs, dont le titre diffère, sont listés à
part et non comptés. Les titres trouvés sont notés par archive dans
`cache/jorf/catnat_flux.json` : un nouvel appel ne lit que les archives parues
depuis. Le tableau donne alors une ligne par arrêté.

Pour une année commencée avant les flux (2025 et avant), `--annee` lit aussi
l'archive globale, une seule fois, en flux continu (une dizaine de minutes,
rien n'est écrit sur disque hors cache). Elle garde tous les fichiers des textes
dont le titre cite la catastrophe naturelle, de toutes les années : article et
section portent le titre de leur texte (CONTEXTE/TITRE_TXT). Le catalogue de ces
textes est noté dans `cache/jorf/catnat_global.json`. Un fichier déjà présent
dans le cache n'est pas écrasé, la version des flux étant plus récente.

Usage :
    python3 taux_succes_catnat.py "https://www.legifrance.gouv.fr/loda/id/JORFTEXT000054435561?..."
    python3 taux_succes_catnat.py JORFTEXT000054435561 JORFTEXT000054245373 --csv decisions.csv
    python3 taux_succes_catnat.py --annee 2026
"""

import argparse
import difflib
import io
import json
import re
import tarfile
import urllib.request
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
DILA = "https://echanges.dila.gouv.fr/OPENDATA/JORF/"
CACHE_DEFAUT = ROOT / "cache" / "jorf"
SORTIE_DEFAUT = ROOT / "output" / "decisions_catnat.csv"
VOISINES = (3, 4)  # archives parcourues avant / après le point trouvé par dichotomie

ID_TEXTE = re.compile(r"JORFTEXT\d{12}")
FLUX = re.compile(r'href="(JORF_\d{8}-\d{6}\.tar\.gz)"')
GLOBALE = re.compile(r'href="(Freemium_jorf_global_\d{8}-\d{6}\.tar\.gz)"')
ELEMENT = re.compile(r"jorf/global/(article|section_ta|texte/struct|texte/version)/JORF/"
                     r"(?:ARTI|SCTA|TEXT)/(?:\d\d/){5}(JORF(?:ARTI|SCTA|TEXT)\d{12})\.xml$")
TITRE_TXT = re.compile(rb'<TITRE_TXT[^>]*id_txt="(JORFTEXT\d{12})"[^>]*>[^<]*catastrophe naturelle')
STRUCT_DANS_FLUX = re.compile(r"texte/struct/JORF/TEXT/(?:\d\d/){5}JORFTEXT(\d{12})\.xml$")
VERSION_DANS_FLUX = re.compile(r"texte/version/JORF/TEXT/(?:\d\d/){5}(JORFTEXT\d{12})\.xml$")
TITREFULL = re.compile(rb"<TITREFULL>([^<]*catastrophe naturelle[^<]*)</TITREFULL>")
DATE_TEXTE = re.compile(rb"<DATE_TEXTE>(\d{4}-\d\d-\d\d)</DATE_TEXTE>")
ARRETE_CATNAT = re.compile(r"Arrêté du \d{1,2}(?:er)? \w+ \d{4} "
                           r"portant reconnaissance de l'état de catastrophe naturelle")
TELECHARGEMENTS = 4  # archives téléchargées en parallèle par --annee
TITRE_ANNEXE = re.compile(r"communes (non )?reconnues", re.I)
# Le numéro prime sur le titre : l'annexe II de l'arrêté du 15 avril 2024 était intitulée
# « communes reconnues » (rectificatif JORFTEXT000049486473). L'annexe III de l'arrêté du
# 18 janvier 2024 (JO du 30) liste des communes reconnues.
NUMERO_ANNEXE = re.compile(r"annexe ((?-i:I{1,3}))(?!I)", re.I)  # « ANNEXE IICOMMUNES » après <br/>
CRITERE = re.compile(r"critère (géotechnique|météorologique)"
                     r"(?: fixé par la circulaire n° ?\w+ du (?:\d{1,2} \w+ \d{4}|[\d.]*\d))?"
                     r" (n'est pas|est) satisfait", re.I)
REUNIS = re.compile(r"critères météorologiques et géotechniques .*? sont réunis", re.I)  # circulaire de 2019
NON_SATISFAITS = re.compile(r"critères? (?:géotechniques? et météorologiques?|météorologiques?"
                            r" et géotechniques?) .*? ne sont pas satisfaits", re.I)  # arrêtés de 2024
SECHERESSE = re.compile(r"différentiels consécutifs à la sécheresse", re.I)

# Annexes en paragraphes, jusqu'en avril 2022
EN_TETE_ANNEXE = re.compile(r"annexes?(?: (I{1,3}))?", re.I)
TERRITOIRE = re.compile(r"(?:département|collectivité) (?:de la |de l' ?|des |du |d' ?|de )"
                        r"(.+?)\.?", re.I)
PERIODE = re.compile(r"(.*?)[ ,.]*\b(?:du|le) (\d{1,2}) ?(?:er)?(?: ?([^\W\d]+) ?(\d{4}))?"
                     r"(?: au (\d{1,2}) ?(?:er)? ?([^\W\d]+) (\d{4}))?\.?", re.I)
PREFIXE_COMMUNES = re.compile(r"Commu\w* (?:(?:de|du|des) |d' ?)?")
NOMBRE = re.compile(r"(.*?) ?(?:\(?(\d+)\))?")  # « Neuve-Grange (La) (1) », « Ceyreste 1) »
FAUTE_SECHERESSE = re.compile(r"(consécutifs à la) s\w+sse\b")  # « sécheesse », « séchresse »
MOIS = {m: i for i, m in enumerate(("janvier février mars avril mai juin juillet août septembre "
                                    "octobre novembre décembre").split(), 1)}
MINUSCULES = {"de", "du", "des", "d", "et", "la", "le"}  # dans un nom de département

COLONNES = {  # motif de l'en-tête -> nom de colonne
    r"département": "departement",
    r"commune": "commune",
    r"phénomène": "phenomene",
    r"début": "debut",
    r"\bfin\b": "fin",
    r"nombre": "nb_reconnaissances_5_ans",
    r"motivation": "motivation",
}
DOSSIERS = {"TEXT": "texte/struct", "SCTA": "section_ta", "ARTI": "article"}


def telecharger(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (taux_succes_catnat.py)"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return r.read()


def chemin(ident: str, version: bool = False) -> str:
    """Chemin d'un élément dans une archive DILA, sans le préfixe daté.

    JORFARTI000054435587 -> jorf/global/article/JORF/ARTI/00/00/54/43/55/JORFARTI000054435587.xml
    """
    num = ident[8:]
    dossier = "texte/version" if version else DOSSIERS[ident[4:8]]
    rep = "/".join(num[i:i + 2] for i in range(0, 10, 2))
    return f"jorf/global/{dossier}/JORF/{ident[4:8]}/{rep}/{ident}.xml"


def texte(el: ET.Element) -> str:
    return " ".join("".join(el.itertext()).replace("’", "'").split())


class Flux:
    """Archives incrémentales de la DILA, indexées par identifiant de texte."""

    def __init__(self, cache: Path):
        self.cache = cache
        self.fichier_index = cache / "index_flux.json"
        self.index = (json.loads(self.fichier_index.read_text())
                      if self.fichier_index.exists() else {})
        self.fichier_catnat = cache / "catnat_flux.json"
        self.catnat = (json.loads(self.fichier_catnat.read_text())
                       if self.fichier_catnat.exists() else {})
        self.fichier_global = cache / "catnat_global.json"
        self._liste = None
        self._archives = {}

    @property
    def liste(self) -> str:
        if self._liste is None:
            self._liste = telecharger(DILA).decode()
        return self._liste

    @property
    def noms(self) -> list[str]:
        return sorted(set(FLUX.findall(self.liste)))

    def archive(self, nom: str, donnees: bytes | None = None
                ) -> tuple[tarfile.TarFile, dict[str, tarfile.TarInfo]]:
        """Membres de l'archive, par chemin sans préfixe daté. Gardée en mémoire."""
        if nom not in self._archives:
            if donnees is None:
                print(f"  téléchargement {nom}")
                donnees = telecharger(DILA + nom)
            tar = tarfile.open(fileobj=io.BytesIO(donnees), mode="r:gz")
            self._archives[nom] = (tar, {m.name.split("/", 1)[-1]: m for m in tar.getmembers()})
        return self._archives[nom]

    def textes(self, nom: str) -> list[int]:
        if nom not in self.index:
            self.index[nom] = sorted(int(m.group(1)) for c in self.archive(nom)[1]
                                     if (m := STRUCT_DANS_FLUX.search(c)))
            self.fichier_index.parent.mkdir(parents=True, exist_ok=True)
            self.fichier_index.write_text(json.dumps(self.index))
        return self.index[nom]

    def localiser(self, ident: str) -> list[str]:
        """Archives contenant le texte, de la plus ancienne à la plus récente.

        Le plus grand identifiant d'une archive croît presque toujours d'une
        archive à la suivante ; les petits reculs sont couverts par les voisines.
        """
        num, noms = int(ident[8:]), self.noms
        if max(self.textes(noms[-1]), default=0) < num:
            raise LookupError(f"{ident} : absent des flux, pas encore publié ?")
        bas, haut = 0, len(noms) - 1
        while bas < haut:
            milieu = (bas + haut) // 2
            if max(self.textes(noms[milieu]), default=0) >= num:
                haut = milieu
            else:
                bas = milieu + 1
        voisines = noms[max(bas - VOISINES[0], 0): bas + VOISINES[1] + 1]
        trouvees = [n for n in voisines if num in self.textes(n)]
        if not trouvees:
            raise LookupError(f"{ident} : introuvable dans les flux depuis {noms[0][5:13]}. "
                              "Un texte plus ancien n'est que dans l'archive Freemium_jorf_global.")
        return trouvees

    def extraire(self, nom: str, ident: str) -> dict[str, bytes]:
        """Fichiers du texte présents dans une archive : struct, version, sections, articles."""
        tar, membres = self.archive(nom)
        lus, a_lire = {}, [chemin(ident), chemin(ident, version=True)]
        while a_lire:
            c = a_lire.pop()
            if c in lus or c not in membres:
                continue
            lus[c] = tar.extractfile(membres[c]).read()
            if "/version/" not in c:
                a_lire += [chemin(i) for i in liens(ET.fromstring(lus[c]))]
        return lus

    def liberer(self):
        self._archives.clear()

    def titres_catnat(self, nom: str) -> list[dict]:
        """Textes de l'archive dont le titre cite la catastrophe naturelle.

        Les arrêtés de reconnaissance retenus sont écrits dans le cache des
        textes ; une archive plus récente écrase les fichiers d'une plus ancienne.
        """
        tar, membres = self.archive(nom)
        trouves = []
        for c, m in membres.items():
            if not (v := VERSION_DANS_FLUX.search(c)):
                continue
            contenu = tar.extractfile(m).read()
            if not (t := TITREFULL.search(contenu)):
                continue
            titre = " ".join(t.group(1).decode().replace("’", "'").split())
            d = DATE_TEXTE.search(contenu)
            trouves.append({"id": v.group(1), "titre": titre,
                            "date_texte": d.group(1).decode() if d else ""})
            if ARRETE_CATNAT.fullmatch(titre):
                ecrire(self.cache / "textes" / v.group(1), self.extraire(nom, v.group(1)))
        return trouves

    def rechercher(self, annee: int) -> list[dict]:
        """Textes CatNat parus dans les archives depuis le 1er janvier de l'année.

        Une archive déjà parcourue n'est pas retéléchargée : ses titres sont
        relus dans `catnat_flux.json`. Un texte présent dans plusieurs archives
        garde la description de la plus récente.
        """
        noms = [n for n in self.noms if n[5:13] >= f"{annee}0101"]
        if not noms:
            raise LookupError(f"aucune archive depuis le 1er janvier {annee}")
        textes = {}
        if f"{annee}0101" < self.noms[0][5:13]:
            textes = {t["id"]: t for t in self.importer_global()}
        a_lire = [n for n in noms if n not in self.catnat]
        print(f"  {len(noms)} archives depuis le 1er janvier {annee}, {len(a_lire)} à lire")
        with ThreadPoolExecutor(TELECHARGEMENTS) as ex:
            en_cours = {}
            for i, nom in enumerate(a_lire):
                for n in a_lire[i:i + TELECHARGEMENTS]:
                    if n not in en_cours:
                        en_cours[n] = ex.submit(telecharger, DILA + n)
                self.archive(nom, en_cours.pop(nom).result())
                self.textes(nom)
                self.catnat[nom] = self.titres_catnat(nom)
                self.liberer()
                if (i + 1) % 50 == 0 or i + 1 == len(a_lire):
                    print(f"  {i + 1}/{len(a_lire)} archives lues ({nom[5:13]})")
                    self.fichier_catnat.write_text(json.dumps(self.catnat, ensure_ascii=False))
        textes |= {t["id"]: t for n in noms for t in self.catnat[n]}
        return sorted(textes.values(), key=lambda t: (t["date_texte"], t["id"]))

    def importer_global(self) -> list[dict]:
        """Textes CatNat de l'archive globale, de toutes les années, lue une fois.

        L'archive est lue en flux (mode r|gz) sans être enregistrée. Un article
        ou une section est rattaché à son texte par CONTEXTE/TITRE_TXT, qui porte
        le titre complet, et la version par TITREFULL. La structure ne porte pas
        le titre : elle est gardée si une section du texte a été vue, les
        sections précédant les structures dans l'archive. Un texte sans section
        n'a donc pas de structure ; ses articles sont alors lus par identifiant.
        """
        nom = sorted(GLOBALE.findall(self.liste))[-1]
        if self.fichier_global.exists():
            deja = json.loads(self.fichier_global.read_text())
            if deja["archive"] == nom:
                return deja["textes"]
        print(f"  lecture de {nom} en flux (1,6 Go, une dizaine de minutes)")
        req = urllib.request.Request(DILA + nom,
                                     headers={"User-Agent": "Mozilla/5.0 (taux_succes_catnat.py)"})
        textes, n_fichiers, avec_section = {}, 0, set()
        with urllib.request.urlopen(req, timeout=300) as r, \
                tarfile.open(fileobj=r, mode="r|gz") as tar:
            for i, m in enumerate(tar):
                tar.members.clear()  # sinon la liste des millions de membres reste en mémoire
                if i % 500_000 == 0 and i:
                    print(f"  {i:,} membres lus, {len(textes)} textes CatNat")
                if not m.isfile() or not (e := ELEMENT.search(m.name)):
                    continue
                texte_id = e.group(2)
                if e.group(1) == "texte/struct":
                    if texte_id not in avec_section:
                        continue
                    contenu = tar.extractfile(m).read()
                elif e.group(1) == "texte/version":
                    contenu = tar.extractfile(m).read()
                    if not (t := TITREFULL.search(contenu)):
                        continue
                    titre = " ".join(t.group(1).decode().replace("’", "'").split())
                    d = DATE_TEXTE.search(contenu)
                    textes[texte_id] = {"id": texte_id, "titre": titre,
                                        "date_texte": d.group(1).decode() if d else ""}
                else:
                    contenu = tar.extractfile(m).read()
                    if not (t := TITRE_TXT.search(contenu)):
                        continue
                    texte_id = t.group(1).decode()
                    if e.group(1) == "section_ta":
                        avec_section.add(texte_id)
                f = self.cache / "textes" / texte_id / m.name[m.name.index("jorf/global/"):]
                if not f.exists():
                    ecrire(f.parent, {f.name: contenu})
                    n_fichiers += 1
        print(f"  {len(textes)} textes CatNat, {len(textes.keys() - avec_section)} sans section, "
              f"{n_fichiers} fichiers écrits dans le cache")
        catalogue = sorted(textes.values(), key=lambda t: (t["date_texte"], t["id"]))
        self.fichier_global.parent.mkdir(parents=True, exist_ok=True)
        self.fichier_global.write_text(json.dumps({"archive": nom, "textes": catalogue},
                                                  ensure_ascii=False))
        return catalogue


def liens(racine: ET.Element) -> list[str]:
    """Articles et sections d'une structure de texte ou de section, dans l'ordre."""
    struct = racine.find("STRUCT")
    if struct is None:
        struct = racine.find("STRUCTURE_TA")
    if struct is None:
        return []
    return [e.get("id") for e in struct if e.tag in ("LIEN_ART", "LIEN_SECTION_TA")]


def recuperer(ident: str, cache: Path, flux: Flux) -> dict[str, bytes]:
    """Fichiers XML d'un texte, depuis le cache ou les flux DILA."""
    dossier = cache / "textes" / ident
    if dossier.is_dir():
        return {str(f.relative_to(dossier)): f.read_bytes() for f in dossier.rglob("*.xml")}
    fichiers = {}
    for nom in flux.localiser(ident):
        fichiers.update(flux.extraire(nom, ident))
    flux.liberer()
    ecrire(dossier, fichiers)
    return fichiers


def ecrire(dossier: Path, fichiers: dict[str, bytes]):
    for c, contenu in fichiers.items():
        (dossier / c).parent.mkdir(parents=True, exist_ok=True)
        (dossier / c).write_bytes(contenu)


def articles_dans_l_ordre(ident: str, fichiers: dict[str, bytes]) -> list[ET.Element]:
    """Articles du texte dans l'ordre de lecture, annexes comprises.

    Sans structure (texte sans section de l'archive globale), les articles
    présents sont pris par identifiant croissant, l'ordre de leur création.
    """
    if chemin(ident) not in fichiers:
        arti = sorted(c for c in fichiers if "/article/" in c)
        if not arti:
            raise LookupError(f"{ident} : ni structure ni article dans le cache")
        print(f"  {ident} : pas de structure, {len(arti)} articles lus par identifiant")
        return [ET.fromstring(fichiers[c]) for c in arti]
    sortie, manquants = [], []

    def parcourir(i: str):
        c = chemin(i)
        if c not in fichiers:
            manquants.append(i)
        elif i.startswith("JORFARTI"):
            sortie.append(ET.fromstring(fichiers[c]))
        else:
            for j in liens(ET.fromstring(fichiers[c])):
                parcourir(j)

    parcourir(ident)
    if manquants:
        raise LookupError(f"{ident} : éléments absents des archives : {', '.join(manquants)}")
    return sortie


def decisions(articles: list[ET.Element]) -> tuple[list[dict], int]:
    """Une ligne par demande des annexes I et II, et le nombre de lignes illisibles.

    L'annexe courante est fixée par le dernier paragraphe « Communes reconnues »
    ou « Communes non reconnues » rencontré, hors tableau. La ligne d'en-tête
    est en <th> ou en <td> selon les arrêtés : elle se reconnaît à sa cellule
    « Commune ».
    """
    lignes, illisibles, annexe = [], 0, None
    for art in articles:
        contenu = art.find("BLOC_TEXTUEL/CONTENU")
        if contenu is None:
            continue
        en_tableau = {e for t in contenu.iter("table") for e in t.iter()}
        colonnes = None
        for el in contenu.iter():
            if el.tag == "p" and el not in en_tableau and (m := TITRE_ANNEXE.search(texte(el))):
                n = NUMERO_ANNEXE.search(texte(el))
                annexe = n.group(1).upper() if n else "II" if m.group(1) else "I"
            elif el.tag == "tr":
                cellules = [texte(c) for c in el if c.tag in ("th", "td")]
                if not cellules:
                    continue
                if any(c.tag == "th" for c in el) or any(c.lower() == "commune" for c in cellules):
                    colonnes = [next((nom for motif, nom in COLONNES.items()
                                      if re.search(motif, e, re.I)), None) for e in cellules]
                    continue
                if annexe is None or colonnes is None or len(cellules) != len(colonnes):
                    illisibles += 1
                    continue
                ligne = {"annexe": annexe, "reconnue": annexe != "II"}
                ligne |= {nom: v for nom, v in zip(colonnes, cellules) if nom}
                ligne |= criteres(ligne.get("motivation", ""))
                lignes.append(ligne)
    return lignes, illisibles


def lignes_de(p: ET.Element) -> list[str]:
    """Texte d'un paragraphe, coupé aux <br/>."""
    sortie, courant = [], [p.text or ""]

    def parcourir(el: ET.Element):
        for e in el:
            if e.tag == "br":
                sortie.append("".join(courant))
                courant.clear()
            else:
                courant.append(e.text or "")
                parcourir(e)
            courant.append(e.tail or "")

    parcourir(p)
    sortie.append("".join(courant))
    return [t for ligne in sortie if (t := " ".join(ligne.replace("’", "'").split()))]


def departement(nom: str) -> str:
    """« ALPES-DE-HAUTE-PROVENCE » -> « Alpes-de-Haute-Provence », comme dans les tableaux."""
    nom = re.sub(r"[^\W\d]+", lambda m: m.group() if m.group() in MINUSCULES
                 else m.group().capitalize(), nom.lower().replace(" '", "'"))
    return {"Réunion": "La Réunion"}.get(nom, nom[:1].upper() + nom[1:])


def jour(j: str, mois: str, annee: str) -> str:
    """« 1er », « arvil », « 2017 » -> « 01/04/2017 » ; le mois peut porter une faute de frappe."""
    proche = difflib.get_close_matches(mois.lower(), MOIS, n=1, cutoff=0.7)
    if not proche:
        raise ValueError(mois)
    return f"{int(j):02d}/{MOIS[proche[0]]:02d}/{annee}"


def communes(liste: str) -> list[tuple[str, str]]:
    """Communes d'une liste et leur nombre de reconnaissances sur cinq ans.

    « Communes de Lucciana, Valle-di-Rostino et Vescovato. » : le « et » ne sépare
    deux communes qu'à la fin d'une liste au pluriel. Ailleurs il fait partie du
    nom (« Commune de Mont Lozère et Goulet », « Bassillac et Auberoche, Bergerac »).
    """
    noms = re.split(r" ?[,;] ?|\. ", PREFIXE_COMMUNES.sub("", liste, count=1).rstrip(" ."))
    if liste.startswith("Communes"):
        noms[-1:] = noms[-1].split(" et ")
    return [(nom, nombre or "") for nom, nombre in (NOMBRE.fullmatch(n).groups() for n in noms)
            if nom]


def decisions_paragraphes(articles: list[ET.Element]) -> tuple[list[dict], int, int]:
    """Demandes d'un arrêté aux annexes en paragraphes, lignes illisibles et lignes d'annexe III.

    Jusqu'en avril 2022 les annexes n'ont ni tableau ni motivation. Chaque ligne,
    entre deux <br/>, est un département, un phénomène suivi de sa période, ou
    une liste de communes ; le nombre de reconnaissances sur cinq ans suit la
    commune entre parenthèses :

        DÉPARTEMENT DE L'AIN
        Inondations et coulées de boue du 10 mai 2021 au 11 mai 2021
        Communes de Ceyzériat (1), Saint-Jean-le-Vieux.

    Une « ANNEXE » sans numéro vaut annexe I. L'annexe III modifie des arrêtés
    antérieurs : elle n'est pas lue, comme les rectificatifs. Les fautes de
    frappe sur « sécheresse » sont corrigées, le libellé servant de filtre.
    """
    lignes, illisibles, modifs = [], 0, 0
    annexe = dep = phen = debut = fin = None
    numerotee, attente = False, ""
    for art in articles:
        contenu = art.find("BLOC_TEXTUEL/CONTENU")
        if contenu is None:
            continue
        en_tableau = {e for t in contenu.iter("table") for e in t.iter()}
        hors_tableau = (p for p in contenu.iter("p") if p not in en_tableau)
        for ligne in (ligne for p in hors_tableau for ligne in lignes_de(p)):
            if m := EN_TETE_ANNEXE.fullmatch(ligne):
                annexe, numerotee = (m.group(1) or "I").upper(), bool(m.group(1))
                dep = phen = None
                illisibles, attente = illisibles + bool(attente), ""
            elif annexe is None:
                continue
            elif annexe == "III":
                modifs += 1
            elif m := re.fullmatch(r"communes (non )?reconnues[^,;]*", ligne, re.I):
                if not numerotee:
                    annexe = "II" if m.group(1) else "I"
            elif m := TERRITOIRE.fullmatch(ligne):
                dep, phen = departement(m.group(1)), None
                illisibles, attente = illisibles + bool(attente), ""
            elif (not PREFIXE_COMMUNES.match(ligne)
                  and (m := PERIODE.fullmatch(f"{attente} {ligne}".strip()))):
                attente = ""
                nom, j1, m1, a1, j2, m2, a2 = m.groups()
                try:  # « du 19 au 20 septembre 2020 » : mois et année pris à la fin
                    debut = jour(j1, m1 or m2, a1 or a2)
                    fin = jour(j2, m2, a2) if j2 else debut
                except (AttributeError, ValueError):
                    phen, illisibles = None, illisibles + 1
                    continue
                phen = FAUTE_SECHERESSE.sub(r"\1 sécheresse", nom) or phen
            elif PREFIXE_COMMUNES.match(ligne) or ligne.endswith("."):
                if dep is None or phen is None:
                    illisibles += 1
                    continue
                lignes += [{"annexe": annexe, "reconnue": annexe != "II", "departement": dep,
                            "commune": commune, "phenomene": phen, "debut": debut, "fin": fin,
                            "nb_reconnaissances_5_ans": nombre, "criteres": ""}
                           for commune, nombre in communes(ligne)]
            else:  # phénomène coupé avant sa période, complété par la ligne suivante
                illisibles, attente = illisibles + bool(attente), ligne
    return lignes, illisibles + bool(attente), modifs


def criteres(motivation: str) -> dict:
    """État des critères géotechnique et météorologique cités dans la motivation."""
    etats = {"critere_" + c.lower().replace("é", "e"):
             "pas satisfait" if "pas" in e else "satisfait"
             for c, e in CRITERE.findall(motivation)}
    if REUNIS.search(motivation):
        etats |= {"critere_geotechnique": "satisfait", "critere_meteorologique": "satisfait"}
    if NON_SATISFAITS.search(motivation):
        etats |= {"critere_geotechnique": "pas satisfait", "critere_meteorologique": "pas satisfait"}
    etats["criteres"] = ("pas satisfait" if "pas satisfait" in etats.values()
                         else "satisfait" if etats else "")
    return etats


def ecarts_criteres(df: pd.DataFrame) -> pd.Series:
    """Lignes sécheresse dont les critères « satisfait » contredisent l'annexe.

    Sans objet pour les annexes en paragraphes, qui n'ont pas de motivation.
    """
    if "motivation" not in df:
        return pd.Series(False, index=df.index)
    attendu = df["reconnue"].map({True: "satisfait", False: "pas satisfait"})
    return (df["phenomene"].str.contains(SECHERESSE) & df["motivation"].notna()
            & (df["criteres"] != attendu))


def resume(df: pd.DataFrame, par: str) -> pd.DataFrame:
    """Demandes reconnues et rejetées par groupe, et part reconnue."""
    t = df.groupby(par)["reconnue"].agg(reconnues="sum", demandes="size")
    t.loc["Total"] = t.sum()
    t = t.astype(int)
    t["rejetees"] = t["demandes"] - t["reconnues"]
    t["part_reconnue"] = t["reconnues"] / t["demandes"]
    return t[["reconnues", "rejetees", "demandes", "part_reconnue"]]


def afficher(titre: str, df: pd.DataFrame):
    """Part reconnue par phénomène, puis par année de sécheresse, et contrôle des critères.

    Un arrêté peut statuer sur plusieurs années de sécheresse : la part est
    donnée pour chacune, l'année étant celle du début de la période demandée.
    """
    print(f"\n{titre}")
    t = resume(df, "phenomene")
    t = pd.concat([t.iloc[:-1].sort_values("demandes", ascending=False), t.iloc[-1:]])
    court = [p if len(p) <= 58 else p[:57] + "…" for p in t.index]
    largeur = max(map(len, court)) + 2
    print(f"  {'phénomène':<{largeur}}{'reconnues':>10}{'rejetées':>10}{'part':>8}")
    for r, nom in zip(t.itertuples(), court):
        marque = "→ " if SECHERESSE.search(r.Index) else "  "
        print(f"{marque}{nom:<{largeur}}{r.reconnues:>10}{r.rejetees:>10}{r.part_reconnue:>8.1%}")

    sech = df[df["phenomene"].str.contains(SECHERESSE)]
    if sech.empty:
        return
    par_an = resume(sech.assign(annee=sech["debut"].str[-4:]), "annee")
    if len(par_an) > 2:
        print("  sécheresse par année : " + " · ".join(
            f"{r.Index} {r.reconnues}/{r.demandes} ({r.part_reconnue:.1%})"
            for r in par_an.iloc[:-1].itertuples()))
    if "motivation" not in sech or sech["motivation"].isna().all():
        print("  contrôle « satisfait » (sécheresse) : sans objet, annexes sans motivation")
        return
    ecarts = sech[ecarts_criteres(sech)]
    print(f"  contrôle « satisfait » (sécheresse) : "
          f"{(sech['criteres'] == 'satisfait').sum()} lignes satisfait, "
          f"{(sech['criteres'] == 'pas satisfait').sum()} lignes pas satisfait, "
          f"{len(ecarts)} en désaccord avec l'annexe")
    for _, r in ecarts.iterrows():
        print(f"    annexe {r.annexe} · {r.departement} · {r.commune} : "
              f"critères « {r.criteres or 'non cités'} »")


def par_arrete(df: pd.DataFrame):
    """Une ligne par arrêté : sécheresse reconnue / examinée, puis tous phénomènes.

    « écarts » compte les lignes sécheresse dont les critères « satisfait »
    contredisent l'annexe ; « déjà vues » celles qu'un arrêté précédent de la
    liste avait déjà examinées (même commune, phénomène et période).
    """
    cle = ["departement", "commune", "phenomene", "debut", "fin"]
    df = df.sort_values(["date_texte", "texte"], kind="stable")
    df = df.assign(sech=df["phenomene"].str.contains(SECHERESSE),
                   deja_vue=df.duplicated(cle), ecart=ecarts_criteres(df))
    print(f"\n  {'arrêté':<26}{'JO':>11}  {'sécheresse':>14}{'part':>8}  {'années':<10}"
          f"{'tous phénomènes':>16}{'part':>8}{'écarts':>8}{'déjà vues':>10}")
    for (texte, titre), g in df.groupby(["texte", "arrete"], sort=False):
        s = g[g["sech"]]
        annees = ",".join(sorted(s["debut"].str[-4:].unique()))
        sech = (f"{s['reconnue'].sum():>5} / {len(s):<5}{s['reconnue'].mean():>8.1%}"
                if len(s) else f"{'—':>14}{'':>8}")
        print(f"  {titre:<26}{g['date_publication'].iloc[0]:>11}  {sech}  {annees:<10}"
              f"{g['reconnue'].sum():>7} / {len(g):<6}{g['reconnue'].mean():>8.1%}"
              f"{g['ecart'].sum():>8}{g['deja_vue'].sum():>10}")
    s = df[df["sech"]]
    print(f"  {'Total':<26}{'':>11}  {s['reconnue'].sum():>5} / {len(s):<5}"
          f"{s['reconnue'].mean():>8.1%}  {'':<10}{df['reconnue'].sum():>7} / {len(df):<6}"
          f"{df['reconnue'].mean():>8.1%}{df['ecart'].sum():>8}{df['deja_vue'].sum():>10}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("textes", nargs="*",
                    help="URL Légifrance ou identifiant JORFTEXT d'un arrêté CatNat")
    ap.add_argument("--annee", type=int,
                    help="tous les arrêtés de reconnaissance signés cette année-là")
    ap.add_argument("--cache", type=Path, default=CACHE_DEFAUT,
                    help="index des flux et XML des textes (défaut : cache/jorf)")
    ap.add_argument("--csv", type=Path, default=SORTIE_DEFAUT,
                    help="une ligne par demande (défaut : output/decisions_catnat.csv)")
    a = ap.parse_args()
    if not a.textes and a.annee is None:
        ap.error("donner un arrêté (URL ou JORFTEXT) ou --annee")

    flux = Flux(a.cache)
    idents = []
    for arg in a.textes:
        if not (m := ID_TEXTE.search(arg)):
            ap.error(f"pas d'identifiant JORFTEXT dans « {arg} »")
        idents.append(m.group())
    if a.annee is not None:
        print(f"Recherche des arrêtés CatNat de {a.annee} dans les flux DILA")
        try:
            trouves = flux.rechercher(a.annee)
        except LookupError as e:
            print(e)
            return 1
        retenus = [t for t in trouves if ARRETE_CATNAT.fullmatch(t["titre"])
                   and t["date_texte"].startswith(str(a.annee))]
        print(f"  {len(retenus)} arrêtés de reconnaissance signés en {a.annee}")
        for t in trouves:
            if t not in retenus and t["date_texte"].startswith(str(a.annee)):
                print(f"  non compté : {t['id']} · {t['titre']}")
        idents += [t["id"] for t in retenus if t["id"] not in idents]
    detail = a.annee is None

    tables = []
    for ident in idents:
        try:
            fichiers = recuperer(ident, a.cache, flux)
            articles = articles_dans_l_ordre(ident, fichiers)
        except LookupError as e:
            print(f"\n{e}")
            continue
        meta = ET.fromstring(fichiers[chemin(ident, version=True)])
        titre = meta.findtext(".//TITREFULL") or meta.findtext(".//TITRE") or ident
        publi = meta.findtext(".//DATE_PUBLI")
        lignes, illisibles = decisions(articles)
        forme = "de tableau ignorées, annexe ou en-têtes non reconnus"
        if not lignes:
            lignes, illisibles, modifs = decisions_paragraphes(articles)
            forme = "d'annexe ignorées, phénomène, période ou département non reconnus"
            if modifs:
                print(f"  {ident} : annexe III non lue, {modifs} lignes modifiant "
                      "des arrêtés antérieurs")
        if not lignes:
            print(f"\n{ident} · {titre} : aucune ligne d'annexe I ou II")
            continue
        df = pd.DataFrame(lignes).assign(
            texte=ident, arrete=meta.findtext(".//TITRE") or ident,
            date_texte=meta.findtext(".//DATE_TEXTE"), date_publication=publi)
        if detail:
            afficher(f"{ident} · {titre} · JO du {publi}", df)
        if illisibles:
            print(f"  {ident} : {illisibles} lignes {forme}")
        tables.append(df)

    if not tables:
        return 1
    tout = pd.concat(tables, ignore_index=True)
    if len(tables) > 1:
        par_arrete(tout)
        afficher(f"Ensemble des {len(tables)} arrêtés", tout)
        print("  (une même demande examinée par deux arrêtés y compte deux fois)")
    # Mêmes colonnes pour toutes les années, y compris sans critères ni motivation.
    premieres = ["texte", "arrete", "date_texte", "date_publication", "annexe", "reconnue",
                 "departement", "commune",
                 "phenomene", "debut", "fin", "criteres", "critere_geotechnique",
                 "critere_meteorologique", "nb_reconnaissances_5_ans", "motivation"]
    tout = tout.reindex(columns=premieres + [c for c in tout if c not in premieres])
    a.csv.parent.mkdir(parents=True, exist_ok=True)
    tout.to_csv(a.csv, sep=";", index=False)
    print(f"\n{len(tout)} demandes -> {a.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
