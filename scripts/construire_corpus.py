"""Construit un corpus de VRAI code Python, reduit au vocabulaire de la mouche.

Contrainte infranchissable : un jeton = un MBON, et le circuit n'en a que 48. On ne
peut donc pas elargir le vocabulaire -- il faut REDUIRE le code. Les identifiants
deviennent x/y/n/i/f, les nombres 0/1/2, la structure syntaxique reste intacte.
C'est cette structure que le corps pedoncule peut apprendre : « apres `def f (`
vient un nom », « apres `:` NL vient IND ».

Source : la bibliotheque standard de Python installee sur la machine. Du vrai code,
ecrit par des humains, pas quatorze patrons ecrits a la main.

    python scripts/construire_corpus.py --max 600
"""
from __future__ import annotations
import argparse
import io
import json
import random
import sys
import tokenize
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from droso.corpus import ID, VOCAB          # noqa: E402
from droso.config import ROOT               # noqa: E402

NOMS = ["f", "x", "y", "n", "i"]            # identifiants disponibles, dans l'ordre
MOTS = {"def", "return", "if", "else", "for", "while", "in", "and", "or", "not",
        "True", "False", "None", "range", "len", "print"}
# Mots-cles du langage que notre vocabulaire n'a PAS. Un extrait qui en contient est
# rejete : les laisser passer les transformerait en identifiants (`is` deviendrait
# `n`), ce qui enseignerait a la mouche une grammaire fausse -- un mot-cle et une
# variable ne s'emploient pas au meme endroit. Vu sur un vrai extrait d'optparse.
import keyword as _kw
# ⚠️ Les parentheses comptent : `-` lie plus fort que `|`, donc sans elles on
# obtient kwlist | (softkwlist - MOTS), qui contient `def` et `return`. Inoffensif
# tant que le test `in MOTS` passe AVANT, donc un piege latent, pas une panne.
INTERDITS = (set(_kw.kwlist) | set(_kw.softkwlist)) - MOTS
MIN_JETONS, MAX_JETONS = 8, 60


def reduire(source: str) -> list[str] | None:
    """Reduit un extrait au vocabulaire, ou rend None s'il n'y entre pas."""
    sortie: list[str] = []
    connus: dict[str, str] = {}
    try:
        flux = tokenize.generate_tokens(io.StringIO(source).readline)
        for t in flux:
            if t.type == tokenize.ENDMARKER:
                break
            if t.type in (tokenize.COMMENT, tokenize.NL):
                continue
            if t.type == tokenize.NEWLINE:
                sortie.append("NL")
            elif t.type == tokenize.INDENT:
                sortie.append("IND")
            elif t.type == tokenize.DEDENT:
                sortie.append("DED")
            elif t.type == tokenize.NAME:
                if t.string in MOTS:
                    sortie.append(t.string)
                elif t.string in INTERDITS:
                    return None
                else:
                    # Un identifiant inconnu prend le prochain nom libre, et GARDE le
                    # meme dans tout l'extrait : sinon `x = x + 1` perdrait le fait
                    # que les deux x sont le meme, et la structure disparaitrait.
                    if t.string not in connus:
                        if len(connus) >= len(NOMS):
                            return None
                        connus[t.string] = NOMS[len(connus)]
                    sortie.append(connus[t.string])
            elif t.type == tokenize.NUMBER:
                try:
                    v = float(t.string)
                except ValueError:
                    return None
                sortie.append("0" if v == 0 else "1" if v == 1 else "2")
            elif t.type == tokenize.OP:
                if t.string not in ID:
                    return None
                sortie.append(t.string)
            else:
                return None            # chaine, f-string, decorateur exotique...
    except (tokenize.TokenError, IndentationError, SyntaxError, ValueError):
        return None
    if any(j not in ID for j in sortie):
        return None
    return sortie


def extraits_de(chemin: Path) -> list[str]:
    """Decoupe un fichier en definitions de fonctions, une par extrait."""
    try:
        texte = chemin.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return []
    lignes, out, courant, indent = texte.splitlines(), [], None, 0
    for ligne in lignes:
        nu = ligne.strip()
        creux = len(ligne) - len(ligne.lstrip())
        if nu.startswith("def ") and (courant is None or creux <= indent):
            if courant:
                out.append("\n".join(courant))
            courant, indent = [ligne[creux:]], creux
        elif courant is not None:
            if nu and creux <= indent:
                out.append("\n".join(courant))
                courant = None
            else:
                courant.append(ligne[indent:] if len(ligne) > indent else ligne)
    if courant:
        out.append("\n".join(courant))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max", type=int, default=600, help="nombre d'extraits gardes")
    ap.add_argument("--graine", type=int, default=20260912)
    a = ap.parse_args()

    racine = Path(sys.prefix) / "Lib"
    if not racine.exists():
        racine = Path(tokenize.__file__).parent
    fichiers = sorted(p for p in racine.glob("*.py") if p.stat().st_size < 400_000)
    print(f"source : {racine} ({len(fichiers)} fichiers)")

    vus, gardes, lus, rejetes = set(), [], 0, 0
    for f in fichiers:
        for src in extraits_de(f):
            lus += 1
            jetons = reduire(src)
            if jetons is None or not (MIN_JETONS <= len(jetons) <= MAX_JETONS):
                rejetes += 1
                continue
            cle = " ".join(jetons)
            if cle in vus:                 # les doublons fausseraient le decoupage
                continue
            vus.add(cle)
            gardes.append({"jetons": jetons, "source": f.name})
    random.Random(a.graine).shuffle(gardes)
    gardes = gardes[:a.max]

    dest = ROOT / "data" / "corpus.json"
    dest.write_text(json.dumps(
        {"vocabulaire": VOCAB, "extraits": gardes,
         "origine": f"bibliotheque standard Python, {racine.name}",
         "lus": lus, "rejetes": rejetes}, indent=1), encoding="utf-8")
    n_jetons = sum(len(g["jetons"]) for g in gardes)
    print(f"{lus} definitions lues, {rejetes} hors vocabulaire, "
          f"{len(vus)} distinctes retenues -> {len(gardes)} gardees")
    print(f"{n_jetons} jetons, {n_jetons / max(1, len(gardes)):.1f} par extrait")
    print(f"ecrit {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
