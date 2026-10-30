"""Le mini-langage que la mouche doit apprendre.

CONTRAINTE FONDATRICE : le vocabulaire ne depasse pas le nombre de MBON du vrai
circuit (48 cote droit). Un jeton = un MBON. Il n'y a donc AUCUNE couche de
lecture ajoutee par-dessus le cerveau : la decision sort des neurones de sortie
de la mouche, tels que le connectome les donne. Si le vocabulaire depassait 48,
il faudrait rajouter un classifieur -- et ce ne serait plus la mouche qui decide.
"""
from __future__ import annotations
import random

# 40 jetons <= 48 MBON.
VOCAB = [
    "def", "return", "if", "else", "for", "while", "in", "range", "len", "print",
    "and", "or", "not", "True", "False", "None",
    "(", ")", "[", "]", ":", ",", ".", "=", "==", "<", ">", "+", "-", "*",
    "f", "x", "y", "n", "i", "0", "1", "2",
    "NL", "IND", "DED",
]
ID = {t: i for i, t in enumerate(VOCAB)}
assert len(VOCAB) == len(set(VOCAB)), "vocabulaire avec doublons"

# Patrons de code Python reduits au vocabulaire ci-dessus.
PATRONS = [
    "def f ( x ) : NL IND return x + 1 DED",
    "def f ( x ) : NL IND return x * 2 DED",
    "def f ( x , y ) : NL IND return x + y DED",
    "def f ( n ) : NL IND for i in range ( n ) : NL IND print ( i ) DED DED",
    "def f ( x ) : NL IND if x > 0 : NL IND return x DED else : NL IND return 0 DED DED",
    "def f ( n ) : NL IND i = 0 NL while i < n : NL IND i = i + 1 DED NL return i DED",
    "def f ( x ) : NL IND if x == None : NL IND return False DED return True DED",
    "def f ( x ) : NL IND return len ( x ) DED",
    "for i in range ( 2 ) : NL IND print ( i ) DED",
    "if x and y : NL IND return True DED",
    "if not x : NL IND return None DED",
    "x = [ 0 , 1 , 2 ] NL print ( len ( x ) ) ",
    "def f ( x ) : NL IND y = x * 2 NL return y + 1 DED",
    "while x < n : NL IND x = x + 2 DED",
]


def jetons(p: str) -> list[str]:
    t = p.split()
    inconnus = sorted({j for j in t if j not in ID})
    if inconnus:
        raise ValueError(f"Patron hors vocabulaire : {inconnus}")
    return t


def _charger_reel() -> list[list[str]] | None:
    """Corpus de vrai code Python reduit au vocabulaire, s'il a ete construit.
    A defaut, on retombe sur les patrons ecrits a la main -- en le disant."""
    import json
    from .config import ROOT
    f = ROOT / "data" / "corpus.json"
    if not f.exists():
        return None
    try:
        d = json.loads(f.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    ex = [e["jetons"] for e in d.get("extraits", []) if e.get("jetons")]
    # Bride pour les tests : evaluer sur 7771 exemples a chaque appel rend la suite
    # inutilisable (plus de 15 min mesurees). La bride ne touche PAS la production,
    # qui n'a pas cette variable -- un corpus reduit en douce fausserait tous les
    # chiffres sans qu'aucun test ne le voie.
    import os
    bride = os.environ.get("DROSO_MAX_EXTRAITS")
    if bride:
        ex = ex[:max(2, int(bride))]
    hors = {j for seq in ex for j in seq} - set(ID)
    if hors:
        raise ValueError(f"data/corpus.json contient des jetons hors vocabulaire : {sorted(hors)}")
    return ex or None


_REEL = _charger_reel()
SOURCE = "vrai code Python reduit" if _REEL else "patrons ecrits a la main"


def sequences() -> list[list[str]]:
    """Les sequences du corpus : vrai code si disponible, patrons sinon."""
    return _REEL if _REEL else [jetons(p) for p in PATRONS]


def corpus() -> list[list[str]]:
    return sequences()


PART_TEST = 0.25
GRAINE_DECOUPE = 20260912


def decouper(graine: int = GRAINE_DECOUPE, part: float = PART_TEST) -> tuple[list, list]:
    """Separe les PATRONS, jamais les exemples.

    Decouper par exemple laisserait des contextes du MEME patron des deux cotes :
    le jeu de test contiendrait alors des morceaux deja vus et le score mesurerait
    de la memorisation. Mesure a l'appui : decoupe par exemple, 13,2 % en
    apprentissage contre 6,2 % sur le retenu -- et 6,2 % est SOUS la base du jeton
    majoritaire (9,2 %). Un decoupage par patron est plus severe, et honnete.
    """
    if not (0.0 < float(part) < 1.0):
        raise ValueError(f"part doit etre dans ]0, 1[ (recu {part!r})")
    n = len(sequences())
    idx = list(range(n))
    random.Random(int(graine)).shuffle(idx)
    n_test = max(1, round(n * float(part)))
    test = sorted(idx[:n_test])
    app = sorted(idx[n_test:])
    if not app:
        raise ValueError("Plus aucun patron pour l'apprentissage : part trop grande.")
    return app, test


def paires_de(indices, k: int) -> list[tuple[list[str], str]]:
    seqs = sequences()
    paires = []
    for i_p in indices:
        seq = seqs[i_p]
        for i in range(len(seq)):
            ctx = seq[max(0, i - k):i]
            ctx = ["NL"] * (k - len(ctx)) + ctx      # rembourrage a gauche
            paires.append((ctx, seq[i]))
    return paires


def exemples(k: int = 4, melange: int | None = None,
             jeu: str = "tout") -> list[tuple[list[str], str]]:
    """Paires (contexte de k jetons, jeton suivant).

    `jeu` vaut "app" (patrons d'apprentissage), "test" (patrons retenus) ou "tout".
    """
    if not isinstance(k, int) or k < 1 or k > 16:
        raise ValueError(f"k doit etre un entier dans [1, 16] (recu {k!r})")
    if jeu not in ("app", "test", "tout"):
        raise ValueError(f"jeu doit valoir 'app', 'test' ou 'tout' (recu {jeu!r})")
    app, test = decouper()
    indices = {"app": app, "test": test, "tout": list(range(len(sequences())))}[jeu]
    paires = paires_de(indices, k)
    if melange is not None:
        random.Random(melange).shuffle(paires)
    return paires


def stats() -> dict:
    from collections import Counter
    app, test = decouper()
    ex_app, ex_test = exemples(jeu="app"), exemples(jeu="test")
    c = Counter(cible for _, cible in ex_app)
    majoritaire = c.most_common(1)[0][0]
    # La base « jeton majoritaire » se mesure sur le TEST, avec le jeton appris sur
    # l'apprentissage : c'est la barre qu'il faut battre sur du jamais-vu.
    base_test = sum(1 for _, cible in ex_test if cible == majoritaire) / len(ex_test)
    return {"patrons": len(sequences()), "source": SOURCE,
            "patrons_app": len(app), "patrons_test": len(test),
            "vocabulaire": len(VOCAB),
            "exemples": len(ex_app) + len(ex_test),
            "exemples_app": len(ex_app), "exemples_test": len(ex_test),
            "jetons_distincts_cibles": len(c),
            "hasard": round(1 / len(VOCAB), 4),
            "jeton_majoritaire": majoritaire,
            "base_majoritaire": round(c.most_common(1)[0][1] / len(ex_app), 4),
            "base_majoritaire_test": round(base_test, 4)}
