"""Contexte de code -> activation des neurones de projection (l'"odeur" du code).

Chez la mouche, une odeur active un sous-ensemble reproductible de glomerules,
donc de neurones de projection. On garde ce principe : chaque couple
(jeton, position dans le contexte) active un petit sous-ensemble FIXE d'ALPN,
tire deterministiquement par SHA-256. Deux contextes qui partagent des jetons
partagent des ALPN -- c'est ce qui donne la generalisation.
"""
from __future__ import annotations
import hashlib
from functools import lru_cache

import numpy as np

from .config import SETTINGS

N_ALPN_PAR_JETON = 12          # ~3,5% des 344 ALPN par jeton, a contexte court

# Budget TOTAL d'ALPN actifs, tenu quelle que soit la longueur du contexte.
# Sans lui, chaque jeton allume ses 12 ALPN fixes et k jetons en allument 12k : a
# k=16 c'est 152 ALPN sur 344 (44%), et deux contextes quelconques partagent 56% de
# leurs actifs -- ils sont deja presque indiscernables AVANT d'atteindre les Kenyon
# cells. Mesure : le repere biologique tombe de 60,7% (k=2) a 39,1% (k=16).
# 0 desactive le budget et retablit l'ancien comportement (12 par jeton, toujours).


@lru_cache(maxsize=8192)
def _glomerules(jeton: str, position: int, n_pn: int, n_actifs: int) -> tuple:
    """Sous-ensemble d'ALPN active par un jeton a une position donnee.
    Deterministe et stable entre executions (SHA-256, pas hash() qui est sale)."""
    graine = hashlib.sha256(f"{jeton}|{position}".encode("utf-8")).digest()
    rng = np.random.default_rng(int.from_bytes(graine[:8], "big"))
    return tuple(int(i) for i in rng.choice(n_pn, size=min(n_actifs, n_pn), replace=False))


def alpn_par_jeton(k: int, budget: int | None = None) -> int:
    """Combien d'ALPN allumer par jeton pour tenir le budget total.

    Sans budget, on garde N_ALPN_PAR_JETON quelle que soit la longueur du contexte —
    c'est ce qui sature l'entree. Avec budget, on divise, en gardant au moins 2 ALPN
    par jeton (en dessous, deux jetons differents se confondent trop souvent).
    """
    b = SETTINGS.budget_alpn if budget is None else int(budget)
    if b <= 0:
        return N_ALPN_PAR_JETON
    return max(2, min(N_ALPN_PAR_JETON, round(b / max(1, int(k)))))


def encoder(contexte, n_pn: int = 344, n_actifs: int | None = None,
            budget: int | None = None) -> np.ndarray:
    """Transforme une liste de jetons en vecteur d'activation ALPN dans [0, 1].

    Les jetons recents pesent plus lourd : l'activation decroit avec la distance,
    sinon l'ordre des jetons est perdu et 'x = y' devient indiscernable de 'y = x'.
    """
    if not isinstance(contexte, (list, tuple)) or not contexte:
        raise ValueError("Le contexte doit etre une liste non vide de jetons.")
    if any(not isinstance(j, str) for j in contexte):
        raise ValueError("Le contexte ne doit contenir que des chaines.")
    # ⚠️ n_actifs est lu A L'APPEL, jamais fige en valeur par defaut : une valeur
    # par defaut est evaluee A LA DEFINITION de la fonction, donc changer la globale
    # du module n'aurait AUCUN effet -- un banc qui fait varier ce parametre
    # mesurerait deux fois la meme chose, avec des chiffres identiques au centieme.
    if n_actifs is None:
        n_actifs = alpn_par_jeton(len(contexte), budget)
    v = np.zeros(int(n_pn), dtype=np.float32)
    k = len(contexte)
    for pos, jeton in enumerate(contexte):
        poids = 0.55 + 0.45 * (pos + 1) / k          # du plus ancien au plus recent
        for i in _glomerules(jeton, pos, int(n_pn), int(n_actifs)):
            v[i] = max(v[i], poids)
    return v


def recouvrement(a, b, n_pn: int = 344) -> float:
    """Part d'ALPN communs a deux contextes : mesure de leur ressemblance."""
    x, y = encoder(a, n_pn) > 0, encoder(b, n_pn) > 0
    u = (x | y).sum()
    return float((x & y).sum() / u) if u else 0.0
