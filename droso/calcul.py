"""Choix du moteur de calcul : NumPy (defaut) ou CuPy sur GPU.

Le GPU n'est PAS active par defaut, et c'est deliberé. Le gain vient d'abord du
traitement en LOT (x14 mesure sur le seul CPU, en remplacant 238 produits
matrice-vecteur par un produit matrice-matrice) ; le GPU ne fait que multiplier ce
gain. Porter le code sur CuPy sans batcher donnerait bien moins, la surcharge de
lancement des noyaux dominant sur huit pas sequentiels.

Activation : `DROSO_GPU=1`. Si CuPy ou le pilote manquent, on retombe sur NumPy en
le DISANT -- un repli silencieux ferait croire a un GPU qui calcule alors qu'il
n'est meme pas la.
"""
from __future__ import annotations
import os

import numpy as np

_xp = np
_gpu = False
_raison = "non demande (DROSO_GPU absent ou != 1)"

if os.environ.get("DROSO_GPU", "0") == "1":
    try:
        import cupy as _cp
        _cp.cuda.runtime.getDeviceCount()
        _cp.zeros(1) + 1                       # force l'init du contexte CUDA
        _xp, _gpu = _cp, True
        _raison = _cp.cuda.runtime.getDeviceProperties(0)["name"].decode()
    except Exception as exc:                   # noqa: BLE001
        _raison = f"CuPy indisponible ({type(exc).__name__}: {str(exc)[:80]}) -> NumPy"


def xp():
    """Le module de tableaux a utiliser (numpy ou cupy)."""
    return _xp


def sur_gpu() -> bool:
    return _gpu


def info() -> dict:
    return {"gpu": _gpu, "moteur": "cupy" if _gpu else "numpy", "detail": _raison}


def vers_device(a: np.ndarray):
    return _xp.asarray(a) if _gpu else a


def vers_hote(a) -> np.ndarray:
    return _xp.asnumpy(a) if _gpu else np.asarray(a)


def synchroniser() -> None:
    """A appeler AVANT tout chronometrage : les appels CuPy sont asynchrones, donc
    un chrono qui ne synchronise pas mesure le temps de MISE EN FILE, pas le calcul."""
    if _gpu:
        _xp.cuda.Stream.null.synchronize()
