"""Chargement du circuit extrait du connectome FlyWire."""
from __future__ import annotations
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from .config import CIRCUIT, SETTINGS

CLASSES = ("ALPN", "Kenyon_Cell", "MBON", "DAN", "APL")


@dataclass(frozen=True)
class Circuit:
    W: np.ndarray            # matrice ionotropique signee (n x n)
    M: np.ndarray            # matrice modulatrice (dopamine), non signee
    classe: np.ndarray       # classe de chaque neurone
    root_id: np.ndarray      # identifiant FlyWire
    cell_type: np.ndarray
    nt: np.ndarray
    idx: dict                # classe -> indices

    def bloc(self, source: str, cible: str, mod: bool = False) -> np.ndarray:
        A = self.M if mod else self.W
        return A[np.ix_(self.idx[source], self.idx[cible])]

    @property
    def n(self) -> int:
        return len(self.classe)

    def resume(self) -> dict:
        return {c: int(len(self.idx[c])) for c in CLASSES}


def nom_circuit(side: str | None = None) -> str:
    """Nom du fichier de circuit. `DROSO_CIRCUIT` prime, sinon `mb_<side>`.

    Permet de basculer entre connectomes sans toucher au code : `mb_right`,
    `mb_both` (FlyWire femelle) ou `mb_mcns_both` (Male CNS, Google + Janelia).
    """
    return SETTINGS.circuit or f"mb_{side or SETTINGS.side}"


@lru_cache(maxsize=4)
def charger(side: str | None = None) -> Circuit:
    nom = nom_circuit(side)
    f = CIRCUIT / f"{nom}.npz"
    if not f.exists():
        dispo = sorted(x.stem for x in CIRCUIT.glob("mb_*.npz"))
        raise FileNotFoundError(
            f"Circuit absent : {f}. Disponibles : {dispo or 'aucun'}. "
            "Lancer `python scripts/extract_mb.py` (FlyWire) ou "
            "`python scripts/extract_mb_mcns.py` (Male CNS).")
    d = np.load(f, allow_pickle=True)
    classe = d["classe"]
    idx = {c: np.where(classe == c)[0] for c in CLASSES}
    manquantes = [c for c in CLASSES if len(idx[c]) == 0]
    if manquantes:
        raise ValueError(f"Classes absentes du circuit : {manquantes}")
    return Circuit(W=d["W"], M=d["M"], classe=classe, root_id=d["root_id"],
                   cell_type=d["cell_type"], nt=d["nt"], idx=idx)
