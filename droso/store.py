"""Persistance de l'etat de la mouche — pour qu'un redemarrage ne perde rien.

Ecriture ATOMIQUE : fichier temporaire puis `os.replace`. Ouvrir la cible en 'w'
la vide AVANT d'ecrire, donc le moindre echec en cours de route laisse un fichier
de zero octet a la place de l'etat precedent.
"""
from __future__ import annotations
import json
import os
import tempfile
from pathlib import Path

import numpy as np

from .config import ROOT

DOSSIER = ROOT / "data" / "etat"
VERSION = 1


def _ecrire_atomique(cible: Path, ecrire) -> None:
    DOSSIER.mkdir(parents=True, exist_ok=True)
    # Le suffixe DOIT etre celui de la cible : np.savez_compressed ajoute ".npz"
    # quand le nom n'en a pas, donc il ecrivait dans "xxx.tmp.npz" pendant que
    # os.replace deplacait le "xxx.tmp" reste VIDE -- et la sauvegarde annoncait
    # "sauve" en laissant un fichier de zero octet.
    fd, tmp = tempfile.mkstemp(dir=str(cible.parent), suffix=cible.suffix)
    os.close(fd)
    try:
        ecrire(Path(tmp))
        if os.path.getsize(tmp) == 0:
            raise IOError(f"Sauvegarde vide refusee pour {cible.name}.")
        os.replace(tmp, cible)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def sauver(m, journal: list | None = None) -> dict:
    """Sauve ce qui a ete APPRIS, pas le connectome : le bloc plastique KC->MBON,
    la taille du cerveau et le journal. Le cablage se relit depuis data/circuit/."""
    cb = m.cerveau
    sous = np.ix_(cb.i_kc, cb.i_mbon)
    meta = {"version": VERSION, "lecons": int(m.lecons), "k": int(m.k),
            "n_kc": int(len(cb.i_kc)), "n_neurones": int(cb.c.n),
            "canaux": [int(x) for x in m.canaux],
            "journal": journal or []}
    _ecrire_atomique(DOSSIER / "cerveau.npz", lambda p: np.savez_compressed(
        p, Wp=cb.W[sous], W0p=cb.W0[sous], base=m.base,
        # le cablage des KC neoformees ne se retrouve pas depuis data/circuit/
        pk=cb.c.W[np.ix_(cb.i_pn, cb.i_kc)], km=cb.c.W[sous],
        ka=cb.c.W[np.ix_(cb.i_kc, cb.i_apl)], ak=cb.c.W[np.ix_(cb.i_apl, cb.i_kc)]))
    _ecrire_atomique(DOSSIER / "cerveau.json",
                     lambda p: p.write_text(json.dumps(meta, indent=2), encoding="utf-8"))
    return {"sauve": True, **{k: meta[k] for k in ("lecons", "n_kc", "n_neurones")}}


def etat_sauve() -> dict | None:
    f = DOSSIER / "cerveau.json"
    if not f.exists():
        return None
    try:
        meta = json.loads(f.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    return meta if meta.get("version") == VERSION else None


def charger_dans(m) -> bool:
    """Restaure l'etat appris dans une mouche fraiche. Rend False si rien a charger
    ou si la sauvegarde ne correspond pas au circuit courant."""
    meta = etat_sauve()
    if meta is None or not (DOSSIER / "cerveau.npz").exists():
        return False
    cb = m.cerveau
    manque = meta["n_kc"] - len(cb.i_kc)
    if manque > 0:
        cb.croitre(manque)
        m.recalculer_canaux()
    elif manque < 0:
        return False
    d = np.load(DOSSIER / "cerveau.npz", allow_pickle=False)
    sous = np.ix_(cb.i_kc, cb.i_mbon)
    if d["Wp"].shape != cb.W[sous].shape:
        return False
    # On remet le cablage exact des neoformees, puis les poids appris.
    cb.c.W[np.ix_(cb.i_pn, cb.i_kc)] = d["pk"]
    cb.c.W[sous] = d["km"]
    cb.c.W[np.ix_(cb.i_kc, cb.i_apl)] = d["ka"]
    cb.c.W[np.ix_(cb.i_apl, cb.i_kc)] = d["ak"]
    cb._poser_poids()
    cb.W[np.ix_(cb.i_kc, cb.i_mbon)] = d["Wp"]
    cb.W0[np.ix_(cb.i_kc, cb.i_mbon)] = d["W0p"]
    m.canaux = np.array(meta["canaux"], dtype=int)
    m.base = d["base"]
    m.lecons = int(meta["lecons"])
    return True


def journal() -> list:
    meta = etat_sauve()
    return meta.get("journal", []) if meta else []


def oublier() -> None:
    for f in ("cerveau.npz", "cerveau.json"):
        p = DOSSIER / f
        if p.exists():
            p.unlink()
