"""Distributions du circuit et du corpus, pour que l'interface montre des FAITS.

Tout est calcule depuis les donnees reelles : aucun chiffre n'est ecrit en dur ici,
sinon il finirait par mentir sans que rien ne le signale.
"""
from __future__ import annotations
from collections import Counter
from functools import lru_cache

import numpy as np

from . import corpus as C
from .connectome import charger


def histogramme(valeurs, n_paniers: int = 24, mini: float | None = None) -> dict:
    v = np.asarray(valeurs, dtype=np.float64)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return {"bords": [], "effectifs": []}
    lo = float(v.min() if mini is None else mini)
    hi = float(v.max())
    if hi <= lo:
        hi = lo + 1.0
    eff, bords = np.histogram(v, bins=int(n_paniers), range=(lo, hi))
    return {"bords": [round(float(b), 3) for b in bords],
            "effectifs": [int(e) for e in eff]}


@lru_cache(maxsize=2)
def circuit(side: str = "right") -> dict:
    """Ce que le connectome contient reellement : degres, types, neurotransmetteurs."""
    c = charger(side)
    pk = c.bloc("ALPN", "Kenyon_Cell")
    km = c.bloc("Kenyon_Cell", "MBON")
    griffes = (pk != 0).sum(axis=0)
    kc_par_mbon = (km != 0).sum(axis=0)
    cibles_par_kc = (km != 0).sum(axis=1)
    poids = np.abs(km[km != 0])

    types = Counter(str(t) for t in c.cell_type[c.idx["Kenyon_Cell"]] if str(t))
    nt = Counter(str(n) for n in c.nt)
    return {
        "effectifs": c.resume(),
        "griffes": {"hist": histogramme(griffes, 12, 0),
                    "mediane": float(np.median(griffes[griffes > 0])),
                    "sans_entree": int((griffes == 0).sum())},
        "kc_par_mbon": {"hist": histogramme(kc_par_mbon, 16, 0),
                        "min": int(kc_par_mbon.min()), "max": int(kc_par_mbon.max()),
                        "mediane": float(np.median(kc_par_mbon)),
                        "tri": [int(x) for x in np.sort(kc_par_mbon)[::-1]]},
        "cibles_par_kc": {"hist": histogramme(cibles_par_kc, 16, 0),
                          "moyenne": round(float(cibles_par_kc.mean()), 2)},
        "synapses_par_paire": {"hist": histogramme(poids, 20, 0),
                               "mediane": float(np.median(poids)) if poids.size else 0.0},
        "types_kc": types.most_common(8),
        "neurotransmetteurs": nt.most_common(8),
        "densite_kc_mbon": round(float((km != 0).mean()), 4),
    }


def corpus_stats() -> dict:
    """Ce que le corpus contient : frequences, longueurs, provenance."""
    seqs = C.sequences()
    freq = Counter(j for s in seqs for j in s)
    total = max(1, sum(freq.values()))
    longueurs = [len(s) for s in seqs]
    app, test = C.decouper()
    # Frequence du jeton CIBLE, dans l'ordre du vocabulaire, pour l'interface.
    par_jeton = [{"jeton": j, "n": freq.get(j, 0),
                  "part": round(freq.get(j, 0) / total, 5)} for j in C.VOCAB]
    par_jeton.sort(key=lambda d: -d["n"])
    return {"source": C.SOURCE, "extraits": len(seqs), "jetons": total,
            "longueurs": {"hist": histogramme(longueurs, 18, 0),
                          "mediane": float(np.median(longueurs)) if longueurs else 0.0},
            "frequences": par_jeton,
            "decoupe": {"app": len(app), "test": len(test)},
            **C.stats()}
