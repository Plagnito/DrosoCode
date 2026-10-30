"""Le cerveau grandit-il utilement ? Courbe « taille du corps pedoncule -> precision ».

Trois mesures par taille, qui ne disent pas la meme chose :

  * `plafond_libre`      : ce que les motifs de Kenyon cells contiennent, lu par un
                           lecteur lineaire sans contrainte de cablage.
  * `plafond_connectome` : le meilleur lecteur qui n'utilise QUE les synapses
                           KC->MBON existantes -- une regression par MBON, sur ses
                           propres afferentes. C'est la borne haute reelle du circuit.
  * `appris`             : ce que la mouche atteint en apprenant par dopamine.

Plus un TEMOIN a cablage remanie : memes degres, partenaires permutes. Il repond a
« le cablage precis de la mouche apporte-t-il quelque chose, ou seules ses
statistiques comptent ? » -- sans lui, un gain de taille ne prouve rien.
"""
from __future__ import annotations
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from droso import corpus as C                 # noqa: E402
from droso.mesures import motifs, plafond_libre, plafond_connectome  # noqa: E402
from droso.network import CerveauMouche       # noqa: E402
from droso.trainer import MoucheQuiCode       # noqa: E402

TAILLES = [0, 1000, 3000, 6000]
LECONS = 1200


def remanier(cb: CerveauMouche, rng) -> None:
    """Permute les partenaires en conservant les degres : meme nombre de griffes par
    Kenyon cell, meme nombre d'afferentes par MBON, mais QUI parle a QUI est brouille."""
    for bloc in ((cb.i_pn, cb.i_kc), (cb.i_kc, cb.i_mbon)):
        src, dst = bloc
        for attr in ("W", "W0"):
            A = getattr(cb, attr)
            sous = A[np.ix_(src, dst)]
            for j in range(sous.shape[1]):
                col = sous[:, j]
                nz = np.flatnonzero(col)
                if len(nz):
                    neuf = rng.choice(len(col), size=len(nz), replace=False)
                    v = col[nz].copy()
                    col[:] = 0.0
                    col[neuf] = v
            A[np.ix_(src, dst)] = sous
    cb.c.W[np.ix_(cb.i_kc, cb.i_mbon)] = cb.W0[np.ix_(cb.i_kc, cb.i_mbon)]


def mesurer(ajout: int, remanie: bool = False) -> dict:
    t0 = time.time()
    cb = CerveauMouche()
    if ajout:
        cb.croitre(ajout)
    if remanie:
        remanier(cb, np.random.default_rng(12345))
    m = MoucheQuiCode(cerveau=cb)
    K, y = motifs(m)
    d = {"kc": int(len(cb.i_kc)),
         "synapses_plastiques": int((cb.W0[np.ix_(cb.i_kc, cb.i_mbon)] != 0).sum()),
         "kc_actives": round(float((K > 0).mean()), 4),
         "plafond_libre": round(plafond_libre(K, y), 4),
         "plafond_connectome": round(plafond_connectome(K, y, m), 4)}
    m.entrainer(n_pas=LECONS, regle="delta")
    e = m.evaluer()
    d["appris"] = e["precision"]
    d["appris_top3"] = e["precision_top3"]
    d["lecons"] = LECONS
    d["secondes"] = round(time.time() - t0, 1)
    return d


def main() -> int:
    res = {"base": C.stats(), "lecons": LECONS, "tailles": [], "temoin_remanie": None}
    for a in TAILLES:
        d = mesurer(a)
        res["tailles"].append(d)
        print(f"  +{a:<5} -> {d['kc']:>5} KC | plafond libre {d['plafond_libre']:.1%} | "
              f"plafond connectome {d['plafond_connectome']:.1%} | appris {d['appris']:.1%} "
              f"({d['secondes']}s)", flush=True)
    print("  temoin a cablage remanie (memes degres)...", flush=True)
    res["temoin_remanie"] = mesurer(0, remanie=True)
    t = res["temoin_remanie"]
    print(f"  remanie -> plafond connectome {t['plafond_connectome']:.1%} | "
          f"appris {t['appris']:.1%}", flush=True)
    Path(__file__).parent.parent.joinpath("data/resultats_croissance.json").write_text(
        json.dumps(res, indent=2), encoding="utf-8")
    print("ecrit data/resultats_croissance.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
