"""Les MBON se parlent entre eux : est-ce que ca sert ?

Le reseau entraine depasse la meilleure lecture LINEAIRE du code des Kenyon cells.
Deux explications possibles : l'estimateur lineaire converge mal, ou le reseau n'est
pas une fonction lineaire de ce code. Ce temoin tranche : on entraine deux fois le
meme circuit, une fois avec les 386 aretes MBON->MBON, une fois sans.
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from droso.network import CerveauMouche      # noqa: E402
from droso.trainer import MoucheQuiCode      # noqa: E402

EPOQUES = 20


def essai(couper: bool) -> dict:
    cb = CerveauMouche()
    n_av = int((cb.W[np.ix_(cb.i_mbon, cb.i_mbon)] != 0).sum())
    if couper:
        sous = np.ix_(cb.i_mbon, cb.i_mbon)
        cb.W[sous] = 0.0
        cb.W0[sous] = 0.0
        cb._version_poids += 1
    m = MoucheQuiCode(cerveau=cb)
    t = time.time()
    r = m.entrainer_lot(epoques=EPOQUES, taille_lot=128, taux=30.0)
    return {"aretes_mbon_mbon": 0 if couper else n_av,
            "test": r["precision_test"], "test_top3": r["precision_test_top3"],
            "app": r["precision"], "secondes": round(time.time() - t, 1)}


def main() -> int:
    out = {}
    for couper, nom in ((False, "intact"), (True, "MBON->MBON coupe")):
        out[nom] = essai(couper)
        d = out[nom]
        print(f"{nom:<18} {d['aretes_mbon_mbon']:>4} aretes | "
              f"TEST {d['test']:.1%} (top3 {d['test_top3']:.1%}) | {d['secondes']}s",
              flush=True)
    ecart = out["intact"]["test"] - out["MBON->MBON coupe"]["test"]
    out["ecart"] = round(ecart, 4)
    print(f"\necart du a la recurrence entre MBON : {ecart:+.1%}")
    Path(__file__).parent.parent.joinpath("data/temoin_mbon.json").write_text(
        json.dumps(out, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
