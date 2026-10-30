"""Amene un circuit a la MEME parcimonie que les autres, avant toute comparaison.

Sans ca, comparer deux connectomes compare surtout deux points de fonctionnement.
Mesure qui l'a impose : a gains identiques, FlyWire tourne a 13 % de Kenyon cells
actives et le Male CNS a 4,5 % — trois fois plus epars, parce que sa densite KC->MBON
et ses comptes de synapses different. Le circuit le moins bien regle perd 10 points,
et on croit mesurer le cablage.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from droso import corpus as C                 # noqa: E402
from droso.connectome import nom_circuit      # noqa: E402
from droso.encoding import encoder            # noqa: E402
from droso.trainer import MoucheQuiCode       # noqa: E402


def taux_kc(m, PN) -> float:
    return float((m.cerveau.repondre_lot(PN, n_pas=8)["kc"] > 0).mean())


def calibrer(cible: float = 0.13, k: int = 2, tours: int = 24) -> dict:
    """Regle le gain d'entree pour atteindre la parcimonie cible."""
    m = MoucheQuiCode(k=k)
    ex = C.exemples(k=k, jeu="app")[:300]
    PN = np.array([encoder(c, n_pn=m.n_pn) for c, _ in ex], dtype=np.float32)
    depart = taux_kc(m, PN)
    gain = m.cerveau._gain
    for _ in range(tours):
        t = taux_kc(m, PN)
        if abs(t - cible) < 0.008:
            break
        # Plus d'activite que voulu -> moins de gain. Exposant < 1 pour amortir.
        gain *= (cible / max(t, 1e-6)) ** 0.45
        gain = float(np.clip(gain, 0.05, 400.0))
        m.cerveau._gain = gain
        m.cerveau._poser_poids()
        m.recalculer_canaux()
    return {"circuit": nom_circuit(), "k": k, "cible": cible,
            "taux_depart": round(depart, 4), "taux_final": round(taux_kc(m, PN), 4),
            "gain_depart": 4.0, "gain_final": round(gain, 3)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cible", type=float, default=0.13)
    ap.add_argument("--k", type=int, default=2)
    a = ap.parse_args()
    d = calibrer(a.cible, a.k)
    print(json.dumps(d))
    dest = Path(__file__).parent.parent / "data" / "calibration.json"
    tout = json.loads(dest.read_text(encoding="utf-8")) if dest.exists() else {}
    tout[d["circuit"]] = d
    dest.write_text(json.dumps(tout, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
