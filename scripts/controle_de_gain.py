"""APL comme controle de gain : peut-on donner plus de contexte sans noyer le code ?

Mesure : allonger le contexte DEGRADE la mouche (repere biologique 60,7 % a k=2,
39,1 % a k=16), et la cause est visible — la part de Kenyon cells actives passe de
13 % a 57 %. L'encodeur noie le code epars, et c'est la parcimonie qui fait toute la
discriminabilite du corps pedoncule.

APL est precisement le neurone de CONTROLE DE GAIN de la mouche, et on le fait
tourner a gain fixe. Ici on le regule pour tenir une parcimonie cible quel que soit
k, et on regarde si le contexte redevient utile.
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from droso import corpus as C                 # noqa: E402
from droso import mesures as M                # noqa: E402
from droso.encoding import encoder            # noqa: E402
from droso.trainer import MoucheQuiCode       # noqa: E402

KS = [2, 4, 8, 16]
CIBLE = 0.21          # part de KC actives visee, celle du reglage d'origine a k=4


def regler_apl(m, cible: float, tours: int = 20) -> float:
    """Ajuste le gain d'APL jusqu'a ce que la parcimonie atteigne la cible."""
    ex = C.exemples(k=m.k, jeu="app")[:256]
    PN = np.array([encoder(c, n_pn=m.n_pn) for c, _ in ex], dtype=np.float32)
    for _ in range(tours):
        t = float((m.cerveau.repondre_lot(PN, n_pas=8)["kc"] > 0).mean())
        if abs(t - cible) < 0.012:
            break
        # Plus d'activite que voulu -> plus d'inhibition. Exposant < 1 pour amortir.
        m.cerveau.gain_apl *= (t / max(cible, 1e-6)) ** 0.7
        m.cerveau.gain_apl = float(np.clip(m.cerveau.gain_apl, 0.05, 500.0))
    m.recalculer_canaux()
    return m.cerveau.gain_apl


def mesurer(k: int, regule: bool) -> dict:
    t0 = time.time()
    m = MoucheQuiCode(k=k)
    gain = regler_apl(m, CIBLE) if regule else m.cerveau.gain_apl
    K, y = M.motifs(m, "app")
    Kt, yt = M.motifs(m, "test")
    return {"k": k, "regule": regule, "gain_apl": round(float(gain), 2),
            "kc_actives": round(float((K > 0).mean()), 4),
            "libre_test": round(M.plafond_libre(K, y, Kt, yt), 4),
            "biologique_test": round(M.plafond_biologique(K, y, m, Kt, yt), 4),
            "secondes": round(time.time() - t0, 1)}


def main() -> int:
    out = []
    dest = Path(__file__).parent.parent / "data" / "resultats_gain.json"
    for k in KS:
        for regule in (False, True):
            d = mesurer(k, regule)
            out.append(d)
            print(f"  k={d['k']:>2} {'APL regule' if regule else 'gain fixe ':<11}"
                  f"| KC actives {d['kc_actives']:>5.1%} | gain {d['gain_apl']:>6.2f}"
                  f" | libre {d['libre_test']:>6.1%} | biologique {d['biologique_test']:>6.1%}"
                  f" ({d['secondes']}s)", flush=True)
            dest.write_text(json.dumps({"cible_kc": CIBLE, "points": out}, indent=1),
                            encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
