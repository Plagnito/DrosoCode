"""Budget d'entree constant : le vrai correctif du contexte long.

Mesure precedente : reguler APL ne sauve rien, parce que la saturation est EN AMONT.
Chaque jeton allume 12 ALPN fixes, donc k jetons en allument 12k : a k=16 c'est 152
ALPN sur 344 (44 %), et deux contextes quelconques partagent 56 % de leurs ALPN
actifs. Ils sont deja presque indiscernables avant d'atteindre les Kenyon cells.

Ici on tient le BUDGET constant : n_actifs = budget / k. Le contexte s'allonge sans
que l'entree se densifie.
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from droso import corpus as C                 # noqa: E402
from droso import encoding as E               # noqa: E402
from droso import mesures as M                # noqa: E402
from droso.trainer import MoucheQuiCode       # noqa: E402

KS = [2, 4, 8, 16]
BUDGET = 48          # ALPN actifs au total, quelle que soit la longueur du contexte


def mesurer(k: int, budget: int | None) -> dict:
    t0 = time.time()
    n_actifs = E.N_ALPN_PAR_JETON if budget is None else max(2, round(budget / k))
    ancien = E.N_ALPN_PAR_JETON
    E.N_ALPN_PAR_JETON = n_actifs
    try:
        m = MoucheQuiCode(k=k)
        ex = C.exemples(k=k, jeu="app")[:300]
        P = np.array([E.encoder(c, n_pn=m.n_pn) for c, _ in ex]) > 0
        N = P.astype(float)
        N /= np.maximum(np.linalg.norm(N, axis=1, keepdims=True), 1e-9)
        S = N @ N.T
        iu = np.triu_indices(len(P), 1)
        K, y = M.motifs(m, "app")
        Kt, yt = M.motifs(m, "test")
        return {"k": k, "budget": budget, "alpn_par_jeton": n_actifs,
                "alpn_allumes": round(float(P.sum(1).mean()), 1),
                "recouvrement": round(float(S[iu].mean()), 3),
                "kc_actives": round(float((K > 0).mean()), 4),
                "libre_test": round(M.plafond_libre(K, y, Kt, yt), 4),
                "biologique_test": round(M.plafond_biologique(K, y, m, Kt, yt), 4),
                "secondes": round(time.time() - t0, 1)}
    finally:
        E.N_ALPN_PAR_JETON = ancien
        E._glomerules.cache_clear()


def main() -> int:
    out, dest = [], Path(__file__).parent.parent / "data" / "resultats_budget.json"
    for k in KS:
        for budget in (None, BUDGET):
            d = mesurer(k, budget)
            out.append(d)
            print(f"  k={d['k']:>2} {'budget constant' if budget else '12 ALPN/jeton':<16}"
                  f"| {d['alpn_par_jeton']:>2} ALPN/jeton | {d['alpn_allumes']:>5.1f} allumes"
                  f" | recouvr. {d['recouvrement']:.3f} | KC {d['kc_actives']:>5.1%}"
                  f" | libre {d['libre_test']:>6.1%} | biologique {d['biologique_test']:>6.1%}",
                  flush=True)
            dest.write_text(json.dumps({"budget": BUDGET, "points": out}, indent=1),
                            encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
