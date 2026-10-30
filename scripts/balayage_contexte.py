"""Combien de jetons de contexte la mouche doit-elle voir ?

Mesure le REPERE lineaire pour plusieurs longueurs de contexte, sans entrainer :
c'est ce que l'information contenue dans le code des Kenyon cells permet. Si le
repere monte avec k, alors ce qui limite la mouche est son entree, pas son cerveau.
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from droso import mesures as M               # noqa: E402
from droso.trainer import MoucheQuiCode      # noqa: E402

KS = [1, 2, 4, 6, 8, 12, 16]


def main() -> int:
    out = []
    for k in KS:
        t = time.time()
        m = MoucheQuiCode(k=k)
        K, y = M.motifs(m, "app")
        Kt, yt = M.motifs(m, "test")
        d = {"k": k,
             "libre_test": round(M.plafond_libre(K, y, Kt, yt), 4),
             "connectome_test": round(M.plafond_connectome(K, y, m, Kt, yt), 4),
             "biologique_test": round(M.plafond_biologique(K, y, m, Kt, yt), 4),
             "kc_actives": round(float((K > 0).mean()), 4),
             "secondes": round(time.time() - t, 1)}
        out.append(d)
        print(f"  k={k:>2} | libre {d['libre_test']:>6.1%} | connectome "
              f"{d['connectome_test']:>6.1%} | biologique {d['biologique_test']:>6.1%}"
              f" | KC actives {d['kc_actives']:>5.1%} ({d['secondes']}s)", flush=True)
        Path(__file__).parent.parent.joinpath("data/resultats_contexte.json").write_text(
            json.dumps({"k": out}, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
