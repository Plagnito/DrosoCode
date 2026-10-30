"""Le contexte court gagne-t-il AUSSI pour le vrai reseau, ou seulement pour la sonde ?

Le repere lineaire classe k=2 devant k=4. Mais tous nos entrainements ont tourne a
k=4, et le reseau n'est PAS une lecture lineaire du code des Kenyon cells -- il peut
donc classer autrement. On entraine pour de vrai, aux memes budgets.
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from droso import calcul                      # noqa: E402
from droso import encoding as E               # noqa: E402
from droso.config import SETTINGS             # noqa: E402
from droso.trainer import MoucheQuiCode       # noqa: E402

CAS = [(2, 0), (4, 0), (8, 0), (16, 0), (2, 3000), (4, 3000)]
EPOQUES = 30


def main() -> int:
    print("moteur :", calcul.info()["moteur"], "| budget ALPN :", SETTINGS.budget_alpn,
          flush=True)
    out, dest = [], Path(__file__).parent.parent / "data" / "resultats_contexte_reel.json"
    for k, croissance in CAS:
        t0 = time.time()
        m = MoucheQuiCode(k=k)
        if croissance:
            m.croitre(croissance)
        m.entrainer_lot(epoques=EPOQUES, taille_lot=128, taux=30.0)
        e = m.evaluer()
        d = {"k": k, "croissance": croissance, "kc": int(len(m.cerveau.i_kc)),
             "alpn_par_jeton": E.alpn_par_jeton(k),
             "test": e["precision_test"], "test_top3": e["precision_test_top3"],
             "app": e["precision"], "secondes": round(time.time() - t0, 1)}
        out.append(d)
        print(f"  k={d['k']:>2} {d['kc']:>5} KC ({d['alpn_par_jeton']} ALPN/jeton)"
              f" | TEST {d['test']:>6.1%} (top3 {d['test_top3']:>6.1%})"
              f" | appris {d['app']:>6.1%} | {d['secondes']}s", flush=True)
        dest.write_text(json.dumps({"epoques": EPOQUES, "cas": out}, indent=1),
                        encoding="utf-8")
    meilleur = max(out, key=lambda d: d["test"])
    print(f"\nmeilleur : k={meilleur['k']}, {meilleur['kc']} KC -> {meilleur['test']:.1%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
