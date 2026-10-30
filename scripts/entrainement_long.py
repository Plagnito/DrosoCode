"""Entrainement long : jusqu'ou va la mouche, avec et sans neurogenese."""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from droso import calcul                     # noqa: E402
from droso.mesures import plafonds           # noqa: E402
from droso.trainer import MoucheQuiCode      # noqa: E402

SORTIE = Path(__file__).parent.parent / "data" / "resultats_apprentissage.json"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=4, help="jetons de contexte")
    ap.add_argument("--epoques", type=int, default=40)
    ap.add_argument("--epoques-apres", type=int, default=25)
    ap.add_argument("--croissance", type=int, default=3000)
    ap.add_argument("--taux", type=float, default=30.0)
    a = ap.parse_args()
    from droso.config import SETTINGS
    from droso import encoding as E
    print("moteur :", calcul.info()["moteur"], calcul.info()["detail"], flush=True)
    print(f"contexte k={a.k} | budget ALPN {SETTINGS.budget_alpn or 'desactive'}"
          f" -> {E.alpn_par_jeton(a.k)} ALPN/jeton", flush=True)
    m = MoucheQuiCode(k=a.k)
    res = {"etapes": [], "plafonds": None}
    t0 = time.time()

    def voir(c):
        c["kc"] = int(len(m.cerveau.i_kc))
        res["etapes"].append(c)
        if c["epoque"] % 5 == 0 or c["epoque"] <= 2:
            print(f"  {c['kc']:>5} KC | epoque {c['epoque']:>3} | app {c['app']:>6.1%}"
                  f" | TEST {c['test']:>6.1%} (top3 {c['test_top3']:>5.1%})", flush=True)
        SORTIE.write_text(json.dumps(res, indent=1), encoding="utf-8")

    res["reglages"] = {"k": a.k, "budget_alpn": SETTINGS.budget_alpn,
                       "alpn_par_jeton": E.alpn_par_jeton(a.k), "taux": a.taux}
    print(f"=== {len(m.cerveau.i_kc)} KC ===", flush=True)
    m.entrainer_lot(epoques=a.epoques, taille_lot=128, taux=a.taux, rappel=voir)
    if a.croissance:
        print(f"=== +{a.croissance} Kenyon cells ===", flush=True)
        m.croitre(a.croissance)
        m.entrainer_lot(epoques=a.epoques_apres, taille_lot=128, taux=a.taux, rappel=voir)
    print("plafonds du cerveau agrandi...", flush=True)
    res["plafonds"] = plafonds(m)
    res["secondes"] = round(time.time() - t0, 1)
    SORTIE.write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps(res["plafonds"], indent=1), flush=True)
    print(f"ecrit {SORTIE.name} en {res['secondes']} s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
