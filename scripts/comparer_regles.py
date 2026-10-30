"""Compare les regles d'apprentissage sur le meme circuit et le meme corpus."""
from __future__ import annotations
import json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from droso import corpus as C            # noqa: E402
from droso.mesures import plafonds       # noqa: E402
from droso.trainer import MoucheQuiCode  # noqa: E402


def main() -> int:
    t0 = time.time()
    ref = MoucheQuiCode()
    res = {"plafonds": plafonds(ref), "regles": {}, "base": C.stats()}
    print("plafonds :", json.dumps(res["plafonds"]), flush=True)
    for regle in ("depression", "bidirectionnelle", "delta"):
        m = MoucheQuiCode()
        courbe = []
        for cible in (300, 900, 1800, 3000):
            m.entrainer(n_pas=cible - m.lecons, regle=regle)
            e = m.evaluer()
            courbe.append({"lecons": m.lecons, "precision": e["precision"],
                           "top3": e["precision_top3"]})
            print(f"  {regle:<17} {m.lecons:>5} lecons -> {e['precision']:.1%} "
                  f"(top3 {e['precision_top3']:.1%})", flush=True)
        res["regles"][regle] = courbe
    res["secondes"] = round(time.time() - t0, 1)
    Path(__file__).parent.parent.joinpath("data/resultats_regles.json").write_text(
        json.dumps(res, indent=2), encoding="utf-8")
    print("\necrit data/resultats_regles.json en", res["secondes"], "s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
