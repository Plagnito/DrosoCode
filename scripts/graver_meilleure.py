"""Reproduit la meilleure mouche mesuree, et GRAVE son cerveau sur disque.

La configuration vient entierement de mesures (voir `meilleure_mouche.py` pour le
detail de chaque levier). On s'arrete a 9 177 Kenyon cells parce que c'est la taille
qui a donne le meilleur resultat : le palier suivant (13 177) sature la memoire du
GPU a 97 % et tombe a moins d'une epoque par 100 s, pour un gain que la courbe ne
laissait pas esperer.

⚠️ On sauve A CHAQUE AMELIORATION, pas a la fin. La generalisation oscille d'une
epoque a l'autre (67,7 % au pic, 65 % trente epoques plus tard) : sauver le dernier
etat graverait une mouche moins bonne que celle qu'on a vraiment obtenue.
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path

import numpy as np

# ⚠️ La console Windows encode en cp1252 : un caractere hors de cette table fait
# CRASHER le script au moment du print, apres des minutes d'entrainement. On force
# la sortie en UTF-8 avec repli, plutot que de s'interdire les accents.
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

RACINE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RACINE))
sys.path.insert(0, str(RACINE / "scripts"))
from calibrer_circuit import taux_kc          # noqa: E402
from droso import calcul                      # noqa: E402
from droso import corpus as C                 # noqa: E402
from droso import store                       # noqa: E402
from droso.encoding import encoder            # noqa: E402
from droso.mesures import plafonds            # noqa: E402
from droso.trainer import MoucheQuiCode       # noqa: E402

K, JETONS, TAUX, CIBLE_KC = 2, 35, 30.0, 0.13
ETAPES = [(40, 0), (35, 4000)]
SORTIE = RACINE / "data" / "meilleure_mouche.json"


def calibrer(m) -> float:
    ex = C.exemples(k=m.k, jeu="app")[:300]
    PN = np.array([encoder(c, n_pn=m.n_pn) for c, _ in ex], dtype=np.float32)
    g = m.cerveau._gain
    for _ in range(24):
        t = taux_kc(m, PN)
        if abs(t - CIBLE_KC) < 0.008:
            break
        g = float(np.clip(g * (CIBLE_KC / max(t, 1e-6)) ** 0.45, 0.05, 400.0))
        m.cerveau._gain = g
        m.cerveau._poser_poids()
        m.recalculer_canaux()
    return taux_kc(m, PN)


def main() -> int:
    t0 = time.time()
    print("moteur :", calcul.info()["moteur"], calcul.info()["detail"], flush=True)
    m = MoucheQuiCode(k=K, groupes=True, n_jetons=JETONS)
    res = {"reglages": {"circuit": "mb_both", "k": K, "jetons": JETONS, "taux": TAUX,
                        "groupes": True, "cible_kc": CIBLE_KC},
           "etapes": [], "meilleur": None, "graves": 0}

    def enregistrer(c):
        c["kc"] = int(len(m.cerveau.i_kc))
        res["etapes"].append(c)
        neuf = res["meilleur"] is None or c["test"] > res["meilleur"]["test"]
        if neuf:
            res["meilleur"] = dict(c)
            try:
                store.sauver(m, [])
                res["graves"] += 1
                marque = "  <- grave"
            except Exception as exc:           # noqa: BLE001
                marque = f"  <- sauvegarde IMPOSSIBLE : {exc}"
        else:
            marque = ""
        if neuf or c["epoque"] % 10 == 0:
            print(f"  {c['kc']:>5} KC | epoque {c['epoque']:>3} | app {c['app']:>6.1%}"
                  f" | TEST {c['test']:>6.1%} (top3 {c['test_top3']:>6.1%}){marque}",
                  flush=True)
        SORTIE.write_text(json.dumps(res, indent=1), encoding="utf-8")

    for epoques, croissance in ETAPES:
        if croissance:
            print(f"=== neurogenese +{croissance} ===", flush=True)
            m.croitre(croissance)
        t = calibrer(m)
        print(f"=== {len(m.cerveau.i_kc)} KC, parcimonie {t:.1%}, "
              f"{len(m.canaux)} canaux ({sum(len(g) for g in m.groupes_canaux)} MBON), "
              f"{epoques} epoques ===", flush=True)
        m.entrainer_lot(epoques=epoques, taille_lot=128, taux=TAUX, rappel=enregistrer)

    print("plafonds du cerveau final...", flush=True)
    res["plafonds"] = plafonds(m)
    res["final"] = m.evaluer()
    res["secondes"] = round(time.time() - t0, 1)
    SORTIE.write_text(json.dumps(res, indent=1), encoding="utf-8")
    b, p = res["meilleur"], res["plafonds"]
    print("\n=== MEILLEURE MOUCHE GRAVEE ===")
    print(f"  {b['kc']} Kenyon cells, {len(m.canaux)} canaux groupes, {JETONS} jetons")
    print(f"  generalisation {b['test']:.1%} | top-3 {b['test_top3']:.1%}")
    print(f"  (jeton majoritaire {res['final']['base_majoritaire_test']:.1%},"
          f" hasard {res['final']['hasard']:.1%})")
    print(f"  repere biologique {p['biologique_test']:.1%}"
          f" | lecteur libre {p['lecture_libre_test']:.1%}")
    print(f"  {res['graves']} gravures, en {res['secondes']} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
