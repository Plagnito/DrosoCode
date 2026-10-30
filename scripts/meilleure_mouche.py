"""La mouche la plus intelligente qu'on sache faire — tous les leviers mesures, ensemble.

Chaque choix ci-dessous vient d'une mesure, pas d'une intuition :

  * **circuit `mb_both`** — FlyWire, les DEUX hemispheres. A 35 jetons il bat un seul
    hemisphere (57,4 % contre 54,1 %). ⚠️ A 41 jetons c'etait l'inverse (50,4 % contre
    56,2 %) : il fallait alors racler 41 canaux sur 48 d'un cote, et la comparaison
    mesurait cette contrainte, pas le cablage.
  * **canaux GROUPES** — un canal = un type cellulaire, donc les MBON homologues des
    deux cotes sur la meme decision. 74 % des MBON lisent plus de 95 % de leurs Kenyon
    cells depuis LEUR cote : sans groupement, les jetons se decident sur des
    populations disjointes. Gain mesure : +2,3 points, et le meilleur top-3 vu.
  * **k=2** — 50,9 % contre 43,1 % a k=4 sur le cerveau natif. Plus de contexte
    densifie l'entree (12 ALPN par jeton, donc 12k allumes) et noie le codage epars.
  * **pas 30** — 17,2 % a 0,5 ; 49,1 % a 30 ; instable au-dela.
  * **parcimonie calibree a 13 %** — sans ca on compare des regimes, pas des cablages.
  * **neurogenese** — 49,1 % → 65,6 % en passant de 2 597 a 5 597 Kenyon cells.
  * **beaucoup d'epoques** — a budget constant, un plus gros cerveau est SOUS-ENTRAINE
    (8 597 KC plafonnait a 32,4 % en top-1 pendant que son top-3 montait encore).

35 jetons et non 41 : il n'existe que 35 types de MBON, et un canal groupe en demande un.
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path

import numpy as np

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
ETAPES = [(40, 0), (60, 4000), (50, 4000)]     # (epoques, KC a faire naitre AVANT)
SORTIE = RACINE / "data" / "meilleure_mouche.json"


def calibrer(m) -> float:
    """Ramene la parcimonie a la cible. A refaire apres chaque neurogenese : le
    nombre de Kenyon cells change, donc le regime de fonctionnement aussi."""
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
           "etapes": [], "meilleur": None}

    def enregistrer(c):
        c["kc"] = int(len(m.cerveau.i_kc))
        res["etapes"].append(c)
        if res["meilleur"] is None or c["test"] > res["meilleur"]["test"]:
            res["meilleur"] = dict(c)
        if c["epoque"] % 10 == 0 or c["epoque"] <= 2:
            print(f"  {c['kc']:>5} KC | epoque {c['epoque']:>3} | app {c['app']:>6.1%}"
                  f" | TEST {c['test']:>6.1%} (top3 {c['test_top3']:>6.1%})", flush=True)
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
    res["secondes"] = round(time.time() - t0, 1)
    res["final"] = m.evaluer()
    SORTIE.write_text(json.dumps(res, indent=1), encoding="utf-8")
    try:
        store.sauver(m, [])
        res["sauve"] = True
    except Exception as exc:                   # noqa: BLE001
        print("sauvegarde impossible :", exc, flush=True)
    SORTIE.write_text(json.dumps(res, indent=1), encoding="utf-8")
    f = res["final"]
    print(f"\n=== MEILLEURE MOUCHE ===")
    print(f"  {len(m.cerveau.i_kc)} Kenyon cells, {len(m.canaux)} canaux groupes")
    print(f"  generalisation {f['precision_test']:.1%} | top-3 {f['precision_test_top3']:.1%}")
    print(f"  (jeton majoritaire {f['base_majoritaire_test']:.1%}, hasard {f['hasard']:.1%})")
    print(f"  repere biologique {res['plafonds']['biologique_test']:.1%}"
          f" | lecteur libre {res['plafonds']['lecture_libre_test']:.1%}")
    print(f"  en {res['secondes']} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
