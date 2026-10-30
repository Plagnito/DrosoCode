"""Trois connectomes, la meme tache : qu'est-ce qui change vraiment ?

  * mb_right     — FlyWire femelle, hemisphere droit (le circuit historique)
  * mb_both      — FlyWire femelle, les DEUX hemispheres
  * mb_mcns_both — Male CNS (Google Research + HHMI Janelia, Cell 2026)

Le vocabulaire reste le meme pour que la comparaison porte sur le CABLAGE et rien
d'autre. Un connectome plus gros n'aide que s'il apporte quelque chose a la tache :
c'est precisement ce qu'on mesure, au lieu de le supposer.

⚠️ CHAQUE CIRCUIT EST D'ABORD CALIBRE A LA MEME PARCIMONIE. Sans ca on compare deux
POINTS DE FONCTIONNEMENT, pas deux cablages : a gains identiques, FlyWire tourne a
13 % de Kenyon cells actives et le Male CNS a 4,5 %, et le circuit mal regle perd
10 points. Defaut reellement commis, et qui donnait un verdict faux.
"""
from __future__ import annotations
import json, os, subprocess, sys, time
from pathlib import Path

RACINE = Path(__file__).resolve().parent.parent
CIRCUITS = ["mb_right", "mb_both", "mb_mcns_both"]
GRAINES = [42, 7, 1234]
K = 2                 # meilleur contexte mesure
EPOQUES = 30
CIBLE_KC = 0.13       # parcimonie commune, celle ou FlyWire a ete calibre


ENFANT = '''
import json, sys, time
sys.path.insert(0, %r)
import numpy as np
from droso.connectome import charger
from droso.mesures import plafonds
from droso.trainer import MoucheQuiCode
sys.path.insert(0, %r + "/scripts")
from calibrer_circuit import taux_kc
from droso import corpus as C
from droso.encoding import encoder
t0 = time.time()
m = MoucheQuiCode(k=%d, seed=%d)
# --- parcimonie appariee, avant toute mesure --------------------------------
_ex = C.exemples(k=%d, jeu="app")[:300]
_PN = np.array([encoder(c, n_pn=m.n_pn) for c, _ in _ex], dtype=np.float32)
_depart = taux_kc(m, _PN)
_g = m.cerveau._gain
for _ in range(24):
    _t = taux_kc(m, _PN)
    if abs(_t - %f) < 0.008:
        break
    _g = float(np.clip(_g * (%f / max(_t, 1e-6)) ** 0.45, 0.05, 400.0))
    m.cerveau._gain = _g
    m.cerveau._poser_poids()
    m.recalculer_canaux()
_final = taux_kc(m, _PN)
c = charger().resume()
p = plafonds(m)
m.entrainer_lot(epoques=%d, taille_lot=128, taux=30.0, melange=%d)
e = m.evaluer()
print("RESULTAT " + json.dumps({
    "circuit": %r, "graine": %d, "effectifs": c,
    "kc_actives_avant": round(_depart, 4), "kc_actives_apres": round(_final, 4),
    "gain": round(_g, 3),
    "kc": int(len(m.cerveau.i_kc)), "mbon_utilises": int(len(m.canaux)),
    "synapses_plastiques": int((m.cerveau.W0[__import__("numpy").ix_(
        m.cerveau.i_kc, m.cerveau.i_mbon)] != 0).sum()),
    "repere_libre": p["lecture_libre_test"],
    "repere_biologique": p["biologique_test"],
    "test": e["precision_test"], "test_top3": e["precision_test_top3"],
    "app": e["precision"], "secondes": round(time.time() - t0, 1)}))
'''


def main() -> int:
    out, dest = [], RACINE / "data" / "resultats_connectomes.json"
    for nom in CIRCUITS:
      for graine in GRAINES:
        env = {**os.environ, "DROSO_CIRCUIT": nom, "DROSO_GPU": "1"}
        # Un sous-processus par circuit : `charger` est mis en cache par lru_cache et
        # le reseau lit sa taille au demarrage, donc changer de connectome dans le
        # meme processus melangerait deux circuits sans le dire.
        code = ENFANT % (str(RACINE), str(RACINE), K, graine, K, CIBLE_KC,
                         CIBLE_KC, EPOQUES, graine, nom, graine)
        r = subprocess.run([sys.executable, "-c", code],
                           capture_output=True, text=True, env=env, cwd=str(RACINE))
        ligne = next((l for l in r.stdout.splitlines() if l.startswith("RESULTAT ")), None)
        if not ligne:
            print(f"  {nom:<14} ECHEC\n{r.stdout[-500:]}\n{r.stderr[-800:]}", flush=True)
            continue
        d = json.loads(ligne[len("RESULTAT "):])
        out.append(d)
        print(f"  {d['circuit']:<14} g{d['graine']:<5} | KC actives {d['kc_actives_avant']:>5.1%}"
              f" -> {d['kc_actives_apres']:>5.1%} (gain {d['gain']:>5.2f})"
              f" | repere {d['repere_biologique']:>6.1%}"
              f" | TEST {d['test']:>6.1%} (top3 {d['test_top3']:>6.1%}) | {d['secondes']}s",
              flush=True)
        dest.write_text(json.dumps({"k": K, "epoques": EPOQUES, "circuits": out}, indent=1),
                        encoding="utf-8")
    if out:
        b = max(out, key=lambda d: d["test"])
        print(f"\nmeilleur : {b['circuit']} -> {b['test']:.1%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
