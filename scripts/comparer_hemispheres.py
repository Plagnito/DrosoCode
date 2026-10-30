"""Deux hemispheres : pourquoi ca NUIT, et est-ce que grouper les homologues repare ?

Mesure qui pose la question : 5 177 vraies Kenyon cells (deux hemispheres) donnent
50,4 % la ou 2 597 (un seul) donnent 56,2 %. Doubler le vrai tissu DEGRADE.

Mesure qui donne la cause : 74 % des MBON lisent plus de 95 % de leurs Kenyon cells
depuis LEUR cote. Avec un canal par MBON, les jetons se decident donc sur des
populations DISJOINTES — deux demi-corps pedoncules qui votent sur des faits
separes, pas un corps deux fois plus grand. La neurogenese, elle, branche ses
neurones sur les MEMES MBON : d'ou l'ecart.

Le correctif teste ici : un canal = un TYPE cellulaire, donc les homologues des deux
hemispheres sur la meme decision. Vocabulaire tronque a 35 jetons DES DEUX COTES
(il n'existe que 35 types de MBON), pour que la comparaison reste une comparaison.
"""
from __future__ import annotations
import json, os, subprocess, sys, time
from pathlib import Path

RACINE = Path(__file__).resolve().parent.parent
CAS = [("mb_right", False), ("mb_right", True),
       ("mb_both", False), ("mb_both", True),
       ("mb_mcns_both", False), ("mb_mcns_both", True)]
K, EPOQUES, JETONS, CIBLE_KC = 2, 30, 35, 0.13
GRAINES = [42, 7]

ENFANT = '''
import json, sys, time
sys.path.insert(0, %r); sys.path.insert(0, %r + "/scripts")
import numpy as np
from calibrer_circuit import taux_kc
from droso import corpus as C
from droso.encoding import encoder
from droso.trainer import MoucheQuiCode
t0 = time.time()
m = MoucheQuiCode(k=%d, seed=%d, groupes=%s, n_jetons=%d)
_ex = C.exemples(k=%d, jeu="app")[:300]
_PN = np.array([encoder(c, n_pn=m.n_pn) for c, _ in _ex], dtype=np.float32)
_g = m.cerveau._gain
for _ in range(24):
    _t = taux_kc(m, _PN)
    if abs(_t - %f) < 0.008: break
    _g = float(np.clip(_g * (%f / max(_t, 1e-6)) ** 0.45, 0.05, 400.0))
    m.cerveau._gain = _g; m.cerveau._poser_poids(); m.recalculer_canaux()
m.entrainer_lot(epoques=%d, taille_lot=128, taux=30.0, melange=%d)
e = m.evaluer()
print("RESULTAT " + json.dumps({
    "circuit": %r, "groupes": %s, "graine": %d,
    "canaux": len(m.canaux), "kc": int(len(m.cerveau.i_kc)),
    "mbon_par_canal": round(sum(len(g) for g in m.groupes_canaux) / len(m.groupes_canaux), 2),
    "test": e["precision_test"], "test_top3": e["precision_test_top3"],
    "app": e["precision"], "secondes": round(time.time() - t0, 1)}))
'''


def main() -> int:
    out, dest = [], RACINE / "data" / "resultats_hemispheres.json"
    for nom, grp in CAS:
        for graine in GRAINES:
            env = {**os.environ, "DROSO_CIRCUIT": nom, "DROSO_GPU": "1"}
            code = ENFANT % (str(RACINE), str(RACINE), K, graine, "True" if grp else "False",
                             JETONS, K, CIBLE_KC, CIBLE_KC, EPOQUES, graine, nom,
                             "True" if grp else "False", graine)
            r = subprocess.run([sys.executable, "-c", code], capture_output=True,
                               text=True, env=env, cwd=str(RACINE))
            ligne = next((l for l in r.stdout.splitlines() if l.startswith("RESULTAT ")), None)
            if not ligne:
                print(f"  {nom:<14} groupes={grp} g{graine} ECHEC\n{r.stderr[-700:]}", flush=True)
                continue
            d = json.loads(ligne[len("RESULTAT "):])
            out.append(d)
            print(f"  {d['circuit']:<14} {'groupes' if grp else 'simples':<8} g{graine:<5}"
                  f"| {d['canaux']:>2} canaux, {d['mbon_par_canal']:>4} MBON/canal"
                  f" | TEST {d['test']:>6.1%} (top3 {d['test_top3']:>6.1%}) | {d['secondes']}s",
                  flush=True)
            dest.write_text(json.dumps({"k": K, "epoques": EPOQUES, "jetons": JETONS,
                                        "cas": out}, indent=1), encoding="utf-8")
    if out:
        import statistics as st
        print("\n=== moyennes ===")
        for nom, grp in [(n, g) for n, g in CAS]:
            v = [d["test"] for d in out if d["circuit"] == nom and d["groupes"] == grp]
            if v:
                print(f"  {nom:<14} {'groupes' if grp else 'simples':<8} {st.mean(v):>6.1%}"
                      f"  {[f'{x:.1%}' for x in v]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
