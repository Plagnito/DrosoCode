"""Amelioration automatique : la mouche s'entraine, et grandit quand elle stagne.

Boucle d'un cycle : entrainer -> evaluer -> si le gain est trop faible, faire
naitre des Kenyon cells, recalibrer, continuer. C'est la traduction directe de la
mesure : quand l'apprentissage ne progresse plus, ce qui manque n'est pas du temps,
c'est du tissu.

Tourne dans un fil separe pour que l'interface reste vivante, et sauve l'etat
APRES CHAQUE CYCLE -- un arret ou un redemarrage ne coute qu'un cycle.
"""
from __future__ import annotations
import threading
import time
import traceback

from . import store


class Amelioration:
    def __init__(self, fabrique_mouche):
        self._fabrique = fabrique_mouche
        self._fil: threading.Thread | None = None
        self._stop = threading.Event()
        self._lock = threading.RLock()
        self.journal: list[dict] = list(store.journal())
        self.etat = "au repos"
        self.erreur: str | None = None
        self.cycle, self.cycles_demandes = 0, 0

    # ------------------------------------------------------------------ etat
    def statut(self) -> dict:
        with self._lock:
            return {"etat": self.etat, "en_cours": self.en_cours,
                    "cycle": self.cycle, "cycles_demandes": self.cycles_demandes,
                    "erreur": self.erreur, "journal": self.journal[-40:]}

    @property
    def en_cours(self) -> bool:
        return self._fil is not None and self._fil.is_alive()

    # --------------------------------------------------------------- pilotage
    def demarrer(self, cycles: int = 6, lecons: int = 400, croissance: int = 800,
                 seuil: float = 0.01) -> dict:
        if not (1 <= int(cycles) <= 50):
            raise ValueError(f"cycles doit etre dans [1, 50] (recu {cycles!r})")
        if not (1 <= int(lecons) <= 2000):
            raise ValueError(f"lecons doit etre dans [1, 2000] (recu {lecons!r})")
        if not (0 <= int(croissance) <= 5000):
            raise ValueError(f"croissance doit etre dans [0, 5000] (recu {croissance!r})")
        if not (0.0 <= float(seuil) <= 1.0):
            raise ValueError(f"seuil doit etre dans [0, 1] (recu {seuil!r})")
        with self._lock:
            if self.en_cours:
                raise ValueError("Une amelioration est deja en cours.")
            self._stop.clear()
            self.erreur = None
            self.cycle, self.cycles_demandes = 0, int(cycles)
            self.etat = "demarrage"
            self._fil = threading.Thread(
                target=self._boucle, args=(int(cycles), int(lecons), int(croissance),
                                           float(seuil)), daemon=True)
            self._fil.start()
        return self.statut()

    def arreter(self) -> dict:
        self._stop.set()
        with self._lock:
            if self.en_cours:
                self.etat = "arret demande"
        return self.statut()

    # ----------------------------------------------------------------- boucle
    def _boucle(self, cycles: int, lecons: int, croissance: int, seuil: float):
        try:
            m = self._fabrique()
            precedente = m.evaluer()["precision"]
            # Le journal survit d'un lancement a l'autre : on numerote a la SUITE,
            # sinon la table affiche plusieurs "cycle 1" et devient illisible.
            deja = self.journal[-1]["cycle"] if self.journal else 0
            for i in range(1, cycles + 1):
                if self._stop.is_set():
                    break
                with self._lock:
                    self.cycle, self.etat = i, f"cycle {i}/{cycles} : entrainement"
                t0 = time.time()
                m.entrainer(n_pas=lecons, regle="delta")
                e = m.evaluer()
                gain = e["precision"] - precedente

                a_grandi = 0
                # On ne fait grandir le cerveau que si l'apprentissage stagne : ajouter
                # du tissu a une mouche qui progresse encore ne prouverait rien.
                if croissance and gain < seuil and i < cycles:
                    with self._lock:
                        self.etat = f"cycle {i}/{cycles} : neurogenese"
                    m.croitre(croissance)
                    a_grandi = croissance

                entree = {"cycle": deja + i, "cycle_du_lancement": i,
                          "lecons": m.lecons, "kc": int(len(m.cerveau.i_kc)),
                          "precision": e["precision"], "top3": e["precision_top3"],
                          "gain": round(gain, 4), "neurogenese": a_grandi,
                          "hasard": e["hasard"], "majoritaire": e["base_majoritaire"],
                          "secondes": round(time.time() - t0, 1)}
                with self._lock:
                    self.journal.append(entree)
                store.sauver(m, self.journal)
                precedente = e["precision"]
            with self._lock:
                self.etat = "arrete" if self._stop.is_set() else "termine"
        except Exception as exc:                       # noqa: BLE001
            with self._lock:
                self.erreur = f"{type(exc).__name__}: {exc}"
                self.etat = "en erreur"
            traceback.print_exc()
