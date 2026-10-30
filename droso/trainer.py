"""Apprendre a coder a une mouche, avec la regle d'apprentissage de la mouche.

Chez la drosophile, la dopamine DEPRIME la synapse Kenyon cell -> MBON active.
On n'apprend donc pas en renforcant la bonne reponse : on apprend en affaiblissant
les mauvaises, dans le contexte precis ou elles sont mauvaises. Comme seules ~9%
des KC repondent a un contexte donne, la depression ne touche que leurs synapses :
l'apprentissage est associatif par construction, pas par une astuce ajoutee.
"""
from __future__ import annotations
import threading

import numpy as np

from . import corpus as C
from .config import SETTINGS
from .encoding import encoder
from .network import CerveauMouche


class MoucheQuiCode:
    def __init__(self, cerveau: CerveauMouche | None = None, k: int = 4,
                 seed: int = SETTINGS.seed, groupes: bool = False,
                 n_jetons: int | None = None):
        self.cerveau = cerveau or CerveauMouche(seed=seed)
        self.k = int(k)
        # Un canal = UN MBON par defaut. En mode `groupes`, un canal rassemble tous
        # les MBON d'un MEME TYPE cellulaire, donc les homologues des deux
        # hemispheres. Motif mesure : 74% des MBON lisent plus de 95% de leurs
        # Kenyon cells depuis LEUR cote, donc sur un circuit bi-hemispherique les
        # jetons se decident sur des populations DISJOINTES — deux demi-corps
        # pedoncules qui votent sur des faits separes, et non un corps deux fois
        # plus grand. Grouper les homologues remet les deux cotes sur la meme
        # decision.
        self.groupes = bool(groupes)
        self.n_jetons = len(C.VOCAB) if n_jetons is None else int(n_jetons)
        if not (2 <= self.n_jetons <= len(C.VOCAB)):
            raise ValueError(f"n_jetons doit etre dans [2, {len(C.VOCAB)}]")
        # Restreindre le nombre de canaux restreint AUSSI le vocabulaire : sinon une
        # cible hors plage indexe un tableau de scores plus court. On garde les
        # jetons les PLUS FREQUENTS du jeu d'apprentissage, et on ecarte les
        # exemples qui visent les autres — identiquement quel que soit le circuit,
        # pour que la comparaison reste une comparaison.
        from collections import Counter
        freq = Counter(cible for _, cible in C.exemples(k=self.k, jeu="app"))
        gardes = [j for j, _ in freq.most_common(self.n_jetons)]
        self.jetons = sorted(gardes, key=C.ID.get)
        self.rang = {j: i for i, j in enumerate(self.jetons)}
        self.n_pn = len(self.cerveau.i_pn)
        self.n_mbon = len(self.cerveau.i_mbon)
        if self.n_jetons > self.n_mbon:
            raise ValueError(
                f"Vocabulaire ({self.n_jetons}) > MBON du circuit ({self.n_mbon}) : "
                "il faudrait ajouter un classifieur, donc ce ne serait plus la mouche "
                "qui decide. Reduire le vocabulaire.")
        self._lock = threading.RLock()
        self.lecons = 0
        self.historique: list[dict] = []
        # Dernier score MESURE, garde en cache : `evaluer()` parcourt tout le
        # corpus (7771 exemples) et prend ~50 s sur CPU. Le recalculer a chaque
        # appel d'API rend l'interface inutilisable.
        self.dernier: dict | None = None
        self.recalculer_canaux()

    def _choisir_canaux(self):
        """Les MBON du vrai circuit sont tres inegaux : de 0 a 2089 Kenyon cells
        afferentes. Mesure : la variation ENTRE MBON (0,214) depasse le double de la
        variation selon le CONTEXTE (0,095), correlation 0,69 entre force du MBON et
        score moyen. Associer naivement "jeton i -> MBON i" fait donc gagner toujours
        les memes MBON quel que soit le code presente (7 reponses distinctes sur 41,
        mesurees) et rend inapprenables les jetons tombes sur un MBON sans afferente."""
        cb = self.cerveau
        kc_par_mbon = (cb.W0[np.ix_(cb.i_kc, cb.i_mbon)] != 0).sum(axis=0)
        if self.groupes:
            # Un canal par TYPE cellulaire : les homologues des deux cotes ensemble.
            from collections import defaultdict
            par_type = defaultdict(list)
            for rang, t in enumerate(self.cerveau.c.cell_type[cb.i_mbon]):
                par_type[str(t) or f"?{rang}"].append(rang)
            grp = [(sum(int(kc_par_mbon[r]) for r in rangs), rangs)
                   for rangs in par_type.values()
                   if sum(int(kc_par_mbon[r]) for r in rangs) >= 50]
            if len(grp) < self.n_jetons:
                raise ValueError(
                    f"Le circuit n'offre que {len(grp)} groupes de MBON exploitables "
                    f"pour {self.n_jetons} jetons.")
            grp.sort(key=lambda x: -x[0])
            self.groupes_canaux = [g[1] for g in grp[:self.n_jetons]]
            self.canaux = np.array([g[0] for g in self.groupes_canaux], dtype=int)
        else:
            utilisables = np.where(kc_par_mbon >= 50)[0]
            if len(utilisables) < self.n_jetons:
                raise ValueError(
                    f"Le circuit n'offre que {len(utilisables)} MBON exploitables "
                    f"(>=50 KC afferentes) pour {self.n_jetons} jetons.")
            # Les plus forts d'abord : le vocabulaire tient sur les meilleurs canaux.
            self.canaux = utilisables[np.argsort(-kc_par_mbon[utilisables])][:self.n_jetons]
            self.groupes_canaux = [[int(i)] for i in self.canaux]
        self.base = np.zeros(self.n_jetons, dtype=np.float32)

    def recalculer_canaux(self, n: int = 40):
        """(Re)choisit les canaux de sortie et recalibre leur ligne de base.
        A rappeler apres une neurogenese : le nombre de Kenyon cells a change, donc
        la reponse moyenne de chaque MBON aussi."""
        self._choisir_canaux()
        self._calibrer(n)

    def croitre(self, n_nouvelles: int) -> dict:
        """Neurogenese, puis recalibrage. Les poids appris sont conserves."""
        with self._lock:
            r = self.cerveau.croitre(n_nouvelles)
            self.recalculer_canaux()
            return r

    def _calibrer(self, n: int = 40):
        """Ligne de base de chaque canal : sa reponse moyenne avant tout apprentissage.
        On compare ensuite chaque MBON a LUI-MEME, pas aux autres -- sinon c'est la
        force intrinseque du neurone qui decide, jamais le contexte."""
        ex = self.exemples("app", melange=7)[:n]
        PN = np.array([encoder(c, n_pn=self.n_pn) for c, _ in ex], dtype=np.float32)
        brut = self._par_canal(self.cerveau.repondre_lot(PN, n_pas=8)["mbon"])
        self.base = brut.mean(axis=0).astype(np.float32)

    def _par_canal(self, mbon: np.ndarray) -> np.ndarray:
        """Ramene l'activite des MBON a une valeur par canal. En mode groupes, la
        SOMME des homologues : les deux cotes contribuent a la meme decision."""
        if not self.groupes:
            return mbon[..., self.canaux]
        return np.stack([mbon[..., g].sum(axis=-1) for g in self.groupes_canaux],
                        axis=-1)

    def _brut(self, contexte):
        # Le verrou n'est pas decoratif : l'amelioration automatique fait grandir le
        # cerveau dans un autre fil, ce qui REDIMENSIONNE les matrices. Une lecture
        # concurrente tomberait au milieu.
        with self._lock:
            pn = encoder(contexte, n_pn=self.n_pn)
            r = self.cerveau.repondre(pn, n_pas=8)
            return self._par_canal(r["mbon"]), r["kc"]

    # ------------------------------------------------------------- inference
    def scores(self, contexte):
        """Score de chaque jeton : ecart du MBON a sa propre ligne de base."""
        brut, kc = self._brut(contexte)
        return brut - self.base, kc

    def scores_lot(self, contextes) -> tuple:
        """Scores de plusieurs contextes en UNE passe. Voir `repondre_lot`.

        ⚠️ Le resultat n'est pas identique au bit pres a la version en serie : BLAS
        change l'ordre des sommes entre GEMV et GEMM, ce qui deplace les valeurs de
        ~3e-8 -- et le reseau est a seuil, avec 159 neurones a moins de 1e-6 du
        seuil. Un neurone bascule, sa trajectoire diverge. VERIFIE : sur le corpus
        entier, 0 decision sur 238 change, et la precision est identique au chiffre
        pres. C'est donc une equivalence de COMPORTEMENT, pas de bits -- et un test
        qui exigerait l'egalite exacte serait rouge sans qu'il y ait de defaut.
        """
        with self._lock:
            PN = np.array([encoder(c, n_pn=self.n_pn) for c in contextes],
                          dtype=np.float32)
            r = self.cerveau.repondre_lot(PN, n_pas=8)
            return self._par_canal(r["mbon"]) - self.base, r["kc"]

    def proposer(self, contexte) -> dict:
        s, _ = self.scores(contexte)
        ordre = np.argsort(-s)
        return {"jeton": self.jetons[int(ordre[0])],
                "top3": [{"jeton": self.jetons[int(i)], "score": round(float(s[i]), 4)}
                         for i in ordre[:3]],
                "tout_nul": bool(s.max() <= 0)}

    # ---------------------------------------------------------- apprentissage
    def lecon(self, contexte, cible: str, taux: float = 0.05,
              regle: str = "bidirectionnelle") -> dict:
        """Une lecon : on presente le contexte, on punit les MBON actifs qui ne sont
        pas la bonne reponse. La bonne reponse n'est JAMAIS renforcee."""
        if cible not in C.ID:
            raise ValueError(f"Jeton cible hors vocabulaire : {cible!r}")
        with self._lock:
            s, kc = self.scores(contexte)
            juste = C.ID[cible]
            avant = int(np.argmax(s))
            # On ne punit QUE le MBON qui a gagne a tort -- l'erreur reellement
            # commise. Punir tous les MBON actifs (35 sur 41, mesure) n'est pas un
            # apprentissage : c'est un effacement global, et la precision BAISSE.
            punir = np.zeros(self.n_mbon, dtype=np.float32)
            actifs = np.zeros(len(C.VOCAB), dtype=bool)
            if avant != juste:
                actifs[avant] = True
                punir[self.canaux[avant]] = 1.0     # canal reel, pas l'indice du jeton
            if regle not in ("depression", "bidirectionnelle", "delta"):
                raise ValueError(f"regle inconnue : {regle!r}")
            if regle == "delta":
                cible_v = np.zeros(self.n_mbon, dtype=np.float32)
                pred = np.zeros(self.n_mbon, dtype=np.float32)
                cible_v[self.canaux[juste]] = 1.0
                p = s - s.max()
                p = np.exp(p) / np.exp(p).sum()
                pred[self.canaux] = p
                bouge = self.cerveau.apprentissage_delta(kc, cible_v - pred, taux=taux)
                self.cerveau.recuperation()
                self.lecons += 1
                return {"avant": C.VOCAB[avant], "cible": cible, "regle": regle,
                        "correct_avant": avant == juste, "mbon_punis": 0,
                        "depression": 0.0, "renfort": round(bouge, 5)}
            depression = self.cerveau.depression_dopaminergique(kc, punir, taux=taux)
            renfort = 0.0
            if regle == "bidirectionnelle" and avant != juste:
                # La plasticite KC->MBON est bidirectionnelle selon les compartiments :
                # on renforce aussi le canal de la bonne reponse. Mesure a l'appui plus
                # bas : la depression seule ne depasse pas le hasard sur un choix a 41.
                recompense = np.zeros(self.n_mbon, dtype=np.float32)
                recompense[self.canaux[juste]] = 1.0
                renfort = self.cerveau.potentiation_dopaminergique(kc, recompense, taux=taux)
            self.cerveau.recuperation()
            self.lecons += 1
            return {"avant": C.VOCAB[avant], "cible": cible,
                    "correct_avant": avant == juste,
                    "mbon_punis": int(actifs.sum()),
                    "depression": round(depression, 5),
                    "renfort": round(renfort, 5)}

    def entrainer(self, n_pas: int = 50, taux: float = 0.05, melange: int = 0,
                  regle: str = "bidirectionnelle") -> dict:
        if not isinstance(n_pas, int) or not (1 <= n_pas <= 2000):
            raise ValueError(f"n_pas doit etre un entier dans [1, 2000] (recu {n_pas!r})")
        ex = C.exemples(k=self.k, melange=melange, jeu="app")
        justes = 0
        for i in range(n_pas):
            ctx, cible = ex[(self.lecons) % len(ex)]
            r = self.lecon(ctx, cible, taux=taux, regle=regle)
            justes += int(r["correct_avant"])
        bilan = {"pas": n_pas, "regle": regle, "justes_pendant": justes,
                 "precision_pendant": round(justes / n_pas, 4), "lecons": self.lecons}
        self.historique.append(bilan)
        return bilan

    def entrainer_lot(self, epoques: int = 5, taille_lot: int = 128,
                      taux: float = 0.5, melange: int = 0,
                      rappel=None) -> dict:
        """Apprentissage par mini-lots, sur les patrons d'APPRENTISSAGE seulement.

        Chaque lot fait UNE passe avant en lot puis UNE mise a jour moyennee. C'est
        a la fois beaucoup plus rapide et un meilleur estimateur du gradient que
        l'apprentissage exemple par exemple, qui suit le bruit de chacun.
        """
        if not (1 <= int(epoques) <= 500):
            raise ValueError(f"epoques doit etre dans [1, 500] (recu {epoques!r})")
        if not (1 <= int(taille_lot) <= 4096):
            raise ValueError(f"taille_lot doit etre dans [1, 4096] (recu {taille_lot!r})")
        # Borne large A DESSEIN : la premiere valait 50, et la mesure a montre que
        # l'optimum est au-dela (17,2% de precision a taux=0,5 contre 38,9% a
        # taux=30). Un garde-fou pose au jugé qui interdit la zone utile coute plus
        # qu'il ne protege.
        if not (0.0 < float(taux) <= 10000.0):
            raise ValueError(f"taux doit etre dans ]0, 10000] (recu {taux!r})")
        ex = self.exemples("app", melange=melange)
        y = np.array([self.rang[t] for _, t in ex])
        ctx = [c for c, _ in ex]
        courbe = []
        with self._lock:
            for ep in range(int(epoques)):
                ordre = np.random.default_rng(melange + ep).permutation(len(ex))
                for debut in range(0, len(ordre), int(taille_lot)):
                    lot = ordre[debut:debut + int(taille_lot)]
                    S, K = self.scores_lot([ctx[i] for i in lot])
                    P = np.exp(S - S.max(axis=1, keepdims=True))
                    P /= P.sum(axis=1, keepdims=True)
                    cible = np.zeros_like(P)
                    cible[np.arange(len(lot)), y[lot]] = 1.0
                    E = np.zeros((len(lot), self.n_mbon), dtype=np.float32)
                    err = (cible - P).astype(np.float32)
                    for t, g in enumerate(self.groupes_canaux):
                        E[:, g] = err[:, [t]]      # la correction va a tout le groupe
                    self.cerveau.apprentissage_delta_lot(K, E, taux=taux)
                    self.lecons += len(lot)
                e = self.evaluer()
                courbe.append({"epoque": ep + 1, "lecons": self.lecons,
                               "app": e["precision"], "test": e["precision_test"],
                               "test_top3": e["precision_test_top3"]})
                if rappel:
                    rappel(courbe[-1])
        self.historique.append({"epoques": int(epoques), "taille_lot": int(taille_lot),
                                "taux": float(taux), "courbe": courbe})
        return {"epoques": int(epoques), "lecons": self.lecons, "courbe": courbe,
                **self.evaluer()}

    # ---------------------------------------------------------------- mesure
    def exemples(self, jeu: str, melange: int | None = None) -> list:
        """Exemples dont la cible tient dans le vocabulaire retenu."""
        ex = C.exemples(k=self.k, melange=melange, jeu=jeu)
        if self.n_jetons == len(C.VOCAB):
            return ex
        return [(c, t) for c, t in ex if t in self.rang]

    def _mesurer(self, jeu: str, limite: int | None) -> dict:
        ex = self.exemples(jeu)
        if limite:
            ex = ex[:int(limite)]
        S, _ = self.scores_lot([c for c, _ in ex])
        y = np.array([self.rang[t] for _, t in ex])
        ordre = np.argsort(-S, axis=1)
        justes = int((ordre[:, 0] == y).sum())
        top3 = int((ordre[:, :3] == y[:, None]).any(axis=1).sum())
        return {"exemples": len(ex), "precision": round(justes / len(ex), 4),
                "top3": round(top3 / len(ex), 4)}

    def evaluer(self, limite: int | None = None) -> dict:
        """Mesure sur les patrons APPRIS et sur les patrons RETENUS.

        `precision` reste la valeur d'apprentissage pour ne pas casser les appelants,
        mais `precision_test` est la seule qui dise quelque chose : mesure, un
        decoupage par exemple donnait 13,2 % en apprentissage pour 6,2 % sur du
        jamais-vu -- soit SOUS la base du jeton majoritaire.
        """
        a = self._mesurer("app", limite)
        t = self._mesurer("test", limite)
        st = C.stats()
        self.dernier = {"exemples": a["exemples"], "exemples_test": t["exemples"],
                       "precision": a["precision"], "precision_top3": a["top3"],
                       "precision_test": t["precision"], "precision_test_top3": t["top3"],
                       "hasard": st["hasard"], "base_majoritaire": st["base_majoritaire"],
                       "base_majoritaire_test": st["base_majoritaire_test"],
                       "lecons": self.lecons}
        return self.dernier
