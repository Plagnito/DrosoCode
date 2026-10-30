"""Reseau LIF tournant sur le VRAI cablage du corps pedoncule de la drosophile.

Trois choses distinguent ce reseau d'un reseau bio-inspire ecrit a la main :

1. Aucune connexion NATIVE n'est tiree au hasard : W vient du connectome FlyWire.
   (La neurogenese, elle, tire le cablage des neoformees -- mais en imitant la
   statistique des vraies, pas uniformement. Voir `croitre`.)
2. L'inhibition globale n'est pas un parametre arbitraire : APL est un vrai neurone
   du connectome, reciproquement cable sur toutes les Kenyon cells. Il est traite en
   inhibiteur GRADUE et non en neurone a potentiels d'action -- c'est ce que dit la
   biologie, et c'est ce que la mesure impose : modelise en LIF a seuil, APL
   n'emet JAMAIS (verifie : le couper ne change rien du tout) et l'inhibition
   laterale du corps pedoncule disparait en silence.
3. La regle d'apprentissage est celle de la mouche : la dopamine DEPRIME la
   synapse KC->MBON active, elle ne la renforce pas. On apprend en affaiblissant
   les mauvaises reponses, pas en renforcant la bonne.
"""
from __future__ import annotations
import threading

import numpy as np

from . import calcul
from .connectome import Circuit, charger
from .config import SETTINGS


class CerveauMouche:
    def __init__(self, circuit: Circuit | None = None, seed: int = SETTINGS.seed,
                 gain: float | None = None, gain_apl: float | None = None,
                 gain_mbon: float | None = None):
        self.c = circuit or charger()
        self.rng = np.random.default_rng(int(seed))
        self._lock = threading.RLock()
        self._gain = float(SETTINGS.gain if gain is None else gain)
        # Gain d'APL : convertit son entree normalisee (~0,05 quand 5% des KC
        # emettent) en une sortie graduee utilisable.
        self.gain_apl = float(SETTINGS.gain_apl if gain_apl is None else gain_apl)
        # Gain propre a la lecture KC->MBON : un MBON groupe ~328 KC dont 3% a 5%
        # emettent, soit une dizaine de sources. Sans gain dedie il ne decharge jamais.
        self.gain_mbon = float(SETTINGS.gain_mbon if gain_mbon is None else gain_mbon)
        self.v_thr, self.decay, self.t_refrac = 1.0, 0.80, 2
        self.pas, self.spikes_total = 0, 0
        self._version_poids = 0
        self._poser_poids()

    def _poser_poids(self):
        """Pose les indices, les poids de travail et l'etat, depuis `self.c`.
        Rappele apres une neurogenese, qui agrandit le circuit."""
        self.i_pn = self.c.idx["ALPN"]
        self.i_kc = self.c.idx["Kenyon_Cell"]
        self.i_mbon = self.c.idx["MBON"]
        self.i_dan = self.c.idx["DAN"]
        self.i_apl = self.c.idx["APL"]
        self.n = self.c.n

        # Normalisation : le nombre de synapses couvre plusieurs ordres de grandeur.
        # On ramene chaque neurone a une entree excitatrice totale comparable, sinon
        # une poignee de paires tres synaptiques sature tout le circuit.
        self.W = self.c.W.astype(np.float32).copy()
        exc = np.clip(self.W, 0, None).sum(axis=0)
        exc[exc == 0] = 1.0
        self.W /= exc[None, :]
        self.W *= self._gain
        sous = np.ix_(self.i_kc, self.i_mbon)
        self.W[sous] = self.W[sous] * self.gain_mbon
        # Cablage de naissance, capture APRES tous les gains : c'est la cible de la
        # recuperation lente. Le capturer avant ferait tendre la lecture KC->MBON
        # vers 1/gain_mbon de sa valeur de travail -- une fuite constante des poids
        # que rien ne distingue d'un apprentissage qui echoue.
        self.W0 = self.W.copy()
        # NOTE DE MESURE : W n'est remplie qu'a 2,2%. Une version creuse (scipy CSR,
        # transposee pre-calculee) a ete ecrite et VERIFIEE identique au dense sur 12
        # pas, mais mesuree pendant qu'un entrainement tournait en tache de fond :
        # elle ne CONCLUT RIEN. On garde le dense parce qu'il est plus simple.
        self.v = np.zeros(self.n, dtype=np.float32)
        self.spk = np.zeros(self.n, dtype=np.float32)   # etat de decharge, persiste
        self.refrac = np.zeros(self.n, dtype=np.int32)
        self._version_poids = getattr(self, '_version_poids', 0) + 1
        self._idx_d = None          # le circuit a change de taille
        self._Wd_jeton = None

    # ------------------------------------------------------------------ etat
    def reset_etat(self):
        with self._lock:
            self.v.fill(0.0)
            self.spk.fill(0.0)
            self.refrac.fill(0)

    def _valider_pn(self, pn) -> np.ndarray:
        try:
            x = np.asarray(pn, dtype=np.float32).reshape(len(self.i_pn))
        except (ValueError, TypeError) as err:
            raise ValueError(
                f"Entree PN invalide : attendu {len(self.i_pn)} nombres. ({err})") from err
        if not np.all(np.isfinite(x)):
            raise ValueError("Entree PN invalide : NaN ou infini.")
        return np.clip(x, 0.0, 1.0)

    # ------------------------------------------------------------------- pas
    def pas_temps(self, pn, dan: np.ndarray | None = None) -> dict:
        """Un pas de temps. `pn` pilote les neurones de projection (l'odeur),
        `dan` (optionnel) l'activite dopaminergique (la recompense/punition).

        L'etat de decharge PERSISTE d'un pas au suivant (`self.spk`) : sans lui le
        reseau est purement feed-forward sur une seule synapse, les Kenyon cells
        n'atteignent jamais les MBON, et les seuls MBON actifs sont ceux que les
        ALPN touchent en direct -- le corps pedoncule est court-circuite en silence.
        """
        with self._lock:
            x = self._valider_pn(pn)
            entree = self.spk.copy()                 # ce qui a decharge au pas precedent
            entree[self.i_pn] = x                    # les PN sont pilotes de l'exterieur
            if dan is not None:
                d = np.clip(np.asarray(dan, dtype=np.float32).reshape(len(self.i_dan)), 0, 1)
                entree[self.i_dan] = d
            else:
                entree[self.i_dan] = 0.0

            # APL est gradue : sa sortie suit son entree, sans seuil. Le passer en
            # LIF le rend muet et supprime toute l'inhibition laterale (mesure).
            a_kc = self.W[np.ix_(self.i_kc, self.i_apl)][:, 0]
            entree[self.i_apl] = min(1.0, float(entree[self.i_kc] @ a_kc) * self.gain_apl)

            courant = entree @ self.W
            self.v = self.decay * self.v + courant
            self.v[self.i_pn] = 0.0                  # pilotes, pas integres
            self.v[self.i_apl] = 0.0                 # gradue, pas d'integration
            self.v[self.refrac > 0] = 0.0
            emet = self.v >= self.v_thr
            emet[self.i_pn] = False
            emet[self.i_apl] = False                 # gradue, jamais "tout ou rien"
            v_mbon = self.v[self.i_mbon].copy()
            self.v[emet] = 0.0
            self.refrac = np.maximum(0, self.refrac - 1)
            self.refrac[emet] = self.t_refrac

            self.spk = emet.astype(np.float32)
            self.spk[self.i_pn] = x
            self.spk[self.i_apl] = entree[self.i_apl]

            self.pas += 1
            self.spikes_total += int(emet.sum())
            return {"spikes": self.spk.copy(),
                    "kc": self.spk[self.i_kc].copy(), "mbon": self.spk[self.i_mbon].copy(),
                    "apl": float(self.spk[self.i_apl][0]),
                    "v_mbon": v_mbon,
                    "taux_kc": float(self.spk[self.i_kc].mean())}

    def repondre(self, pn, n_pas: int = 8, dan=None) -> dict:
        """Presente une entree plusieurs pas et accumule la reponse des MBON."""
        self.reset_etat()
        n_pas = max(1, min(int(n_pas), 100))
        acc_m = np.zeros(len(self.i_mbon), dtype=np.float32)
        acc_kc = np.zeros(len(self.i_kc), dtype=np.float32)
        taux = []
        for _ in range(n_pas):
            o = self.pas_temps(pn, dan=dan)
            acc_m += o["mbon"] + 0.25 * np.clip(o["v_mbon"], 0, None)
            acc_kc += o["kc"]
            taux.append(o["taux_kc"])
        return {"mbon": acc_m / n_pas, "kc": acc_kc / n_pas,
                "taux_kc": float(np.mean(taux))}

    # ------------------------------------------------------------------- lot
    def repondre_lot(self, PN: np.ndarray, n_pas: int = 8) -> dict:
        """Meme calcul que `repondre`, mais sur B entrees A LA FOIS.

        Interet : le pas de temps passe d'un produit matrice-VECTEUR a un produit
        matrice-MATRICE. Meme nombre d'operations, mais la version vectorielle est
        limitee par la bande passante memoire (on relit toute W pour un seul
        vecteur) alors que la version matricielle sature le processeur. Mesure sur
        238 exemples : 1025 ms en serie contre 75 ms en lot, soit x14 -- sans GPU.
        """
        with self._lock:
            X = np.asarray(PN, dtype=np.float32)
            if X.ndim != 2 or X.shape[1] != len(self.i_pn):
                raise ValueError(
                    f"Lot invalide : attendu (B, {len(self.i_pn)}), recu {X.shape}.")
            if not np.all(np.isfinite(X)):
                raise ValueError("Lot invalide : NaN ou infini.")
            X = np.clip(X, 0.0, 1.0)
            B = len(X)
            xp = calcul.xp()
            Wd = self._W_device()
            i_pn, i_kc, i_mbon, i_dan, i_apl = self._idx_device()
            X = calcul.vers_device(X)
            V = xp.zeros((B, self.n), dtype=xp.float32)
            SPK = xp.zeros((B, self.n), dtype=xp.float32)
            REF = xp.zeros((B, self.n), dtype=xp.int32)
            a_kc = Wd[i_kc][:, i_apl][:, 0]
            acc_m = xp.zeros((B, len(self.i_mbon)), dtype=xp.float32)
            acc_kc = xp.zeros((B, len(self.i_kc)), dtype=xp.float32)

            for _ in range(max(1, min(int(n_pas), 100))):
                E = SPK.copy()
                E[:, i_pn] = X
                E[:, i_dan] = 0.0
                E[:, i_apl[0]] = xp.minimum(1.0, (E[:, i_kc] @ a_kc) * self.gain_apl)

                V = self.decay * V + E @ Wd          # <- le produit matrice-matrice
                V[:, i_pn] = 0.0
                V[:, i_apl] = 0.0
                V[REF > 0] = 0.0
                emet = V >= self.v_thr
                emet[:, i_pn] = False
                emet[:, i_apl] = False
                v_mbon = V[:, i_mbon].copy()
                V[emet] = 0.0
                REF = xp.maximum(0, REF - 1)
                REF[emet] = self.t_refrac

                SPK = emet.astype(xp.float32)
                SPK[:, i_pn] = X
                SPK[:, i_apl] = E[:, i_apl]
                acc_m += SPK[:, i_mbon] + 0.25 * xp.clip(v_mbon, 0, None)
                acc_kc += SPK[:, i_kc]

            n = max(1, min(int(n_pas), 100))
            return {"mbon": calcul.vers_hote(acc_m / n),
                    "kc": calcul.vers_hote(acc_kc / n)}

    def _W_device(self):
        """W sur le peripherique de calcul, recopiee seulement quand elle a change.
        La recopier a chaque appel annulerait le gain : 40 Mo par transfert."""
        if not calcul.sur_gpu():
            return self.W
        jeton = (self.W.shape, self._version_poids)
        if getattr(self, "_Wd_jeton", None) != jeton:
            self._Wd = calcul.vers_device(self.W)
            self._Wd_jeton = jeton
        return self._Wd

    def _idx_device(self):
        """Index recopies sur le peripherique. Le cache est vide par `_poser_poids`,
        donc par la neurogenese. NE PAS le faire dependre de la taille des ALPN :
        elle ne change JAMAIS quand le cerveau grandit, et les index des Kenyon
        cells resteraient perimes -- defaut reellement rencontre, 7 tests rouges."""
        if not calcul.sur_gpu():
            return self.i_pn, self.i_kc, self.i_mbon, self.i_dan, self.i_apl
        if getattr(self, "_idx_d", None) is None:
            self._idx_d = tuple(calcul.vers_device(i) for i in
                                (self.i_pn, self.i_kc, self.i_mbon, self.i_dan, self.i_apl))
        return self._idx_d

    # --------------------------------------------------------- apprentissage
    def potentiation_dopaminergique(self, kc_actives: np.ndarray, mbon_cibles: np.ndarray,
                                     taux: float = 0.05, plafond: float = 3.0) -> float:
        """Renforcement de la synapse KC->MBON active, borne par un plafond relatif au
        cablage de naissance. On ne CREE jamais de synapse absente du connectome :
        le cablage est une donnee, seule sa force est plastique."""
        with self._lock:
            k = np.clip(np.asarray(kc_actives, dtype=np.float32).reshape(len(self.i_kc)), 0, 1)
            m = np.clip(np.asarray(mbon_cibles, dtype=np.float32).reshape(len(self.i_mbon)), 0, 1)
            if not (0.0 < float(taux) <= 1.0):
                raise ValueError(f"taux hors ]0,1] : {taux!r}")
            sous = np.ix_(self.i_kc, self.i_mbon)
            w, w0 = self.W[sous], self.W0[sous]
            marge = np.clip(plafond * w0 - w, 0, None)
            self.W[sous] = w + float(taux) * np.outer(k, m) * marge
            self._version_poids += 1
            return float((float(taux) * np.outer(k, m) * marge).sum())

    def apprentissage_delta(self, kc_actives, erreur, taux: float = 0.05) -> float:
        """Erreur signee portee par la dopamine, appliquee UNIQUEMENT aux synapses
        KC->MBON qui existent dans le connectome. C'est la borne haute atteignable
        sans inventer de cablage : le connectome dit qui parle a qui, l'apprentissage
        ne regle que la force."""
        with self._lock:
            k = np.clip(np.asarray(kc_actives, dtype=np.float32).reshape(len(self.i_kc)), 0, 1)
            e = np.asarray(erreur, dtype=np.float32).reshape(len(self.i_mbon))
            if not np.all(np.isfinite(e)):
                raise ValueError("Erreur non finie transmise a l'apprentissage.")
            sous = np.ix_(self.i_kc, self.i_mbon)
            masque = self.W0[sous] != 0
            d = float(taux) * np.outer(k, e) * masque
            self.W[sous] = np.clip(self.W[sous] + d, 0.0, None)   # KC = cholinergiques
            self._version_poids += 1
            return float(np.abs(d).sum())

    def depression_dopaminergique(self, kc_actives: np.ndarray, mbon_cibles: np.ndarray,
                                  taux: float = 0.05) -> float:
        """Regle de la mouche : la coincidence (KC active + dopamine) DEPRIME la
        synapse KC->MBON. `mbon_cibles` vaut 1 pour les MBON a punir."""
        with self._lock:
            k = np.clip(np.asarray(kc_actives, dtype=np.float32).reshape(len(self.i_kc)), 0, 1)
            m = np.clip(np.asarray(mbon_cibles, dtype=np.float32).reshape(len(self.i_mbon)), 0, 1)
            if not (0.0 < float(taux) <= 1.0):
                raise ValueError(f"taux de depression hors ]0,1] : {taux!r}")
            sous = np.ix_(self.i_kc, self.i_mbon)
            w = self.W[sous]
            delta = -float(taux) * np.outer(k, m) * w      # multiplicatif : jamais negatif
            self.W[sous] = w + delta
            self._version_poids += 1
            return float(-delta.sum())

    def recuperation(self, tau: float = 0.002):
        """Retour lent vers le cablage de naissance : sans oubli, tout finit a zero."""
        with self._lock:
            sous = np.ix_(self.i_kc, self.i_mbon)
            self.W[sous] += tau * (self.W0[sous] - self.W[sous])
            self._version_poids += 1

    # ------------------------------------------------------------- croissance
    def croitre(self, n_nouvelles: int = 500) -> dict:
        """Neurogenese : ajoute des Kenyon cells au circuit reel.

        Le cablage des nouvelles n'est PAS tire uniformement. Chaque nouvelle KC
        copie le profil d'une vraie KC prise au hasard (son nombre de griffes, son
        nombre de cibles MBON) puis tire ses partenaires en suivant la frequence
        EMPIRIQUE des vrais partenaires, et ses poids dans la distribution des vrais
        poids. On extrapole la statistique de cablage de la mouche, on n'invente pas
        une architecture.

        ATTENTION : les MBON recoivent alors davantage de sources. Pour que la
        comparaison entre tailles porte sur la representation et non sur un simple
        surcroit de courant, la colonne de chaque MBON est renormalisee -- ce qui
        REECHELONNE les poids deja appris, en preservant leur structure relative.
        """
        with self._lock:
            n = int(n_nouvelles)
            if not (1 <= n <= 20000):
                raise ValueError(f"n_nouvelles doit etre dans [1, 20000] (recu {n_nouvelles!r})")
            i_pn, i_kc, i_mbon, i_apl = self.i_pn, self.i_kc, self.i_mbon, self.i_apl
            brut = self.c.W                       # comptes de synapses signes, non normalises
            pk = brut[np.ix_(i_pn, i_kc)]         # ALPN -> KC
            km = brut[np.ix_(i_kc, i_mbon)]       # KC -> MBON
            ka = brut[np.ix_(i_kc, i_apl)][:, 0]  # KC -> APL
            ak = brut[np.ix_(i_apl, i_kc)][0, :]  # APL -> KC

            # Distributions empiriques a imiter.
            freq_pn = (pk != 0).sum(axis=1).astype(np.float64)
            freq_pn = freq_pn / freq_pn.sum()
            poids_pn = pk[pk != 0]
            freq_mb = (km != 0).sum(axis=0).astype(np.float64)
            freq_mb = freq_mb / freq_mb.sum()
            poids_mb = km[km != 0]
            griffes = (pk != 0).sum(axis=0)
            cibles = (km != 0).sum(axis=1)
            ka_nz, ak_nz = ka[ka != 0], ak[ak != 0]

            n_kc0 = len(i_kc)
            modeles = self.rng.integers(0, n_kc0, size=n)
            new_pk = np.zeros((len(i_pn), n), dtype=np.float32)
            new_km = np.zeros((n, len(i_mbon)), dtype=np.float32)
            for j, m in enumerate(modeles):
                g = int(griffes[m])
                if g > 0:
                    qui = self.rng.choice(len(i_pn), size=min(g, len(i_pn)),
                                          replace=False, p=freq_pn)
                    new_pk[qui, j] = self.rng.choice(poids_pn, size=len(qui))
                c = int(cibles[m])
                if c > 0:
                    vers = self.rng.choice(len(i_mbon), size=min(c, len(i_mbon)),
                                           replace=False, p=freq_mb)
                    new_km[j, vers] = self.rng.choice(poids_mb, size=len(vers))

            # --- Agrandissement de la matrice brute du circuit ------------------
            N, n_tot = self.c.n, self.c.n + n
            def agrandir(A):
                B = np.zeros((n_tot, n_tot), dtype=np.float32)
                B[:N, :N] = A
                return B
            Wb = agrandir(brut)
            neuves = np.arange(N, n_tot)
            Wb[np.ix_(i_pn, neuves)] = new_pk
            Wb[np.ix_(neuves, i_mbon)] = new_km
            Wb[neuves, i_apl[0]] = self.rng.choice(ka_nz, size=n)
            Wb[i_apl[0], neuves] = self.rng.choice(ak_nz, size=n)

            # --- Reconstruction du circuit et des poids de travail --------------
            from .connectome import Circuit
            classe = np.concatenate([self.c.classe, np.full(n, "Kenyon_Cell", dtype=object)])
            vide = np.concatenate([self.c.cell_type, np.full(n, "KC-neoformee", dtype=object)])
            self.c = Circuit(W=Wb, M=agrandir(self.c.M), classe=classe,
                             root_id=np.concatenate([self.c.root_id, -np.arange(1, n + 1)]),
                             cell_type=vide,
                             nt=np.concatenate([self.c.nt, np.full(n, "acetylcholine", dtype=object)]),
                             idx={c: np.where(classe == c)[0] for c in
                                  ("ALPN", "Kenyon_Cell", "MBON", "DAN", "APL")})
            appris = self.W[np.ix_(i_kc, i_mbon)] / np.where(self.W0[np.ix_(i_kc, i_mbon)] != 0,
                                                             self.W0[np.ix_(i_kc, i_mbon)], 1.0)
            self._poser_poids()
            # On rejoue le rapport appris/naissance sur les anciennes synapses.
            anciennes = np.ix_(i_kc, self.i_mbon)
            self.W[anciennes] = self.W0[anciennes] * np.where(self.W0[anciennes] != 0, appris, 0.0)
            return {"ajoutees": n, "kc": len(self.i_kc), "neurones": int(self.c.n),
                    "synapses_plastiques": int((self.W0[np.ix_(self.i_kc, self.i_mbon)] != 0).sum())}

    def apprentissage_delta_lot(self, K, E, taux: float = 0.05) -> float:
        """Erreur signee appliquee sur TOUT un lot d'un coup.

        Mathematiquement identique a B applications successives sur des poids figes
        (la somme des rangs-1 est un produit matriciel), mais en UNE operation au
        lieu de B, et surtout avec un gradient moyenne : l'apprentissage en ligne sur
        un exemple a la fois suit le bruit de chaque exemple.
        """
        with self._lock:
            K = np.asarray(K, dtype=np.float32)
            E = np.asarray(E, dtype=np.float32)
            if K.ndim != 2 or E.ndim != 2 or len(K) != len(E):
                raise ValueError(f"Formes de lot incompatibles : K{K.shape} E{E.shape}")
            if K.shape[1] != len(self.i_kc) or E.shape[1] != len(self.i_mbon):
                raise ValueError(
                    f"Attendu K(B,{len(self.i_kc)}) et E(B,{len(self.i_mbon)}), "
                    f"recu K{K.shape} E{E.shape}")
            if not (np.all(np.isfinite(K)) and np.all(np.isfinite(E))):
                raise ValueError("Lot d'apprentissage : NaN ou infini.")
            sous = np.ix_(self.i_kc, self.i_mbon)
            d = (float(taux) / max(1, len(K))) * (K.T @ E) * (self.W0[sous] != 0)
            self.W[sous] = np.clip(self.W[sous] + d, 0.0, None)   # KC cholinergiques
            self._version_poids += 1
            return float(np.abs(d).sum())

    # ---------------------------------------------------------------- mesure
    def capacite(self) -> dict:
        sous = np.ix_(self.i_kc, self.i_mbon)
        plastiques = int((self.W0[sous] != 0).sum())
        derive = float(np.abs(self.W[sous] - self.W0[sous]).sum())
        return {"neurones": int(self.n),
                "kc": int(len(self.i_kc)), "mbon": int(len(self.i_mbon)),
                "pn": int(len(self.i_pn)), "dan": int(len(self.i_dan)),
                "synapses_ionotropiques": int((self.c.W != 0).sum()),
                "synapses_modulatrices": int((self.c.M != 0).sum()),
                "synapses_plastiques": plastiques,
                "derive_synaptique": round(derive, 4),
                "pas": self.pas, "spikes_total": self.spikes_total}
