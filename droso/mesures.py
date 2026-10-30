"""Les bornes hautes du circuit — mesurees une seule fois, au meme endroit.

ATTENTION, DEFAUT CORRIGE : une premiere version estimait le plafond sous contrainte
de cablage en prenant la solution LIBRE puis en la masquant. Ce n'est pas la
meilleure solution contrainte, c'est une solution optimisee pour d'autres poids a
qui on en retire une partie -- et elle annonçait 31,9 % la ou le vrai plafond
contraint est de 83,6 %. Le connectome ne coupe presque rien ; l'estimateur, si.
Le bon calcul regresse chaque MBON sur ses PROPRES afferentes.
"""
from __future__ import annotations

import numpy as np

from . import corpus as C

RIDGE = 1e-3


def motifs(m, jeu: str = "tout"):
    """Motif de Kenyon cells et cible, pour chaque exemple du jeu demande."""
    ex = C.exemples(k=m.k, jeu=jeu)
    _, K = m.scores_lot([c for c, _ in ex])      # une seule passe en lot
    y = np.array([C.ID[t] for _, t in ex])
    return K.astype(np.float64), y


def ridge_dual(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Regression ridge, en choisissant la forme la MOINS CHERE des deux.

    Le cout est celui du cote le plus petit : forme duale (systeme n_exemples^2)
    quand il y a peu d'exemples et beaucoup de Kenyon cells, forme primale
    (n_traits^2) dans le cas inverse. Choisir une fois pour toutes rend le calcul
    inabordable des qu'on change d'echelle -- avec 5858 exemples et une regression
    par MBON, la forme duale seule demanderait 41 systemes de 5858x5858.
    """
    n, d = X.shape
    if d <= n:
        A = X.T @ X + RIDGE * np.eye(d)
        return np.linalg.solve(A, X.T @ Y)
    G = X @ X.T
    return X.T @ np.linalg.solve(G + RIDGE * np.eye(n), Y)


def plafond_libre(K: np.ndarray, y: np.ndarray,
                  Kt: np.ndarray | None = None, yt: np.ndarray | None = None) -> float:
    """Ce que les motifs de Kenyon cells contiennent, pour un lecteur sans contrainte.

    Le lecteur est TOUJOURS ajuste sur `K`/`y` ; s'il recoit `Kt`/`yt` il est note
    dessus. Sans ca on mesure ce qu'un lecteur parfait retient, pas ce qu'il
    generalise -- et les deux different beaucoup.
    """
    Y = np.zeros((len(y), len(C.VOCAB)))
    Y[np.arange(len(y)), y] = 1
    W = ridge_dual(K, Y)
    Ke, ye = (K, y) if Kt is None else (Kt, yt)
    return float(((Ke @ W).argmax(1) == ye).mean())


def plafond_connectome(K: np.ndarray, y: np.ndarray, m,
                       Kt: np.ndarray | None = None, yt: np.ndarray | None = None) -> float:
    """Meilleur lecteur possible SOUS la contrainte du cablage reel : chaque MBON
    n'est regresse que sur les Kenyon cells qui lui parlent vraiment.
    Ajuste sur `K`/`y`, note sur `Kt`/`yt` quand ils sont fournis."""
    cb = m.cerveau
    W0 = cb.W0[np.ix_(cb.i_kc, cb.i_mbon)]
    Ke, ye = (K, y) if Kt is None else (Kt, yt)
    scores = np.zeros((len(ye), len(C.VOCAB)))
    for t, canal in enumerate(m.canaux):
        aff = np.flatnonzero(W0[:, canal])
        if len(aff) == 0:
            scores[:, t] = -np.inf
            continue
        w = ridge_dual(K[:, aff], (y == t).astype(np.float64)[:, None])
        scores[:, t] = (Ke[:, aff] @ w).ravel()
    return float((scores.argmax(1) == ye).mean())


def ridge_positif(X: np.ndarray, y1: np.ndarray, iterations: int = 3000) -> np.ndarray:
    """Ridge sous contrainte de POSITIVITE, par gradient projete.

    Indispensable pour une cible honnete : les Kenyon cells sont cholinergiques,
    donc le reseau ne peut JAMAIS former un poids KC->MBON negatif. Une borne
    calculee sans cette contrainte vise un optimum que la biologie interdit.

    ⚠️ CONVERGENCE MESUREE, parce que le resultat en depend : 100 iterations
    donnent 54,3 %, 300 -> 55,7 %, 1000 -> 56,7 %, 3000 -> 57,1 %. La premiere
    version en faisait 300 et annoncait donc une valeur basse comme si elle etait
    l'optimum. On en fait 3000, et la valeur est un MINORANT, jamais un maximum.
    """
    n, d = X.shape
    G = X.T @ X + RIDGE * np.eye(d)
    b = X.T @ y1
    pas = 1.0 / (np.linalg.norm(G, 2) + 1e-9)
    w = np.zeros(d)
    for _ in range(int(iterations)):
        w = np.maximum(0.0, w - pas * (G @ w - b))
    return w


def plafond_biologique(K, y, m, Kt=None, yt=None) -> float:
    """Meilleure LECTURE LINEAIRE du code des Kenyon cells sous les deux contraintes
    du circuit : uniquement les synapses qui existent, uniquement des poids positifs.

    ⚠️ CE N'EST PAS UN PLAFOND DU RESEAU, et le reseau le depasse (65,4 % contre
    57,1 % mesures). C'est normal : le reseau n'est pas une fonction lineaire du
    vecteur de Kenyon cells moyenne dans le temps -- il a 386 aretes MBON->MBON, une
    periode refractaire, et son score accumule aussi le potentiel de membrane sur
    huit pas. Une sonde lineaire jette tout cela. A lire comme un REPERE : ce que
    l'information contenue dans le code des KC donne a un lecteur lineaire honnete.
    """
    cb = m.cerveau
    W0 = cb.W0[np.ix_(cb.i_kc, cb.i_mbon)]
    Ke, ye = (K, y) if Kt is None else (Kt, yt)
    scores = np.zeros((len(ye), len(C.VOCAB)))
    for t, canal in enumerate(m.canaux):
        aff = np.flatnonzero(W0[:, canal])
        if len(aff) == 0:
            scores[:, t] = -np.inf
            continue
        w = ridge_positif(K[:, aff], (y == t).astype(np.float64))
        scores[:, t] = Ke[:, aff] @ w
    return float((scores.argmax(1) == ye).mean())


def plafonds(m) -> dict:
    """Quatre bornes : ce qu'un lecteur parfait RETIENT (patrons appris) et ce qu'il
    GENERALISE (patrons retenus), avec et sans la contrainte du cablage."""
    K, y = motifs(m, jeu="app")
    Kt, yt = motifs(m, jeu="test")
    cb = m.cerveau
    W0 = cb.W0[np.ix_(cb.i_kc, cb.i_mbon)]
    return {"lecture_libre": round(plafond_libre(K, y), 4),
            "lecture_libre_test": round(plafond_libre(K, y, Kt, yt), 4),
            "lecture_contrainte_par_le_connectome":
                round(plafond_connectome(K, y, m), 4),
            "lecture_contrainte_test": round(plafond_connectome(K, y, m, Kt, yt), 4),
            "biologique": round(plafond_biologique(K, y, m), 4),
            "biologique_test": round(plafond_biologique(K, y, m, Kt, yt), 4),
            "part_de_synapses_reelles": round(float((W0 != 0).mean()), 4)}
