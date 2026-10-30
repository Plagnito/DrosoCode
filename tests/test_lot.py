"""Passe en lot et moteur GPU : meme comportement, beaucoup plus vite."""
from __future__ import annotations
import numpy as np
import pytest

from droso import calcul
from droso import corpus as C
from droso.connectome import charger
from droso.encoding import encoder
from droso.network import CerveauMouche
from droso.trainer import MoucheQuiCode


@pytest.fixture(scope="module")
def cerveau():
    return CerveauMouche(circuit=charger())


@pytest.fixture(scope="module")
def entrees():
    ex = C.exemples(k=4, jeu="tout")
    return np.array([encoder(c) for c, _ in ex], dtype=np.float32)


def test_le_lot_donne_les_memes_decisions_que_la_serie():
    """Equivalence de COMPORTEMENT, pas de bits.

    BLAS change l'ordre des sommes entre GEMV et GEMM : les valeurs bougent de ~3e-8,
    et le reseau est a seuil avec 159 neurones a moins de 1e-6 du seuil. Un neurone
    bascule et sa trajectoire diverge -- mesure : 14 valeurs sur 618 000 different
    au-dela d'un lot de 128. Ce qui doit etre identique, c'est la DECISION.
    Un test qui exigerait l'egalite exacte serait rouge sans qu'il y ait de defaut.
    """
    m = MoucheQuiCode()
    m.entrainer(n_pas=120, regle="delta")
    for jeu in ("app", "test"):
        ex = C.exemples(k=m.k, jeu=jeu)
        serie = np.array([m.scores(c)[0] for c, _ in ex])
        lot, _ = m.scores_lot([c for c, _ in ex])
        assert (serie.argmax(1) == lot.argmax(1)).all(), \
            f"{jeu} : le lot ne decide pas comme la serie"


def test_le_lot_refuse_une_forme_fausse(cerveau):
    with pytest.raises(ValueError):
        cerveau.repondre_lot(np.zeros((4, 10), dtype=np.float32))
    with pytest.raises(ValueError):
        cerveau.repondre_lot(np.full((2, 344), np.nan, dtype=np.float32))
    with pytest.raises(ValueError):
        cerveau.repondre_lot(np.zeros(344, dtype=np.float32))   # 1D, pas un lot


def test_un_lot_de_un_egale_la_version_seule(cerveau, entrees):
    a = cerveau.repondre(entrees[0], n_pas=8)
    b = cerveau.repondre_lot(entrees[:1], n_pas=8)
    assert np.allclose(a["mbon"], b["mbon"][0], atol=1e-6)
    assert np.allclose(a["kc"], b["kc"][0], atol=1e-6)


def test_le_lot_ne_depend_pas_de_l_etat_laisse_par_un_appel_precedent(cerveau, entrees):
    """Chaque exemple part d'un cerveau au repos : sinon le resultat dependrait de
    l'ordre des appels, et deux evaluations identiques ne donneraient pas le meme
    chiffre."""
    a = cerveau.repondre_lot(entrees[:16], n_pas=8)["mbon"]
    cerveau.repondre_lot(entrees[16:48], n_pas=8)
    b = cerveau.repondre_lot(entrees[:16], n_pas=8)["mbon"]
    assert np.array_equal(a, b)


def test_le_moteur_dit_ce_qu_il_est():
    """Un repli silencieux du GPU vers le CPU ferait croire a un GPU qui calcule
    alors qu'il n'est meme pas la."""
    i = calcul.info()
    assert i["moteur"] in ("numpy", "cupy")
    assert i["gpu"] is (i["moteur"] == "cupy")
    assert i["detail"], "le moteur doit dire POURQUOI il est celui-la"


@pytest.mark.skipif(not calcul.sur_gpu(), reason="DROSO_GPU=1 non demande")
def test_le_gpu_donne_les_memes_decisions_que_le_cpu(entrees):
    """Lance avec DROSO_GPU=1. Mesure : evaluer passe de 1,01 s a 0,07 s, et les
    precisions sont identiques au chiffre pres."""
    m = MoucheQuiCode()
    e = m.evaluer()
    assert 0.0 <= e["precision_test"] <= 1.0
    lot = m.cerveau.repondre_lot(entrees[:32], n_pas=8)
    assert isinstance(lot["mbon"], np.ndarray), "le resultat doit revenir sur l'hote"
    assert np.all(np.isfinite(lot["mbon"]))
