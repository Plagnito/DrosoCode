"""Ce que le connectome doit dire — et ce qu'il ne doit PAS dire.

Plusieurs de ces tests verrouillent des defauts reellement rencontres pendant la
construction : chacun est annote avec ce qu'il empeche de revenir.
"""
from __future__ import annotations
import numpy as np
import pytest

from droso.connectome import charger
from droso.config import NT_SIGN


@pytest.fixture(scope="module")
def c():
    return charger()


def test_effectifs_conformes_a_la_publication(c):
    r = c.resume()
    assert r["Kenyon_Cell"] == 2597, "FlyWire annonce 2597 KC a droite"
    assert r["MBON"] == 48
    assert r["APL"] == 1, "APL est UN neurone geant, pas une population"
    assert r["ALPN"] == 344


def test_toutes_les_kenyon_cells_sont_cholinergiques(c):
    """DEFAUT VERROUILLE : le predicteur de neurotransmetteur classe 5172 des 5177
    KC en 'dopamine'. Se fier a `top_nt` rendrait toute la sortie du corps
    pedoncule modulatrice -- donc de signe 0, donc absente de W -- en silence."""
    nt = c.nt[c.idx["Kenyon_Cell"]]
    assert set(nt) == {"acetylcholine"}


def test_signes_coherents_avec_le_neurotransmetteur(c):
    kc_mbon = c.bloc("Kenyon_Cell", "MBON")
    assert (kc_mbon[kc_mbon != 0] > 0).all(), "KC cholinergiques => excitatrices"
    apl_kc = c.bloc("APL", "Kenyon_Cell")
    assert (apl_kc[apl_kc != 0] < 0).all(), "APL GABAergique => inhibiteur"


def test_le_glutamate_est_inhibiteur_chez_la_drosophile():
    """Piege classique : chez les vertebres le glutamate excite, chez la drosophile
    il inhibe majoritairement (recepteur GluCl-alpha)."""
    assert NT_SIGN["glutamate"] < 0


def test_apl_est_cable_sur_la_quasi_totalite_des_kenyon_cells(c):
    n_kc = len(c.idx["Kenyon_Cell"])
    recoit = (c.bloc("Kenyon_Cell", "APL") != 0).sum()
    renvoie = (c.bloc("APL", "Kenyon_Cell") != 0).sum()
    assert recoit > 0.95 * n_kc and renvoie > 0.95 * n_kc


def test_nombre_de_griffes_par_kenyon_cell(c):
    """Chaque KC echantillonne une poignee de neurones de projection (~5).
    DEFAUT VERROUILLE : un seuil de 5 synapses par paire faisait tomber la densite
    KC->MBON de 18,9% a 6,0% et effacait de vraies connexions."""
    pk = c.bloc("ALPN", "Kenyon_Cell")
    griffes = (pk != 0).sum(axis=0)
    assert 3 <= np.median(griffes[griffes > 0]) <= 8
    densite = (c.bloc("Kenyon_Cell", "MBON") != 0).mean()
    assert densite > 0.12, f"densite KC->MBON trop faible ({densite:.1%}) : seuil trop dur ?"


def test_les_dan_survivent_a_la_mise_en_signe(c):
    """DEFAUT VERROUILLE : les neuromodulateurs ont un signe 0, donc les DAN
    disparaissaient de W -- et avec eux tout le signal d'apprentissage."""
    assert (c.bloc("DAN", "MBON", mod=True) != 0).sum() > 0
    assert (c.bloc("DAN", "Kenyon_Cell", mod=True) != 0).sum() > 0
    assert (c.bloc("DAN", "MBON") == 0).all(), "un DAN n'a pas d'effet ionotropique"


def test_certains_mbon_sont_inexploitables(c):
    """Le vrai circuit est tres inegal : des MBON n'ont aucune Kenyon cell
    afferente. Les ignorer n'est pas un choix esthetique, c'est une contrainte."""
    kc_par_mbon = (c.bloc("Kenyon_Cell", "MBON") != 0).sum(axis=0)
    assert kc_par_mbon.min() == 0
    assert kc_par_mbon.max() > 1500
