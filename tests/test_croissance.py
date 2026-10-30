"""Neurogenese, persistance, amelioration automatique."""
from __future__ import annotations
import numpy as np
import pytest

from droso import corpus as C
from droso import store
from droso.amelioration import Amelioration
from droso.connectome import charger
from droso.network import CerveauMouche
from droso.trainer import MoucheQuiCode


@pytest.fixture
def m():
    return MoucheQuiCode(cerveau=CerveauMouche(circuit=charger()))


def bloc(cb):
    return cb.c.W[np.ix_(cb.i_kc, cb.i_mbon)]


# ---------------------------------------------------------------- croissance
def test_la_croissance_imite_la_statistique_du_vrai_cablage(m):
    """Les nouvelles Kenyon cells ne sont pas cablees uniformement : elles copient
    le profil de vraies KC. Si les statistiques derivaient, la comparaison entre
    tailles ne mesurerait plus la taille mais un changement d'architecture."""
    cb = m.cerveau
    avant_pk = cb.c.W[np.ix_(cb.i_pn, cb.i_kc)]
    g0 = (avant_pk != 0).sum(axis=0)
    d0, cib0 = (bloc(cb) != 0).mean(), (bloc(cb) != 0).sum(axis=1).mean()

    m.croitre(1500)
    apres_pk = cb.c.W[np.ix_(cb.i_pn, cb.i_kc)]
    g1 = (apres_pk != 0).sum(axis=0)
    assert abs(np.median(g1[g1 > 0]) - np.median(g0[g0 > 0])) <= 2
    assert abs((bloc(cb) != 0).mean() - d0) < 0.05
    assert abs((bloc(cb) != 0).sum(axis=1).mean() - cib0) < 2.0


def test_la_croissance_ajoute_des_synapses_plastiques(m):
    avant = int((m.cerveau.W0[np.ix_(m.cerveau.i_kc, m.cerveau.i_mbon)] != 0).sum())
    r = m.croitre(600)
    assert r["kc"] == 2597 + 600
    assert r["synapses_plastiques"] > avant


def test_le_circuit_reste_fonctionnel_apres_croissance(m):
    m.croitre(800)
    ctx, _ = C.exemples(k=m.k)[8]
    s, kc = m.scores(ctx)
    assert len(s) == len(C.VOCAB)
    assert kc.sum() > 0, "plus aucune Kenyon cell n'emet apres la neurogenese"
    assert np.all(np.isfinite(s))


def test_les_neurones_neoformes_sont_cholinergiques(m):
    """Une Kenyon cell est excitatrice. En creer d'inhibitrices retournerait le
    signe de la sortie du corps pedoncule sans lever d'erreur."""
    m.croitre(300)
    cb = m.cerveau
    km = bloc(cb)
    assert (km[km != 0] > 0).all()
    assert set(cb.c.nt[cb.i_kc]) == {"acetylcholine"}


@pytest.mark.parametrize("mauvais", [0, -5, 999999])
def test_croissance_hors_bornes_refusee(m, mauvais):
    with pytest.raises(ValueError):
        m.cerveau.croitre(mauvais)


# --------------------------------------------------------------- persistance
def test_aller_retour_de_sauvegarde(m, tmp_path, monkeypatch):
    monkeypatch.setattr(store, "DOSSIER", tmp_path / "etat")
    m.entrainer(n_pas=120, regle="delta")
    m.croitre(300)
    attendu = m.evaluer(limite=40)["precision"]
    store.sauver(m, [{"cycle": 1}])

    neuve = MoucheQuiCode(cerveau=CerveauMouche(circuit=charger()))
    assert store.charger_dans(neuve) is True
    assert neuve.lecons == m.lecons
    assert len(neuve.cerveau.i_kc) == len(m.cerveau.i_kc)
    assert neuve.evaluer(limite=40)["precision"] == attendu
    assert store.journal() == [{"cycle": 1}]


def test_sans_sauvegarde_le_chargement_rend_faux(m, tmp_path, monkeypatch):
    monkeypatch.setattr(store, "DOSSIER", tmp_path / "vide")
    assert store.charger_dans(m) is False


def test_une_sauvegarde_illisible_ne_fait_pas_tomber(m, tmp_path, monkeypatch):
    d = tmp_path / "etat"
    d.mkdir()
    monkeypatch.setattr(store, "DOSSIER", d)
    (d / "cerveau.json").write_text("{ pas du json", encoding="utf-8")
    assert store.etat_sauve() is None
    assert store.charger_dans(m) is False


def test_l_ecriture_atomique_refuse_un_fichier_vide(tmp_path, monkeypatch):
    """DEFAUT VERROUILLE : np.savez_compressed ajoute '.npz' quand le nom n'en a
    pas. Le temporaire sans suffixe restait VIDE, os.replace le deplacait, et la
    sauvegarde annonçait 'sauve' en laissant zero octet."""
    monkeypatch.setattr(store, "DOSSIER", tmp_path)
    with pytest.raises(IOError):
        store._ecrire_atomique(tmp_path / "x.npz", lambda p: None)
    assert not (tmp_path / "x.npz").exists()


# ------------------------------------------------------- amelioration automatique
def test_amelioration_enchaine_les_cycles(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "DOSSIER", tmp_path / "etat")
    partagee = MoucheQuiCode(cerveau=CerveauMouche(circuit=charger()))
    a = Amelioration(lambda: partagee)
    a.demarrer(cycles=2, lecons=30, croissance=100, seuil=1.0)
    a._fil.join(timeout=180)
    st = a.statut()
    assert st["erreur"] is None, st["erreur"]
    assert st["etat"] == "termine"
    assert len(st["journal"]) == 2
    assert st["journal"][0]["neurogenese"] == 100, "seuil a 1.0 => croissance a chaque cycle"
    assert partagee.lecons == 60


def test_amelioration_refuse_un_second_depart(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "DOSSIER", tmp_path / "etat")
    partagee = MoucheQuiCode(cerveau=CerveauMouche(circuit=charger()))
    a = Amelioration(lambda: partagee)
    a.demarrer(cycles=1, lecons=20, croissance=0)
    try:
        with pytest.raises(ValueError):
            a.demarrer(cycles=1, lecons=20, croissance=0)
    finally:
        a.arreter()
        a._fil.join(timeout=120)


@pytest.mark.parametrize("kw", [{"cycles": 0}, {"cycles": 99}, {"lecons": 0},
                                {"croissance": -1}, {"seuil": 2.0}])
def test_parametres_hors_bornes_refuses(kw):
    a = Amelioration(lambda: None)
    base = {"cycles": 2, "lecons": 10, "croissance": 10, "seuil": 0.01}
    with pytest.raises(ValueError):
        a.demarrer(**{**base, **kw})
