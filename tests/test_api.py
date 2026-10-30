"""L'API du serveur : elle repond, et elle refuse ce qu'elle doit refuser."""
from __future__ import annotations
import pytest

import app as application
from droso import store


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "DOSSIER", tmp_path / "etat")
    monkeypatch.setattr(application, "_mouche", None)
    # L'amelioration est un singleton construit A L'IMPORT : il a deja lu le journal
    # du vrai dossier data/etat/. Sans ce reset, le test voit l'etat de la machine
    # qui le lance -- et passe ou echoue selon ce qui a tourné avant lui.
    monkeypatch.setattr(application._amelioration, "journal", [])
    application.app.config.update(TESTING=True)
    return application.app.test_client()


def test_accueil_rend_la_page(client):
    r = client.get("/")
    assert r.status_code == 200
    assert b"DrosoCode" in r.data


def test_sante_annonce_le_circuit_et_sa_source(client):
    d = client.get("/api/sante").get_json()
    assert d["ok"] and d["circuit"]["Kenyon_Cell"] == 2597
    assert d["circuit"]["APL"] == 1
    assert "FlyWire" in d["source"], "la provenance des donnees doit etre affichable"


def test_circuit_expose_natives_et_neoformees(client):
    c = client.get("/api/circuit").get_json()["capacite"]
    assert c["kc_natives"] == 2597 and c["kc_neoformees"] == 0
    assert c["canaux_utilises"] == c["vocabulaire"]


def test_proposer_rend_un_jeton_du_vocabulaire(client):
    from droso import corpus as C
    d = client.post("/api/proposer", json={"contexte": "def f ( x"}).get_json()
    assert d["ok"] and d["proposition"] in C.VOCAB
    assert len(d["votes"]) == 8 and d["kc_total"] == 2597


@pytest.mark.parametrize("corps,motif", [
    ({"contexte": ""}, "vide"),
    ({"contexte": "lambda yield"}, "hors vocabulaire"),
    ({"contexte": 42}, "chaine"),
])
def test_proposer_refuse_les_entrees_fausses(client, corps, motif):
    r = client.post("/api/proposer", json=corps)
    assert r.status_code == 400
    assert motif in r.get_json()["erreur"]


@pytest.mark.parametrize("corps", [{"pas": 0}, {"pas": 99999}, {"pas": "trois"}])
def test_entrainer_refuse_les_bornes(client, corps):
    assert client.post("/api/entrainer", json=corps).status_code == 400


def test_entrainer_puis_croitre_puis_relire(client):
    a = client.post("/api/entrainer", json={"pas": 60, "regle": "delta"}).get_json()
    assert a["ok"] and a["lecons"] == 60
    b = client.post("/api/croitre", json={"kc": 300}).get_json()
    assert b["ok"] and b["kc"] == 2897
    c = client.get("/api/circuit").get_json()["capacite"]
    assert c["kc_neoformees"] == 300 and c["lecons"] == 60


def test_croitre_refuse_une_taille_absurde(client):
    assert client.post("/api/croitre", json={"kc": 0}).status_code == 400
    assert client.post("/api/croitre", json={"kc": 10 ** 9}).status_code == 400


def test_regle_inconnue_rend_400(client):
    r = client.post("/api/entrainer", json={"pas": 5, "regle": "magie"})
    assert r.status_code == 400 and "regle" in r.get_json()["erreur"]


def test_resultats_absents_rendent_404_avec_la_commande(client, monkeypatch, tmp_path):
    monkeypatch.setattr(application, "ROOT", tmp_path)
    r = client.get("/api/croissance")
    assert r.status_code == 404
    assert "courbe_croissance" in r.get_json()["erreur"], \
        "un 404 doit dire COMMENT produire la mesure manquante"


def test_amelioration_au_repos(client):
    d = client.get("/api/amelioration").get_json()
    assert d["ok"] and d["en_cours"] is False and d["journal"] == []
