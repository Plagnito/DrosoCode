"""DrosoCode — serveur web : apprendre a coder a un vrai cerveau de mouche."""
from __future__ import annotations
import json
import logging
import threading
from pathlib import Path

from flask import Flask, jsonify, render_template, request

from droso import corpus as C
from droso.config import SETTINGS, ROOT
from droso import calcul
from droso.amelioration import Amelioration
from droso.connectome import charger, nom_circuit
from droso import statistiques
from droso import store
from droso.trainer import MoucheQuiCode

app = Flask(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

_verrou = threading.RLock()
_mouche: MoucheQuiCode | None = None


def mouche() -> MoucheQuiCode:
    """La mouche du serveur. Au premier appel, on restaure ce qui a ete appris :
    un redemarrage ne doit pas effacer des heures d'entrainement."""
    global _mouche
    with _verrou:
        if _mouche is None:
            m = MoucheQuiCode()
            try:
                if store.charger_dans(m):
                    app.logger.info("etat restaure : %d lecons, %d KC",
                                    m.lecons, len(m.cerveau.i_kc))
                    # Une seule evaluation au demarrage : sans elle l'interface
                    # affiche un tiret alors qu'un cerveau entraine est en memoire.
                    m.evaluer()
            except Exception as exc:                   # noqa: BLE001
                app.logger.warning("etat sauve illisible, on repart a neuf : %s", exc)
            _mouche = m
        return _mouche


_amelioration = Amelioration(mouche)


def erreur(msg: str, code: int = 400):
    return jsonify({"ok": False, "erreur": msg}), code


@app.errorhandler(ValueError)
def _valeur(e):
    return erreur(str(e), 400)


@app.errorhandler(FileNotFoundError)
def _absent(e):
    return erreur(f"{e} — lancer `python scripts/extract_mb.py`.", 503)


@app.get("/")
def accueil():
    return render_template("index.html", version=SETTINGS.version)


@app.get("/api/sante")
def sante():
    c = charger()
    meta = json.loads((ROOT / "data" / "circuit" / f"{nom_circuit()}.json").read_text())
    return jsonify({"ok": True, "version": SETTINGS.version, "circuit": c.resume(),
                    "source": meta["source"], "side": meta["side"],
                    "aretes": meta["n_aretes"],
                    "aretes_modulatrices": meta["n_aretes_modulatrices"],
                    "moteur": calcul.info()})


@app.get("/api/circuit")
def circuit():
    m = mouche()
    cap = m.cerveau.capacite()
    cap["canaux_utilises"] = len(m.canaux)
    cap["vocabulaire"] = len(C.VOCAB)
    cap["lecons"] = m.lecons
    # Nombre de KC du VRAI circuit, lu dans les metadonnees d'extraction plutot
    # qu'ecrit en dur : il change avec l'hemisphere et avec le seuil de synapses.
    natives = json.loads((ROOT / "data" / "circuit" / f"{nom_circuit()}.json")
                         .read_text())["par_classe"]["Kenyon_Cell"]
    cap["kc_natives"] = int(natives)
    cap["kc_neoformees"] = max(0, int(cap["kc"] - natives))
    return jsonify({"ok": True, "capacite": cap, "corpus": C.stats(),
                    # Cache : `evaluer()` prend ~50 s sur tout le corpus. On rend le
                    # DERNIER score mesure, jamais un recalcul a chaque requete.
                    "dernier": m.dernier,
                    "amelioration": _amelioration.statut()})


@app.get("/api/vocabulaire")
def vocabulaire():
    return jsonify({"ok": True, "jetons": C.VOCAB, "patrons": C.PATRONS})


@app.post("/api/proposer")
def proposer():
    d = request.get_json(silent=True) or {}
    brut = d.get("contexte", "")
    if isinstance(brut, str):
        ctx = brut.split()
    elif isinstance(brut, list):
        ctx = [str(x) for x in brut]
    else:
        return erreur("`contexte` doit etre une chaine ou une liste de jetons.")
    if not ctx:
        return erreur("Contexte vide.")
    inconnus = [j for j in ctx if j not in C.ID]
    if inconnus:
        return erreur(f"Jetons hors vocabulaire : {inconnus}. Vocabulaire : {C.VOCAB}")
    m = mouche()
    ctx = ctx[-m.k:]
    ctx = ["NL"] * (m.k - len(ctx)) + ctx
    s, kc = m.scores(ctx)
    ordre = s.argsort()[::-1]
    return jsonify({"ok": True, "contexte": ctx,
                    "proposition": C.VOCAB[int(ordre[0])],
                    "votes": [{"jeton": C.VOCAB[int(i)], "score": round(float(s[i]), 4)}
                              for i in ordre[:8]],
                    "kc_actives": int((kc > 0).sum()), "kc_total": int(len(kc)),
                    "kc_apercu": [round(float(v), 3) for v in kc[:600]]})


@app.post("/api/entrainer-lot")
def entrainer_lot():
    """Apprentissage par mini-lots : une passe avant en lot, une mise a jour
    moyennee. Beaucoup plus rapide, et meilleur estimateur du gradient."""
    d = request.get_json(silent=True) or {}
    if _amelioration.en_cours:
        return erreur("Une amelioration automatique est en cours.", 409)
    m = mouche()
    r = m.entrainer_lot(epoques=int(d.get("epoques", 6)),
                        taille_lot=int(d.get("taille_lot", 128)),
                        taux=float(d.get("taux", 30.0)))
    store.sauver(m, _amelioration.journal)
    return jsonify({"ok": True, **r})


@app.post("/api/entrainer")
def entrainer():
    d = request.get_json(silent=True) or {}
    n = d.get("pas", 100)
    regle = d.get("regle", "delta")
    if not isinstance(n, int) or isinstance(n, bool) or not (1 <= n <= 2000):
        return erreur("`pas` doit etre un entier dans [1, 2000].")
    m = mouche()
    if _amelioration.en_cours:
        return erreur("Une amelioration automatique est en cours.", 409)
    bilan = m.entrainer(n_pas=n, regle=str(regle))
    bilan.update(m.evaluer(limite=120))
    store.sauver(m, _amelioration.journal)
    return jsonify({"ok": True, **bilan})


@app.get("/api/evaluer")
def evaluer():
    return jsonify({"ok": True, **mouche().evaluer()})


@app.post("/api/reinitialiser")
def reinitialiser():
    global _mouche
    if _amelioration.en_cours:
        return erreur("Une amelioration est en cours : l'arreter d'abord.", 409)
    with _verrou:
        store.oublier()
        _amelioration.journal = []
        _mouche = MoucheQuiCode()
    return jsonify({"ok": True, "message": "Nouveau cerveau, cablage de naissance."})


@app.get("/api/amelioration")
def amelioration_etat():
    return jsonify({"ok": True, **_amelioration.statut()})


@app.post("/api/amelioration/demarrer")
def amelioration_demarrer():
    d = request.get_json(silent=True) or {}
    return jsonify({"ok": True, **_amelioration.demarrer(
        cycles=int(d.get("cycles", 6)), lecons=int(d.get("lecons", 400)),
        croissance=int(d.get("croissance", 800)), seuil=float(d.get("seuil", 0.01)))})


@app.post("/api/amelioration/arreter")
def amelioration_arreter():
    return jsonify({"ok": True, **_amelioration.arreter()})


@app.post("/api/croitre")
def croitre():
    d = request.get_json(silent=True) or {}
    n = d.get("kc", 800)
    if not isinstance(n, int) or isinstance(n, bool) or not (1 <= n <= 20000):
        return erreur("`kc` doit etre un entier dans [1, 20000].")
    m = mouche()
    r = m.croitre(n)
    store.sauver(m, _amelioration.journal)
    return jsonify({"ok": True, **r, **m.evaluer(limite=120)})


@app.get("/api/croissance")
def croissance():
    f = ROOT / "data" / "resultats_croissance.json"
    if not f.exists():
        return jsonify({"ok": False, "erreur": "Pas encore mesure. "
                        "Lancer `python scripts/courbe_croissance.py`."}), 404
    return jsonify({"ok": True, **json.loads(f.read_text(encoding="utf-8"))})


def _fichier(nom: str, commande: str):
    f = ROOT / "data" / nom
    if not f.exists():
        return jsonify({"ok": False,
                        "erreur": f"Pas encore mesure. Lancer `{commande}`."}), 404
    return jsonify({"ok": True, **json.loads(f.read_text(encoding="utf-8"))})


@app.get("/api/stats/circuit")
def stats_circuit():
    return jsonify({"ok": True, **statistiques.circuit(SETTINGS.side)})


@app.get("/api/stats/corpus")
def stats_corpus():
    return jsonify({"ok": True, **statistiques.corpus_stats()})


@app.get("/api/contexte")
def contexte():
    return _fichier("resultats_contexte.json", "python scripts/balayage_contexte.py")


@app.get("/api/temoin")
def temoin():
    return _fichier("temoin_mbon.json", "python scripts/temoin_mbon.py")


@app.get("/api/meilleure")
def meilleure():
    return _fichier("meilleure_mouche.json", "python scripts/meilleure_mouche.py")


@app.get("/api/hemispheres")
def hemispheres():
    return _fichier("resultats_hemispheres.json",
                    "python scripts/comparer_hemispheres.py")


@app.get("/api/connectomes")
def connectomes():
    return _fichier("resultats_connectomes.json",
                    "python scripts/comparer_connectomes.py")


@app.get("/api/contexte-reel")
def contexte_reel():
    return _fichier("resultats_contexte_reel.json",
                    "python scripts/comparer_contexte.py")


@app.get("/api/gain")
def gain():
    return _fichier("resultats_gain.json", "python scripts/controle_de_gain.py")


@app.get("/api/budget")
def budget():
    return _fichier("resultats_budget.json", "python scripts/budget_entree.py")


@app.get("/api/apprentissage")
def apprentissage():
    f = ROOT / "data" / "resultats_apprentissage.json"
    if not f.exists():
        return jsonify({"ok": False, "erreur": "Pas encore mesure. "
                        "Lancer `python scripts/entrainement_long.py`."}), 404
    return jsonify({"ok": True, **json.loads(f.read_text(encoding="utf-8"))})


@app.get("/api/resultats")
def resultats():
    f = ROOT / "data" / "resultats_regles.json"
    if not f.exists():
        return jsonify({"ok": False, "erreur": "Pas encore mesure. "
                        "Lancer `python scripts/comparer_regles.py`."}), 404
    return jsonify({"ok": True, **json.loads(f.read_text(encoding="utf-8"))})


if __name__ == "__main__":
    app.run(host=SETTINGS.host, port=SETTINGS.port, debug=SETTINGS.debug)
