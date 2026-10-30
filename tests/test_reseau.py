"""Le reseau doit calculer quelque chose — et le chemin doit passer par le cerveau."""
from __future__ import annotations
import numpy as np
import pytest

from droso import corpus as C
from droso.connectome import charger
from droso.encoding import encoder, recouvrement
from droso.network import CerveauMouche
from droso.trainer import MoucheQuiCode


@pytest.fixture(scope="module")
def circuit():
    return charger()


@pytest.fixture(scope="module")
def odeur():
    return (np.random.default_rng(0).random(344) < 0.10).astype(np.float32)


# --------------------------------------------------------------------- reseau
def test_les_kenyon_cells_atteignent_les_mbon(circuit, odeur):
    """DEFAUT VERROUILLE : sans etat de decharge persistant, le reseau etait
    feed-forward sur UNE synapse. Les KC n'atteignaient jamais les MBON et les
    seuls MBON actifs etaient ceux que les ALPN touchent en direct -- le corps
    pedoncule etait court-circuite sans qu'aucune erreur ne le signale."""
    b = CerveauMouche(circuit=circuit)
    b.reset_etat()
    for _ in range(10):
        o = b.pas_temps(odeur)
    assert o["kc"].sum() > 0, "aucune Kenyon cell n'a decharge"
    # contre-epreuve : en coupant KC->MBON, la reponse des MBON doit changer.
    b2 = CerveauMouche(circuit=circuit)
    b2.W[np.ix_(b2.i_kc, b2.i_mbon)] = 0.0
    b2.reset_etat()
    for _ in range(10):
        o2 = b2.pas_temps(odeur)
    assert not np.array_equal(o["v_mbon"], o2["v_mbon"]), \
        "couper KC->MBON ne change rien : les MBON ne lisent pas le corps pedoncule"


def test_apl_inhibe_reellement(circuit, odeur):
    """DEFAUT VERROUILLE : modelise en neurone a seuil, APL n'emettait JAMAIS et
    le couper ne changeait rien -- alors que le code pretendait que la parcimonie
    du codage venait de lui. Il est gradue."""
    def taux(reseau):
        reseau.reset_etat()
        for _ in range(10):
            o = reseau.pas_temps(odeur)
        return o["taux_kc"], o["apl"]

    intact, sortie_apl = taux(CerveauMouche(circuit=circuit))
    coupe = CerveauMouche(circuit=circuit)
    coupe.W[coupe.i_apl, :] = 0.0
    sans, _ = taux(coupe)
    assert sortie_apl > 0, "APL est muet"
    assert sans > 1.5 * intact, f"APL n'inhibe pas ({intact:.2%} avec, {sans:.2%} sans)"


def test_le_cablage_de_naissance_est_capture_apres_les_gains(circuit):
    """DEFAUT VERROUILLE : W0 capture avant l'application du gain KC->MBON faisait
    tendre la recuperation vers 1/gain de la valeur de travail -- une fuite lente
    des poids indiscernable d'un apprentissage qui echoue."""
    b = CerveauMouche(circuit=circuit)
    sous = np.ix_(b.i_kc, b.i_mbon)
    avant = b.W[sous].sum()
    for _ in range(200):
        b.recuperation()
    assert b.W[sous].sum() == pytest.approx(avant, rel=1e-6)


def test_la_plasticite_ne_cree_aucune_synapse(circuit):
    """Le connectome dit qui parle a qui ; l'apprentissage ne regle que la force."""
    b = CerveauMouche(circuit=circuit)
    sous = np.ix_(b.i_kc, b.i_mbon)
    absentes = b.W0[sous] == 0
    b.apprentissage_delta(np.ones(len(b.i_kc)), np.ones(len(b.i_mbon)), taux=0.5)
    assert (b.W[sous][absentes] == 0).all()


def test_la_depression_ne_rend_jamais_une_synapse_negative(circuit):
    b = CerveauMouche(circuit=circuit)
    for _ in range(200):
        b.depression_dopaminergique(np.ones(len(b.i_kc)), np.ones(len(b.i_mbon)), taux=0.9)
    assert (b.W[np.ix_(b.i_kc, b.i_mbon)] >= 0).all()


@pytest.mark.parametrize("mauvais", [np.full(344, np.nan), np.zeros(10), "texte"])
def test_entrees_invalides_refusees(circuit, mauvais):
    with pytest.raises(ValueError):
        CerveauMouche(circuit=circuit).pas_temps(mauvais)


# -------------------------------------------------------------------- codage
def test_encodage_deterministe_et_discriminant():
    a, b, loin = ["def", "f", "(", "x"], ["def", "f", "(", "y"], ["while", "i", "<", "n"]
    assert np.array_equal(encoder(a), encoder(a))
    assert recouvrement(a, b) > 3 * recouvrement(a, loin)


def test_le_budget_d_alpn_agit_reellement():
    """DEFAUT VERROUILLE : `n_actifs` etait une VALEUR PAR DEFAUT lisant la globale du
    module. Une valeur par defaut est evaluee A LA DEFINITION de la fonction, donc
    changer `N_ALPN_PAR_JETON` n'avait aucun effet -- le banc qui fait varier ce
    parametre mesurait deux fois la meme chose, avec des chiffres identiques au
    centieme, et rien ne le signalait."""
    from droso import encoding as E
    ctx = ["def", "f", "(", "x"]
    a = int((E.encoder(ctx) > 0).sum())
    b = int((E.encoder(ctx, n_actifs=4) > 0).sum())
    assert b < a, f"n_actifs ne change rien ({a} contre {b})"
    ancien = E.N_ALPN_PAR_JETON
    try:
        E.N_ALPN_PAR_JETON = 3
        E._glomerules.cache_clear()
        c = int((E.encoder(ctx) > 0).sum())
    finally:
        E.N_ALPN_PAR_JETON = ancien
        E._glomerules.cache_clear()
    assert c < a, "la globale du module n'est pas relue a l'appel"


def test_le_budget_plafonne_mais_ne_gonfle_jamais():
    """Un budget doit PLAFONNER l'entree, jamais la densifier. Defaut du premier banc :
    `budget / k` rendait 24 ALPN par jeton a k=2 (au lieu de 12), donc la condition
    « budget » testait une entree PLUS dense — l'inverse de l'hypothese."""
    from droso import encoding as E
    for k in (1, 2, 4):
        assert E.alpn_par_jeton(k, 48) <= E.N_ALPN_PAR_JETON
    assert E.alpn_par_jeton(16, 48) < E.N_ALPN_PAR_JETON
    assert E.alpn_par_jeton(64, 48) >= 2, "au moins 2 ALPN par jeton"
    assert E.alpn_par_jeton(16, 0) == E.N_ALPN_PAR_JETON, "budget 0 = desactive"


def test_le_budget_garde_l_entree_eparse_quand_le_contexte_s_allonge():
    """Chaque jeton allume ses propres ALPN, donc un contexte plus long densifie
    l'entree : a 12 ALPN/jeton et k=16, 44% des 344 ALPN sont allumes et deux
    contextes partagent 56% de leurs actifs. Le budget constant corrige ca."""
    from droso import encoding as E
    long_ctx = ["x"] * 16
    # ⚠️ Comparer au DEFAUT ne prouve rien depuis que le budget est actif par defaut :
    # les deux cotes seraient la meme chose. On compare budget ETEINT contre ALLUME.
    dense = (E.encoder(long_ctx, budget=0) > 0).mean()
    epars = (E.encoder(long_ctx, budget=48) > 0).mean()
    assert epars < dense / 2, f"le budget ne desature pas ({dense:.1%} -> {epars:.1%})"
    assert (E.encoder(long_ctx) > 0).mean() == epars, "le budget doit etre actif par defaut"


def test_encodage_refuse_un_contexte_vide():
    with pytest.raises(ValueError):
        encoder([])


def test_l_ordre_des_jetons_compte():
    assert not np.array_equal(encoder(["x", "=", "y", "NL"]), encoder(["y", "=", "x", "NL"]))


# -------------------------------------------------------------------- corpus
def test_le_vocabulaire_tient_sous_le_nombre_de_mbon(circuit):
    """Contrainte fondatrice : un jeton = un MBON. Au-dela il faudrait un
    classifieur, et ce ne serait plus la mouche qui decide."""
    assert len(C.VOCAB) <= len(circuit.idx["MBON"])


def test_tous_les_patrons_sont_dans_le_vocabulaire():
    for p in C.PATRONS:
        C.jetons(p)          # leve si un jeton manque


def test_exemples_bien_formes():
    ex = C.exemples(k=4)
    assert all(len(ctx) == 4 and cible in C.ID for ctx, cible in ex)
    with pytest.raises(ValueError):
        C.exemples(k=0)


# -------------------------------------------------------------- apprentissage
def test_une_lecon_modifie_le_cablage_et_reste_locale(circuit):
    m = MoucheQuiCode(cerveau=CerveauMouche(circuit=circuit))
    cb = m.cerveau
    sous = np.ix_(cb.i_kc, cb.i_mbon)
    avant = cb.W[sous].copy()
    ctx, cible = C.exemples(k=m.k)[8]
    m.lecon(ctx, cible, taux=0.2, regle="delta")
    bouge = ~np.isclose(cb.W[sous], avant)
    assert bouge.any(), "la lecon n'a rien change"
    assert bouge.mean() < 0.5, "la lecon devrait etre locale, pas globale"


def test_regle_inconnue_refusee(circuit):
    m = MoucheQuiCode(cerveau=CerveauMouche(circuit=circuit))
    ctx, cible = C.exemples(k=m.k)[0]
    with pytest.raises(ValueError):
        m.lecon(ctx, cible, regle="magie")


def test_cible_hors_vocabulaire_refusee(circuit):
    m = MoucheQuiCode(cerveau=CerveauMouche(circuit=circuit))
    with pytest.raises(ValueError):
        m.lecon(["def", "f", "(", "x"], "lambda")


def test_les_canaux_retenus_ont_tous_des_afferentes(circuit):
    m = MoucheQuiCode(cerveau=CerveauMouche(circuit=circuit))
    cb = m.cerveau
    kc_par_mbon = (cb.W0[np.ix_(cb.i_kc, cb.i_mbon)] != 0).sum(axis=0)
    assert (kc_par_mbon[m.canaux] >= 50).all()
    assert len(set(m.canaux.tolist())) == len(C.VOCAB), "un MBON ne sert qu'un jeton"
