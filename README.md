# DrosoCode — apprendre a coder a un vrai cerveau de mouche

> On prend le **corps pedoncule** d'une drosophile, tel que le connectome FlyWire
> le donne — neurone par neurone, synapse par synapse — et on lui demande de
> predire le prochain jeton d'un programme.

Ce n'est pas un reseau « bio-inspire ». **Aucune connexion n'est tiree au hasard** :
les 2 597 Kenyon cells, leurs 5 griffes chacune, le neurone geant APL et les 48 MBON
viennent d'un cerveau reel reconstruit par microscopie electronique.

## Les donnees — DEUX connectomes

**FlyWire** (drosophile femelle, cerveau, Princeton + MRC-LMB + Vermont, *Nature* 2024) :

| Source | Contenu | Licence |
|---|---|---|
| [Zenodo 10676866](https://zenodo.org/records/10676866) | `proofread_connections_783.feather` — 16,8 M connexions | CC-BY-4.0 |
| [flywire_annotations](https://github.com/flyconnectome/flywire_annotations) | types cellulaires des 139 248 neurones | depot public |

**Male CNS** (systeme nerveux central complet d'un male, Google Research + HHMI Janelia,
*Cell*, 3 septembre 2026 — 166 000 neurones, 125 M de synapses) :
[male-cns.janelia.org/download](https://male-cns.janelia.org/download/), CC-BY.
Fichiers `body-annotations` (13,8 Mo), `body-neurotransmitters` (41,3 Mo) et
`connectome-weights` (1,00 Go), a placer dans `data/raw/mcns/`.

Aucun compte n'est requis pour l'un ni pour l'autre. ⚠️ Codex (portail FlyWire) et
l'API neuPrint, eux, exigent un compte — mais ce sont d'autres chemins d'acces, pas
ceux-ci. Verifier avant de conclure qu'un jeu de donnees est ferme.

```bash
python scripts/extract_mb.py                     # FlyWire, DROSO_SIDE=right|left|both
python scripts/extract_mb_mcns.py --side both    # Male CNS
DROSO_CIRCUIT=mb_mcns_both python app.py         # basculer de circuit
```

### Ce que les deux connectomes donnent

| | FlyWire femelle (2 hem.) | Male CNS (2 hem.) |
|---|---|---|
| Kenyon cells | **5 177** | 4 064 |
| ALPN | 685 | 686 |
| MBON (= plafond du vocabulaire) | 96 | **97** |
| Griffes par KC (mediane) | 6 | 6 |
| Densite KC→MBON | 12,5 % | 15,5 % |

### Comparer deux connectomes : apparier le REGIME d'abord

```bash
python scripts/calibrer_circuit.py        # amene un circuit a la parcimonie cible
python scripts/comparer_connectomes.py    # calibre puis entraine, 3 graines
```

⚠️ **A gains identiques, on ne compare pas deux cablages mais deux POINTS DE
FONCTIONNEMENT.** Mesure : FlyWire tourne a 13 % de Kenyon cells actives, le Male CNS
a **4,5 %** — trois fois plus epars, parce que ses densites synaptiques different. Le
verdict est passe de « Male CNS perd 9,6 points » a « Male CNS gagne 7,9 points » par
la seule calibration d'un gain, soit un renversement de **17,5 points**.

Rien ne le signalait : les deux circuits tournaient, apprenaient, rendaient des chiffres
plausibles, et le classement collait meme a une explication structurelle toute prete
(« il a 21 % de Kenyon cells en moins »). Une histoire causale satisfaisante ne valide
pas un resultat.

⚠️ **ET LA VARIANCE ENTRE EXECUTIONS EST GRANDE** : changer la seule graine de melange
fait passer le meme circuit de 50,9 % a 56,7 %. Tout ecart annonce sur une execution
unique est suspect — d'ou les trois graines par circuit.

⚠️ **Le corps pedoncule du male est PLUS PETIT.** Le connectome de Google est le plus
gros jamais publie (166 000 neurones contre 139 255), mais son interet est ailleurs —
la corde nerveuse ventrale et les voies descendantes. Pour notre tache il n'apporte
pas plus de corps pedoncule, et il faut le dire avant de tout rebrancher dessus.

⚠️ **LE MEME PIEGE DE NEUROTRANSMETTEUR, SUR LES DEUX JEUX.** FlyWire : `top_nt` classe
5 172 des 5 177 Kenyon cells en « dopamine ». Male CNS : `predicted_nt` en classe
**4 058 sur 4 064** pareil. Les deux jeux ont une colonne de verite terrain
(`known_nt`, `consensus_nt`/`ground_truth`) qui les donne **toutes cholinergiques**.
Ce n'est donc pas une bizarrerie de FlyWire mais une propriete du classifieur sur les
Kenyon cells — et s'y fier rend toute la sortie du corps pedoncule modulatrice, donc
de signe nul, donc absente de la matrice, **sans la moindre erreur**.

```bash
# 1. Telecharger les deux fichiers dans data/raw/
# 2. Extraire le circuit (une seule fois, ~1 min)
python scripts/extract_mb.py
# 3. Lancer
pip install -r requirements.txt
python app.py          # http://127.0.0.1:5001
```

Variables : `DROSO_PORT`, `DROSO_SIDE`, `DROSO_MIN_SYN`, `DROSO_GAIN`,
`DROSO_GAIN_APL`, `DROSO_GAIN_MBON`, `DROSO_SEED`.

## Le circuit extrait

Hemisphere droit, 3 155 neurones, 215 518 synapses ionotropiques + 14 969 modulatrices.

| Classe | Effectif | Role |
|---|---|---|
| ALPN | 344 | l'entree — ici, le contexte de code |
| Kenyon cells | 2 597 | l'expansion eparse, ~5 griffes chacune |
| MBON | 48 | la sortie — **un jeton = un MBON** |
| DAN | 165 | dopamine : le signal d'apprentissage |
| APL | **1** | un seul neurone geant, inhibiteur, cable sur toutes les KC |

**Contrainte fondatrice** : le vocabulaire (41 jetons) tient sous le nombre de MBON.
Il n'y a donc **aucune couche de lecture ajoutee** au-dessus du cerveau — la decision
sort des neurones de sortie de la mouche, tels que le connectome les donne.

## Corpus : du VRAI code Python

```bash
python scripts/construire_corpus.py --max 600   # ecrit data/corpus.json
```

Contrainte infranchissable : un jeton = un MBON, et le circuit n'en a que 48. On ne
peut donc pas elargir le vocabulaire — il faut **reduire le code**. Les identifiants
deviennent `x/y/n/i/f` (le meme identifiant garde le meme nom dans tout l'extrait),
les nombres `0/1/2`, la structure syntaxique reste intacte. Un extrait qui contient
un mot-cle absent du vocabulaire (`is`, `lambda`, `try`…) est **rejete** : le laisser
passer le transformerait en identifiant et enseignerait une grammaire fausse.

Source : la bibliotheque standard Python de la machine. **6 795 definitions de
fonctions lues → 341 extraits distincts**, 7 771 jetons, 94 fichiers d'origine.
A defaut de `data/corpus.json`, on retombe sur les patrons ecrits a la main, en le disant.

## Apprentissage et generalisation

Le corpus est decoupe **par extrait, jamais par exemple** : decouper par exemple
laisserait des contextes du meme extrait des deux cotes, et le score mesurerait de la
memorisation. Mesure a l'appui : sur l'ancien corpus, un decoupage par exemple donnait
13,2 % en apprentissage pour 6,2 % sur du jamais-vu — **sous** la base du jeton majoritaire.

Les bornes hautes se mesurent donc en quatre niveaux, tous sur les extraits RETENUS :

| Borne (generalisee, cerveau natif de 2 597 KC) | |
|---|---|
| Lecteur lineaire libre | 71,2 % |
| Restreint aux synapses KC->MBON qui existent | 64,4 % |
| **+ poids positifs** (les Kenyon cells sont cholinergiques) | **55,7 %** |
| Jeton majoritaire / hasard | 10,6 % / 2,4 % |

⚠️ Ces bornes sont celles du cerveau **natif**. Un cerveau agrandi a les siennes,
plus hautes (60,8 % a 5 597 KC) — ne pas lire un score de mouche neoformee contre la
borne de 2 597 KC.

⚠️ **ET CE NE SONT PAS DES PLAFONDS DU RESEAU.** Ce sont les meilleures lectures
LINEAIRES du code des Kenyon cells, ajustees **par MBON independamment**
(un-contre-tous, moindres carres). Le reseau, lui, entraine tous ses MBON
**conjointement** en competition softmax, directement sur la decision — et sur un
cerveau agrandi il passe devant (65,4 % contre 60,8 %). A taille native il reste
dessous (49,1 % contre 57,1 %). Deux hypotheses ont ete testees et ECARTEES pour
expliquer l'ecart : la recurrence MBON->MBON (couper les 386 aretes ne change que
**0,2 point**) et la ponderation temporelle (55,3 % contre 57,1 %, donc pire).

⚠️ **Le gradient projete doit CONVERGER, et sa valeur en depend** : 100 iterations
donnent 54,3 %, 300 -> 55,7 %, 1000 -> 56,7 %, 3000 -> 57,1 %. La premiere version en
faisait 300 et presentait une valeur basse comme si c'etait l'optimum.

⚠️ **La troisieme ligne est la seule vraie cible.** Une borne calculee sans la
contrainte de positivite vise un optimum que la biologie interdit : le reseau ne peut
JAMAIS former un poids KC->MBON negatif. Elle coute 9 points.

## Ce que ca donne

```bash
python scripts/comparer_regles.py     # regles d'apprentissage + bornes hautes
python scripts/courbe_croissance.py   # taille du cerveau -> precision, + temoin remanie
```
Les deux ecrivent dans `data/`, l'interface les affiche. Les bornes hautes sont
mesurees a part et ne dependent d'aucune regle d'apprentissage.

**Le cablage reel ne coute presque rien** : les motifs de Kenyon cells portent 87,8 %
de precision pour un lecteur libre, et **83,6 %** pour le meilleur lecteur restreint
aux seules synapses KC->MBON qui existent — alors que **18,9 %** seulement des couples
(Kenyon cell, MBON) sont cables. Ce qui manque n'est pas dans le connectome, c'est
dans la regle d'apprentissage.

## Faire grandir le cerveau

La neurogenese ajoute des Kenyon cells au circuit reel. Leur cablage n'est **pas**
tire uniformement : chaque nouvelle copie le profil d'une vraie (nombre de griffes,
nombre de cibles), tire ses partenaires selon la frequence empirique des vrais, et
ses poids dans la distribution des vrais poids. Verifie par test : griffes medianes,
densite KC->MBON et cibles par KC restent celles de la mouche.

L'interface propose une **amelioration automatique** : la mouche s'entraine, et quand
le gain d'un cycle passe sous un seuil, elle fait naitre des Kenyon cells et repart.
L'etat est sauve apres chaque cycle (`data/etat/`), donc un redemarrage ne coute
qu'un cycle.

### Ce que la croissance donne (1 200 lecons par taille)

| Taille | Synapses KC->MBON | Plafond connectome | Appris | Top-3 |
|---|---|---|---|---|
| **2 597 KC** (la vraie mouche) | 23 574 | 83,6 % | 12,6 % | 23,9 % |
| 3 597 KC | 32 707 | 87,4 % | **21,0 %** | 35,3 % |
| 5 597 KC | 50 998 | **87,8 %** | **33,2 %** | 41,6 % |
| 8 597 KC | 78 319 | 87,8 % | 32,4 % | **52,9 %** |
| *temoin : cablage remanie, 2 597 KC* | 23 574 | 87,4 % | *18,1 %* | *35,3 %* |

Deux lectures, et la seconde est desagreable :

1. **Grandir leve la contrainte structurelle.** Des 5 597 Kenyon cells, le plafond
   sous contrainte de cablage rejoint exactement le plafond libre (87,8 %) : le
   goulot du cablage a disparu. Au-dela, la precision top-1 plafonne (32,4 % a
   8 597 KC) alors que le top-3 continue de monter — a nombre de lecons constant, un
   plus gros cerveau a plus de synapses a regler et 1 200 lecons ne suffisent plus.
2. **Le cablage precis de la mouche n'aide PAS pour cette tache.** A taille et degres
   identiques, un cablage remanie au hasard apprend **mieux** (18,1 % contre 12,6 %)
   et a un plafond plus haut (87,4 % contre 83,6 %). La structure reelle du corps
   pedoncule — compartiments, ciblage par type cellulaire — est une contrainte ici,
   pas un avantage. C'est attendu (ce circuit n'a pas evolue pour lire du Python)
   mais c'est le temoin qui permet de le DIRE au lieu de le supposer.

## Huit defauts trouves en construisant — et ce qu'ils apprennent

Chacun est verrouille par un test qui le nomme.

1. **Le predicteur de neurotransmetteur classe 5 172 des 5 177 Kenyon cells en
   « dopamine ».** La litterature les donne *toutes* cholinergiques. Se fier a
   `top_nt` rendait toute la sortie du corps pedoncule modulatrice — donc de signe
   nul, donc absente de la matrice — **sans la moindre erreur**. On lit `known_nt`
   d'abord.
2. **Un seuil de 5 synapses par paire efface le circuit.** Mesure : la densite
   KC→MBON tombe de 18,9 % a 6,0 %, et chaque MBON ne voit plus que 100 Kenyon
   cells au lieu de 379. Les paires KC→MBON sont peu synaptiques **par nature**.
   Le seuil a ete choisi par balayage, pas par habitude.
3. **Aucun etat de decharge ne persistait d'un pas au suivant.** Le reseau etait
   feed-forward sur une seule synapse : les Kenyon cells n'atteignaient jamais les
   MBON, et les seuls MBON « actifs » etaient ceux que les ALPN touchent en direct.
   Le corps pedoncule etait court-circuite, en silence, pendant toute la mise au point.
4. **APL, modelise en neurone a seuil, n'emettait jamais** — et le couper ne
   changeait rien, alors que le code *affirmait* que la parcimonie du codage venait
   de lui. C'est un neurone **gradue**. Une fois corrige : APL intact 2,7 % de KC
   actives, APL coupe 16,4 %. Il divise l'activite par six.
5. **`W0`, la reference de recuperation, etait capturee avant l'application des
   gains** — la recuperation tirait donc en permanence les poids vers un vingtieme
   de leur valeur de travail. Une fuite lente, indiscernable d'un apprentissage
   qui echoue.
6. **Punir tous les MBON actifs n'est pas un apprentissage.** Mesure : 35,7 MBON
   punis sur 41 par lecon, et la precision **baisse**. On ne punit que l'erreur
   reellement commise.
7. **Un mauvais estimateur accusait le connectome a tort.** Le plafond sous contrainte
   de cablage etait calcule en prenant la solution LIBRE puis en la masquant — ce qui
   annoncait **31,9 %** la ou le vrai plafond contraint est de **83,6 %**. Le bon
   calcul regresse chaque MBON sur ses propres afferentes. Conclusion inversee : le
   connectome n'etait pas le goulot, mon estimateur l'etait.
8. **`np.savez_compressed` ajoute `.npz` quand le nom n'en a pas.** L'ecriture
   atomique creait un temporaire sans suffixe, numpy ecrivait a cote, et `os.replace`
   deplacait un fichier de **zero octet** — en annoncant « sauve ».

Et une lecon de methode : les MBON du vrai circuit vont de **0 a 2 089** Kenyon
cells afferentes. Associer naivement « jeton *i* → MBON *i* » faisait gagner
toujours les memes MBON quel que soit le code presente — 7 reponses distinctes sur
41 —, parce que la variation **entre** MBON (0,214) depasse le double de la
variation selon le **contexte** (0,095). On compare donc chaque MBON a sa propre
ligne de base, et on n'utilise que les canaux ayant au moins 50 afferentes.

## Ce que ce projet ne pretend pas

- Ce n'est **pas** une simulation fidele de la drosophile : les gains, les seuils
  et le pas de temps sont des reglages, pas des mesures biologiques.
- La mouche ne « comprend » rien. Elle associe un motif d'activite a une sortie,
  ce que son corps pedoncule fait reellement pour les odeurs.
- La regle biologique pure (depression seule) reste **au niveau du hasard** sur un
  choix a 41 options. C'est un resultat, pas un echec a cacher : ce circuit est fait
  pour trancher entre approcher et fuir, pas pour choisir parmi quarante.
- Les Kenyon cells neoformees imitent la **statistique** du cablage reel, pas une
  mouche plus grosse qui existerait. C'est une extrapolation, et le temoin a cablage
  remanie sert justement a savoir ce que le cablage precis apporte.

## Resultats

Corpus reel, generalisation sur 85 extraits jamais vus (jeton majoritaire 10,6 %) :

| | Generalisation | Top-3 |
|---|---|---|
| Pas d'apprentissage a 0,5 (reglage d'origine) | 17,2 % | 33,1 % |
| Pas a 30, 40 epoques | **49,1 %** | 67,5 % |
| + 3 000 Kenyon cells, 25 epoques | **65,4 %** | **87,5 %** |

Le pas d'apprentissage etait le vrai goulot : 17 % -> 49 % rien qu'en le corrigeant.
La neurogenese ajoute 16 points de plus.

### Le contexte : un manque de CAPACITE, pas d'information

| | 2 597 KC (natif) | 5 597 KC |
|---|---|---|
| k=2 | **50,9 %** | **65,6 %** (top-3 88,4 %) |
| k=4 | 43,1 % | 65,5 % (top-3 87,0 %) |
| k=8 | 37,1 % | — |
| k=16 | 33,5 % | — |

⚠️ **L'avantage du contexte court vaut +7,8 points sur le cerveau natif et DISPARAIT
une fois le cerveau agrandi.** La penalite du contexte long n'etait donc pas une perte
d'information mais un manque de capacite : avec 5 597 Kenyon cells, l'expansion reste
discriminante malgre une entree plus dense. Ne pas retenir « allonger le contexte
degrade » — c'est vrai a taille native seulement.

### Le goulot suivant : la PARCIMONIE, pas le contexte

`scripts/balayage_contexte.py` mesure le repere pour plusieurs longueurs de contexte,
et le resultat est contre-intuitif — **allonger le contexte DEGRADE la mouche** :

| k (jetons de contexte) | 1 | 2 | 4 | 8 | 16 |
|---|---|---|---|---|---|
| Repere biologique | 47,4 % | **60,7 %** | 57,1 % | 53,7 % | 39,1 % |
| **Kenyon cells actives** | 9,8 % | 13,4 % | 21,3 % | 34,0 % | **57,3 %** |

La cause est dans la derniere ligne : l'encodeur **noie le code epars**. A k=16, plus
de la moitie des Kenyon cells repondent, et c'est la parcimonie qui fait toute la
discriminabilite du corps pedoncule.

#### Reguler APL ne sauve rien — et c'est ce refus qui designe la vraie cause

APL est le neurone de controle de gain de la mouche, et on le faisait tourner a gain
FIXE. `scripts/controle_de_gain.py` le regule pour tenir une parcimonie cible :

| k | gain fixe | APL regule |
|---|---|---|
| 8 | 53,7 % (34 % KC) | **50,1 %** (24 % KC, gain 46) |
| 16 | 39,1 % (57 % KC) | **32,0 %** (57 % KC, gain 500 = butee) |

A k=8 la parcimonie baisse **et la precision aussi**. Ce n'est donc pas la parcimonie
des Kenyon cells qui manque : l'information est detruite **en amont**.

#### La saturation est a l'ENTREE, pas dans le corps pedoncule

| k | 1 | 4 | 8 | 16 |
|---|---|---|---|---|
| ALPN allumes (sur 344) | 12 (3,5 %) | 46 (13,3 %) | 85 (24,7 %) | **152 (44,1 %)** |
| Similarite entre deux contextes | 0,111 | 0,213 | 0,341 | **0,563** |

Chaque jeton allume 12 ALPN **fixes**, donc k jetons en allument 12k. A k=16, deux
contextes quelconques partagent **56 % de leurs ALPN actifs** : ils sont deja presque
indiscernables avant d'atteindre les Kenyon cells, et aucune inhibition en aval ne
recupere ca. Le correctif est dans l'encodeur — un **budget d'ALPN constant**
(`scripts/budget_entree.py`).

⚠️ **Defaut verrouille en construisant ce banc** : `encoder(..., n_actifs=GLOBALE)`
fige la valeur par defaut **a la definition de la fonction**. Les deux conditions
testaient la meme chose et rendaient des chiffres identiques au centieme — un
resultat negatif entierement faux, sans la moindre erreur. `n_actifs` est desormais
lu **a l'appel**, et deux tests exigent que le parametre AGISSE.

## Les deux hemispheres, et les canaux GROUPES

Le corps pedoncule est bilateral. `DROSO_SIDE=both` extrait les deux cotes :
685 ALPN, **5 177** Kenyon cells, **96** MBON. Mais les brancher naivement ne suffit
pas, et la mesure a failli me faire conclure l'inverse de la verite.

⚠️ **A 41 jetons, deux hemispheres faisaient PIRE qu'un seul** (50,4 % contre 56,2 %).
Conclusion tentante : « doubler le tissu degrade ». **Elle etait fausse** — un seul
hemisphere n'a que 48 MBON, donc en exiger 41 force a racler les plus faibles, et la
comparaison mesurait cette contrainte de comptage, pas le cablage. A **35 jetons**,
les deux hemispheres GAGNENT (57,3 % contre 54,1 %).

⚠️ **Mais le probleme structurel, lui, est reel** : **74 % des MBON lisent plus de
95 % de leurs Kenyon cells depuis LEUR cote** (81,5 % en moyenne), et 58 % des paires
de canaux sont totalement disjointes. Avec un canal par MBON, les jetons se decident
donc sur des populations separees et le softmax compare des scores calcules sur des
faits differents — deux demi-corps pedoncules qui votent chacun sur sa moitie.

**Le correctif : un canal = un TYPE cellulaire**, donc les MBON homologues des deux
cotes sur la meme decision (`MoucheQuiCode(groupes=True)`).

| Configuration (35 jetons, k=2, 30 epoques, parcimonie appariee) | Generalisation |
|---|---|
| **FlyWire 2 hem., canaux groupes** | **59,6 %** |
| FlyWire 2 hem., canaux simples | 57,3 % |
| Male CNS, canaux groupes | 57,2 % |
| Male CNS, canaux simples | 54,8 % |
| FlyWire 1 hem., canaux simples | 54,1 % |

**35 jetons et non 41** : il n'existe que **35 types** de MBON, et un canal groupe en
demande un. ⚠️ Et un seul hemisphere n'en offre que **30** — la bilateralite n'est pas
qu'un doublement de tissu, c'est ce qui rend le groupement possible.

## Performance

Le gain vient d'abord du **lot**, pas du GPU : on remplace 238 produits
matrice-vecteur (limites par la bande passante memoire) par un produit
matrice-matrice qui sature le processeur.

| Passe complete du corpus | Temps | |
|---|---|---|
| un exemple a la fois | 1 025 ms | — |
| en lot, CPU | 75 ms | **x14** |
| en lot, GPU (`DROSO_GPU=1`, CuPy) | 0,68 ms | **x1 500** |

```bash
DROSO_GPU=1 python app.py      # RTX 3070 detectee, repli NumPy en le DISANT
```

⚠️ **Le lot n'est pas identique au bit pres a la version en serie.** BLAS change
l'ordre des sommes entre GEMV et GEMM : les valeurs bougent de ~3e-8, et le reseau
est a seuil avec **159 neurones a moins de 1e-6 du seuil**. Un neurone bascule, sa
trajectoire diverge — 14 valeurs sur 618 000 different au-dela d'un lot de 128.
Verifie : **0 decision sur 238 change**, et la precision est identique au chiffre
pres. C'est une equivalence de COMPORTEMENT, et un test qui exigerait l'egalite
exacte serait rouge sans qu'il y ait de defaut.

## Tests

```bash
python -m pytest tests/ -q
DROSO_GPU=1 python -m pytest tests/ -q    # le meme, sur GPU
```

## Licence

Le code de DrosoCode est sous licence **MIT** (voir [`LICENSE`](LICENSE)).

Les donnees du connectome ne sont **pas** dans ce depot : elles se telechargent depuis
leurs sources (tableau en tete de ce fichier) et restent sous leur propre licence
(CC-BY 4.0 pour FlyWire). Si vous publiez un resultat obtenu avec, citez FlyWire :

- Dorkenwald et al., *Neuronal wiring diagram of an adult brain*, Nature (2024).
- Schlegel et al., *Whole-brain annotation and multi-connectome cell typing of Drosophila*, Nature (2024).
