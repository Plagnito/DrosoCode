"""Bride le corpus pour toute la suite.

Le corpus reel compte 341 extraits et 7771 exemples : `evaluer()` les parcourt tous
a chaque appel, et la suite depassait 15 minutes. On le bride a 24 extraits AVANT
tout import de `droso`, car le corpus est charge au chargement du module.

⚠️ La bride est posee par variable d'environnement et n'existe PAS en production :
reduire le corpus en douce cote produit fausserait tous les chiffres sans qu'aucun
test ne le voie.
"""
import os

os.environ.setdefault("DROSO_MAX_EXTRAITS", "24")
