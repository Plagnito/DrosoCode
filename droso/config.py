"""Reglages centraux de DrosoCode (tout surchargeable par variable d'environnement)."""
from __future__ import annotations
import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
CIRCUIT = ROOT / "data" / "circuit"

# Signe de l'effet post-synaptique par neurotransmetteur.
# Chez la drosophile le glutamate est majoritairement INHIBITEUR (recepteur GluCl-alpha),
# contrairement aux vertebres : ne pas le compter comme excitateur.
NT_SIGN = {
    "acetylcholine": +1.0,
    "gaba": -1.0,
    "glutamate": -1.0,
    # neuromodulateurs : aucun effet ionotropique direct, ils portent l'apprentissage
    "dopamine": 0.0,
    "octopamine": 0.0,
    "serotonin": 0.0,
}


@dataclass(frozen=True)
class Settings:
    host: str = os.environ.get("DROSO_HOST", "127.0.0.1")
    port: int = int(os.environ.get("DROSO_PORT", "5001"))
    debug: bool = os.environ.get("DROSO_DEBUG", "0") == "1"
    seed: int = int(os.environ.get("DROSO_SEED", "42"))
    # Hemisphere retenu : un seul, pour un circuit propre et lateralement coherent.
    side: str = os.environ.get("DROSO_SIDE", "right")
    # Nom du circuit a charger, sans extension. Vide = `mb_<side>`.
    # Ex. : mb_both (FlyWire femelle, 2 hem.) ou mb_mcns_both (Male CNS).
    circuit: str = os.environ.get("DROSO_CIRCUIT", "")
    # Budget total d'ALPN actifs, tenu quelle que soit la longueur du contexte.
    # 0 = pas de budget (12 par jeton toujours), l'ancien comportement.
    budget_alpn: int = int(os.environ.get("DROSO_BUDGET_ALPN", "48"))
    gain: float = float(os.environ.get("DROSO_GAIN", "4.0"))
    gain_apl: float = float(os.environ.get("DROSO_GAIN_APL", "3.0"))
    gain_mbon: float = float(os.environ.get("DROSO_GAIN_MBON", "20.0"))
    # Seuil de synapses par paire. CHOISI PAR MESURE, pas par habitude : un balayage
    # 1..5 montre qu'un seuil de 5 fait tomber la densite KC->MBON de 18,9% a 6,0%
    # et le nombre de KC par MBON de 379 a 100, tres en dessous de la litterature.
    # Les paires KC->MBON sont peu synaptiques par nature : les seuiller les efface.
    min_syn: int = int(os.environ.get("DROSO_MIN_SYN", "1"))
    version: str = "0.1.0"


SETTINGS = Settings()
