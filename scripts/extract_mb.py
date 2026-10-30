"""Extrait le corps pedoncule (mushroom body) du connectome FlyWire 783.

Entree  : data/raw/proofread_connections_783.feather + annotations TSV
Sortie  : data/circuit/mb_<side>.npz  (matrices) + mb_<side>.json (metadonnees)

Le signe d'une arete vient du neurotransmetteur du neurone PRE-synaptique, en
preferant `known_nt` (verite terrain, litterature) a `top_nt` (predicteur).
C'est indispensable : le predicteur classe 5172 des 5177 Kenyon cells en
"dopamine" alors que la litterature les donne TOUTES cholinergiques.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.feather as feather

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from droso.config import RAW, CIRCUIT, NT_SIGN, SETTINGS  # noqa: E402

# Classes retenues, dans l'ordre des couches du circuit.
CLASSES = ["ALPN", "Kenyon_Cell", "MBON", "DAN"]


def nt_of(known: object, top: object) -> str:
    """Neurotransmetteur d'un neurone : known_nt d'abord, top_nt en repli.

    known_nt est multi-valeur ("acetylcholine; sNPF; acetylcholine, sNPF") :
    on retient le premier nom reconnu comme neurotransmetteur rapide.
    """
    for champ in (known, top):
        if not isinstance(champ, str):
            continue
        bas = champ.lower()
        for nom in NT_SIGN:
            if nom in bas:
                return nom
    return "inconnu"


def main() -> int:
    side = SETTINGS.side
    ann = pd.read_csv(RAW / "Supplemental_file1_neuron_annotations.tsv",
                      sep="\t", low_memory=False)

    # --- 1. Les neurones du circuit, sur un seul hemisphere -----------------
    # `side` vaut "right", "left" ou "both". Les deux hemispheres doublent l'entree
    # (344 -> 685 ALPN) ET le nombre de MBON (48 -> 96), donc le plafond du
    # vocabulaire : c'est lui qui fait rejeter 6019 des 6795 fonctions lues.
    cotes = ["left", "right"] if side == "both" else [side]
    sel = ann[(ann.cell_class.isin(CLASSES)) & (ann.side.isin(cotes))].copy()
    apl = ann[(ann.cell_type == "APL") & (ann.side.isin(cotes))].copy()
    if len(apl) != len(cotes):
        raise SystemExit(
            f"APL attendu en {len(cotes)} exemplaire(s) sur {cotes}, trouve {len(apl)}.")
    apl["cell_class"] = "APL"
    sel = pd.concat([sel, apl], ignore_index=True)

    sel["nt"] = [nt_of(k, t) for k, t in zip(sel.known_nt, sel.top_nt)]
    sel["signe"] = sel.nt.map(NT_SIGN).fillna(0.0)

    # Index stable : trie par classe (ordre CLASSES+APL) puis par root_id.
    ordre = {c: i for i, c in enumerate(CLASSES + ["APL"])}
    sel = sel.drop_duplicates(subset=["root_id"])
    sel = sel.sort_values(by=["cell_class", "root_id"],
                          key=lambda s: s.map(ordre) if s.name == "cell_class" else s)
    sel = sel.reset_index(drop=True)
    idx = {int(r): i for i, r in enumerate(sel.root_id)}
    print(f"[1/3] {len(sel)} neurones ({'+'.join(cotes)}) :")
    for c in CLASSES + ["APL"]:
        sous = sel[sel.cell_class == c]
        nts = sous.nt.value_counts().to_dict()
        print(f"      {c:<12} {len(sous):>5}   {nts}")

    # --- 2. Les aretes internes au circuit ----------------------------------
    cache = CIRCUIT / f"aretes_mb_{side}_brutes.parquet"
    if cache.exists():
        print(f"[2/3] cache : {cache.name}")
        e = pd.read_parquet(cache)
    else:
        print("[2/3] lecture des 16,8 M connexions (une seule fois)...")
        t = feather.read_table(RAW / "proofread_connections_783.feather")
        ids = pa.array(list(idx), type=pa.int64())
        garde = pc.and_(pc.is_in(t["pre_pt_root_id"], value_set=ids),
                        pc.is_in(t["post_pt_root_id"], value_set=ids))
        e = t.filter(garde).to_pandas()
        e = e.groupby(["pre_pt_root_id", "post_pt_root_id"], as_index=False).syn_count.sum()
        CIRCUIT.mkdir(parents=True, exist_ok=True)
        e.to_parquet(cache, index=False)
    avant = len(e)
    e = e[e.syn_count >= SETTINGS.min_syn]
    print(f"      {avant} paires connectees, {len(e)} retenues (>= {SETTINGS.min_syn} synapses)")

    # --- 3. Matrice signee, une seule pour tout le circuit ------------------
    n = len(sel)
    W = np.zeros((n, n), dtype=np.float32)
    signe = sel.signe.to_numpy(dtype=np.float32)
    pre = e.pre_pt_root_id.map(idx).to_numpy()
    post = e.post_pt_root_id.map(idx).to_numpy()
    poids = e.syn_count.to_numpy(dtype=np.float32)
    W[pre, post] = poids * signe[pre]
    # Les neuromodulateurs (dopamine, octopamine, serotonine) n'ont pas d'effet
    # ionotropique : leur signe vaut 0, donc ils DISPARAITRAIENT de W. Or les DAN
    # portent le signal d'apprentissage sur la synapse KC->MBON : on les garde a
    # part, non signes, sinon le circuit perd exactement ce qui le fait apprendre.
    M = np.zeros((n, n), dtype=np.float32)
    mod = signe[pre] == 0.0
    M[pre[mod], post[mod]] = poids[mod]

    CIRCUIT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(CIRCUIT / f"mb_{side}.npz", W=W, M=M,
                        classe=sel.cell_class.to_numpy(dtype=object),
                        root_id=sel.root_id.to_numpy(dtype=np.int64),
                        cell_type=sel.cell_type.fillna("").to_numpy(dtype=object),
                        nt=sel.nt.to_numpy(dtype=object))
    meta = {"side": side, "n_neurones": int(n), "min_syn": SETTINGS.min_syn,
            "n_aretes": int((W != 0).sum()),
            "n_aretes_modulatrices": int((M != 0).sum()),
            "synapses_total": int(e.syn_count.sum()),
            "par_classe": sel.cell_class.value_counts().to_dict(),
            "source": "FlyWire FAFB 783 (Zenodo 10676866, CC-BY-4.0)"}
    (CIRCUIT / f"mb_{side}.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[3/3] ecrit data/circuit/mb_{side}.npz — {meta['n_aretes']} aretes, "
          f"{meta['synapses_total']} synapses")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
