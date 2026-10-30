"""Extrait le corps pedoncule du connectome MALE CNS (Google Research + HHMI Janelia).

Publie le 3 septembre 2026 dans Cell : 166 000 neurones, 125 millions de synapses,
le systeme nerveux CENTRAL complet d'une drosophile male. CC-BY.
Telechargement : https://male-cns.janelia.org/download/ (ouvert, sans compte).

Produit le MEME format que `extract_mb.py` (data/circuit/mb_mcns*.npz), pour que le
reste du projet tourne dessus sans une ligne de changement.

    python scripts/extract_mb_mcns.py --side both

⚠️ LE MEME PIEGE QUE FLYWIRE, VERIFIE ICI AUSSI : `predicted_nt` classe 4058 des
4064 Kenyon cells en « dopamine », alors que `consensus_nt` les donne TOUTES
cholinergiques. Ce n'est donc pas une bizarrerie de FlyWire mais une propriete du
classifieur de neurotransmetteurs sur les Kenyon cells. On lit `ground_truth`, puis
`consensus_nt`, et `predicted_nt` seulement en dernier recours.
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.feather as feather

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from droso.config import RAW, CIRCUIT, NT_SIGN  # noqa: E402

MCNS = RAW / "mcns"
ANN = MCNS / "body-annotations-male-cns-v1.0-minconf-0.5.feather"
NT = MCNS / "body-neurotransmitters-male-cns-v1.0.feather"
POIDS = MCNS / "connectome-weights-male-cns-v1.0-minconf-0.5.feather"
CLASSES = ["ALPN", "Kenyon_Cell", "MBON", "DAN"]
COTE = {"R": "right", "L": "left"}


def nt_de(*champs) -> str:
    """Premier neurotransmetteur reconnu, dans l'ordre de fiabilite des champs."""
    for c in champs:
        if isinstance(c, str):
            bas = c.lower()
            for nom in NT_SIGN:
                if nom in bas:
                    return nom
    return "inconnu"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--side", default="both", choices=["both", "left", "right"])
    ap.add_argument("--min-syn", type=int, default=1)
    a = ap.parse_args()
    for f in (ANN, NT, POIDS):
        if not f.exists():
            raise SystemExit(f"Fichier absent : {f}\nVoir https://male-cns.janelia.org/download/")

    # --- 1. Les neurones du circuit ----------------------------------------
    ann = feather.read_table(ANN, columns=["bodyId", "type", "class", "somaSide"]).to_pandas()
    ann["side"] = ann.somaSide.map(COTE)
    cotes = ["left", "right"] if a.side == "both" else [a.side]
    sel = ann[(ann["class"].isin(CLASSES)) & (ann.side.isin(cotes))].copy()
    apl = ann[(ann.type == "APL") & (ann.side.isin(cotes))].copy()
    if len(apl) != len(cotes):
        raise SystemExit(f"APL attendu en {len(cotes)} exemplaire(s), trouve {len(apl)}.")
    apl["class"] = "APL"
    sel = pd.concat([sel, apl], ignore_index=True).drop_duplicates(subset=["bodyId"])

    nt = feather.read_table(NT, columns=["body", "ground_truth", "consensus_nt",
                                         "predicted_nt"]).to_pandas()
    sel = sel.merge(nt, left_on="bodyId", right_on="body", how="left")
    sel["nt"] = [nt_de(g, c, p) for g, c, p in
                 zip(sel.ground_truth, sel.consensus_nt, sel.predicted_nt)]
    sel["signe"] = sel.nt.map(NT_SIGN).fillna(0.0)

    ordre = {c: i for i, c in enumerate(CLASSES + ["APL"])}
    sel = sel.sort_values(by=["class", "bodyId"],
                          key=lambda s: s.map(ordre) if s.name == "class" else s)
    sel = sel.reset_index(drop=True)
    idx = {int(r): i for i, r in enumerate(sel.bodyId)}
    print(f"[1/3] {len(sel)} neurones ({'+'.join(cotes)}) :")
    for c in CLASSES + ["APL"]:
        sous = sel[sel["class"] == c]
        print(f"      {c:<12} {len(sous):>5}   {sous.nt.value_counts().to_dict()}")

    # --- 2. Les aretes internes au circuit ----------------------------------
    cache = CIRCUIT / f"aretes_mcns_{a.side}_brutes.parquet"
    if cache.exists():
        print(f"[2/3] cache : {cache.name}")
        e = pd.read_parquet(cache)
    else:
        print("[2/3] lecture des 151,9 M connexions (une seule fois, memory-mapped)...")
        t = feather.read_table(POIDS, memory_map=True)
        ids = pa.array(list(idx), type=t["body_pre"].type)
        garde = pc.and_(pc.is_in(t["body_pre"], value_set=ids),
                        pc.is_in(t["body_post"], value_set=ids))
        e = t.filter(garde).to_pandas()
        e = e.groupby(["body_pre", "body_post"], as_index=False).weight.sum()
        CIRCUIT.mkdir(parents=True, exist_ok=True)
        e.to_parquet(cache, index=False)
    avant = len(e)
    e = e[e.weight >= a.min_syn]
    print(f"      {avant} paires connectees, {len(e)} retenues (>= {a.min_syn} synapses)")

    # --- 3. Matrices signee et modulatrice ----------------------------------
    n = len(sel)
    W = np.zeros((n, n), dtype=np.float32)
    M = np.zeros((n, n), dtype=np.float32)
    signe = sel.signe.to_numpy(dtype=np.float32)
    pre = e.body_pre.map(idx).to_numpy()
    post = e.body_post.map(idx).to_numpy()
    poids = e.weight.to_numpy(dtype=np.float32)
    W[pre, post] = poids * signe[pre]
    mod = signe[pre] == 0.0
    M[pre[mod], post[mod]] = poids[mod]

    nom = f"mb_mcns_{a.side}"
    np.savez_compressed(CIRCUIT / f"{nom}.npz", W=W, M=M,
                        classe=sel["class"].to_numpy(dtype=object),
                        root_id=sel.bodyId.to_numpy(dtype=np.int64),
                        cell_type=sel.type.fillna("").to_numpy(dtype=object),
                        nt=sel.nt.to_numpy(dtype=object))
    meta = {"side": a.side, "n_neurones": int(n), "min_syn": a.min_syn,
            "n_aretes": int((W != 0).sum()),
            "n_aretes_modulatrices": int((M != 0).sum()),
            "synapses_total": int(e.weight.sum()),
            "par_classe": sel["class"].value_counts().to_dict(),
            "source": "Male CNS v1.0 (Google Research + HHMI Janelia, Cell 2026, CC-BY)"}
    (CIRCUIT / f"{nom}.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[3/3] ecrit data/circuit/{nom}.npz — {meta['n_aretes']} aretes, "
          f"{meta['synapses_total']} synapses")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
