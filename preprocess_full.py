#!/usr/bin/env python
"""Build the FULL entity-complete processed datasets: SMD_FULL, SMAP_FULL, MSL_FULL.

Standalone on purpose: preprocess.py and every existing processed/ directory are left
untouched, so the default protocol and all existing results stay reproducible.

Per-dataset rules, chosen to PRESERVE each dataset's existing representation:

  SMD_FULL   all 28 machines. Byte-for-byte the same pipeline as preprocess.py's
             load_and_save/load_and_save2: np.genfromtxt of the raw txt with NO
             normalization call (SMD ships pre-normalised to [0,1] upstream), and
             labels built from interpretation_label with the same off-by-one
             convention temp[start-1:end-1, indx] = 1.

  SMAP_FULL  every unique SMAP channel in labeled_anomalies.csv. normalize3 fitted on
             TRAIN only, applied to test -- identical to preprocess.py.
  MSL_FULL   same, for MSL.

Entities stay independent: one train/test/labels triplet per entity, never concatenated,
so no window can span two entities (main.py windows a single entity's array at a time).

P-2 (SMAP) appears twice in labeled_anomalies.csv with DIFFERENT anomaly_sequences
(row 17 -> [[5350, 6575]], row 51 -> [[5300, 6420]]). Conflicting duplicates are resolved
by taking the FIRST matching row, which is exactly what preprocess.py does via
values[...]['anomaly_sequences'].values[0]; this keeps P-2 bit-for-bit compatible with the
existing processed/SMAP. The conflict and its resolution are recorded in the manifest under
conflict_resolutions, so the discarded annotation is never silently lost.

Writes processed/<DS>_FULL/<entity>_{train,test,labels}.npy plus manifest.json.
Refuses to overwrite a non-empty output directory unless --force is given.
"""
import argparse, hashlib, json, os, sys, datetime
import numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
md5 = lambda p: hashlib.md5(open(p, "rb").read()).hexdigest()


def normalize3(a, min_a=None, max_a=None):           # identical to preprocess.py
    if min_a is None: min_a, max_a = np.min(a, axis=0), np.max(a, axis=0)
    return (a - min_a) / (max_a - min_a + 0.0001), min_a, max_a


def save(folder, ent, tr, te, lb):
    for name, arr in (("train", tr), ("test", te), ("labels", lb)):
        np.save(os.path.join(folder, f"{ent}_{name}.npy"), arr.astype("float64"))


def entry(folder, ent, tr, te, lb, srcs, norm):
    fh = {f"{ent}_{n}.npy": md5(os.path.join(folder, f"{ent}_{n}.npy")) for n in ("train", "test", "labels")}
    anyf = (lb.sum(1) >= 1).astype(int)
    return {"entity": ent, "source_files": srcs,
            "train_shape": list(tr.shape), "test_shape": list(te.shape), "label_shape": list(lb.shape),
            "feature_count": int(tr.shape[1]),
            "anomaly_count_rows_any_feature": int(anyf.sum()),
            "anomaly_count_label_cells": int(lb.sum()),
            "anomaly_rate_rows": round(float(anyf.mean()), 6),
            "normalization": norm, "file_md5": fh}


def build_smd(out, force):
    src = os.path.join(ROOT, "data", "SMD")
    files = sorted(f for f in os.listdir(os.path.join(src, "train")) if f.endswith(".txt"))
    ents = []
    for fn in files:
        ent = fn[:-4]
        tr = np.genfromtxt(os.path.join(src, "train", fn), dtype=np.float64, delimiter=",")
        te = np.genfromtxt(os.path.join(src, "test", fn), dtype=np.float64, delimiter=",")
        lb = np.zeros(te.shape)
        with open(os.path.join(src, "interpretation_label", fn)) as f:
            for line in f.readlines():
                pos, vals = line.split(":")[0], line.split(":")[1].split(",")
                s, e = int(pos.split("-")[0]), int(pos.split("-")[1])
                idx = [int(i) - 1 for i in vals]
                lb[s - 1:e - 1, idx] = 1          # preprocess.py convention, preserved
        save(out, ent, tr, te, lb)
        ents.append(entry(out, ent, tr, te, lb,
                          {"train": f"data/SMD/train/{fn}", "test": f"data/SMD/test/{fn}",
                           "labels": f"data/SMD/interpretation_label/{fn}"},
                          "none (SMD ships pre-normalised to [0,1]; preprocess.py applies no scaling)"))
        print(f"  SMD  {ent:14s} train{list(tr.shape)} test{list(te.shape)} labels{list(lb.shape)}", flush=True)
    return ents, [], []


def build_smap_msl(ds, out, force):
    src = os.path.join(ROOT, "data", "SMAP_MSL")
    v = pd.read_csv(os.path.join(src, "labeled_anomalies.csv"))
    v = v[v.spacecraft == ds]
    ents, excluded, conflicts = [], [], []
    for chan in sorted(v.chan_id.unique()):
        rows = v[v.chan_id == chan]
        if len(rows) > 1:
            uniq = rows.drop_duplicates()
            if len(uniq) > 1:
                # Conflicting duplicate: resolve by taking the FIRST matching row, identical to
                # preprocess.py's .values[0]. Both annotations are recorded in the manifest.
                sel = rows.index[0]
                conflicts.append({
                    "entity": chan,
                    "issue": f"{chan} has conflicting duplicate metadata in labeled_anomalies.csv",
                    "rows": {f"row {i}": rows.loc[i, "anomaly_sequences"] for i in rows.index},
                    "selected_row": int(sel),
                    "selected_anomaly_sequences": rows.loc[sel, "anomaly_sequences"],
                    "discarded_anomaly_sequences": [rows.loc[i, "anomaly_sequences"] for i in rows.index if i != sel],
                    "resolution": ("row %d selected to preserve compatibility with the existing TranAD "
                                   "preprocessing, which resolves duplicate %s entries by taking the first "
                                   "matching anomaly_sequences value" % (sel, chan)),
                    "identical_fields": [c for c in rows.columns if rows[c].nunique() == 1],
                    "conflicting_fields": [c for c in rows.columns if rows[c].nunique() > 1],
                })
                print(f"  {ds}  {chan:8s} CONFLICT resolved -> row {sel} {rows.loc[sel,'anomaly_sequences']} "
                      f"(discarded {[rows.loc[i,'anomaly_sequences'] for i in rows.index if i != sel]})", flush=True)
        tp, ep = f"{src}/train/{chan}.npy", f"{src}/test/{chan}.npy"
        if not (os.path.exists(tp) and os.path.exists(ep)):
            excluded.append({"entity": chan, "reason": "train and/or test .npy missing"}); continue
        tr_raw, te_raw = np.load(tp), np.load(ep)
        tr, mn, mx = normalize3(tr_raw)                     # train statistics only
        te, _, _ = normalize3(te_raw, mn, mx)               # applied to test
        lb = np.zeros(te.shape)
        idx = [int(i) for i in str(rows.anomaly_sequences.values[0]).replace("]", "").replace("[", "").split(", ")]
        for i in range(0, len(idx), 2):
            lb[idx[i]:idx[i + 1], :] = 1
        save(out, chan, tr, te, lb)
        ents.append(entry(out, chan, tr, te, lb,
                          {"train": f"data/SMAP_MSL/train/{chan}.npy", "test": f"data/SMAP_MSL/test/{chan}.npy",
                           "labels": "data/SMAP_MSL/labeled_anomalies.csv (anomaly_sequences)"},
                          "per-entity min-max (normalize3) fitted on TRAIN only, applied to test"))
        print(f"  {ds}  {chan:8s} train{list(tr.shape)} test{list(te.shape)} labels{list(lb.shape)}", flush=True)
    return ents, excluded, conflicts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["SMD_FULL", "SMAP_FULL", "MSL_FULL"])
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    for full in a.datasets:
        base = full.replace("_FULL", "")
        out = os.path.join(ROOT, "processed", full)
        if os.path.isdir(out) and os.listdir(out) and not a.force:
            sys.exit(f"refusing to overwrite non-empty {out} (use --force)")
        assert not os.path.exists(os.path.join(ROOT, "processed", base, "__from_full__")), "sanity"
        os.makedirs(out, exist_ok=True)
        print(f"\n=== building {full} -> processed/{full}/", flush=True)
        ents, exc, conf = (build_smd(out, a.force) if base == "SMD" else build_smap_msl(base, out, a.force))
        man = {"dataset": full, "base_dataset": base,
               "created": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
               "generator": "preprocess_full.py",
               "protocol_note": "entities are independent train/test/labels triplets; never concatenated, "
                                "so no window can cross an entity boundary",
               "window_note": "windowing happens inside main.py per loaded entity array (W=10)",
               "entity_count": len(ents), "entities": ents, "excluded_entities": exc,
               "conflict_resolutions": conf}
        with open(os.path.join(out, "manifest.json"), "w") as f:
            json.dump(man, f, indent=2)
        print(f"  -> {len(ents)} entities, {len(exc)} excluded, {len(conf)} conflict(s) resolved, manifest.json written", flush=True)


if __name__ == "__main__":
    main()
