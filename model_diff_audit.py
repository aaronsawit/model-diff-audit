#!/usr/bin/env python3
"""What actually changed between this model and the base it claims to come from?

A downloaded checkpoint that says "fine-tuned from X" can be checked against X. Honest training
moves almost every number in a tensor, and moves all of its bits. Tampering looks different:

    low-bit edits     values differ from the base only in their lowest few bits. Training does not
                      do that. Hiding data in a model does.
    sparse row edits  a handful of rows changed in an embedding or projection while the rest are
                      bit-for-bit identical: planted trigger tokens, or rows used to carry data
    a lone tensor     one layer changed and nothing else

This tool compares two safetensors checkpoints tensor by tensor, at the bit level, without
loading either model or running any code from them.

    model_diff_audit.py BASE SUSPECT              files, or folders of sharded *.safetensors
    model_diff_audit.py BASE SUSPECT --json
    model_diff_audit.py BASE SUSPECT --all        list unchanged tensors too

Exit code: 0 identical or ordinary fine-tune, 1 something needs a human look, 2 could not compare.
Needs numpy. Reads with memory mapping, so models larger than RAM are fine.
"""
import argparse
import json
import struct
import sys
from pathlib import Path

try:
    import numpy as np
except ImportError:                                     # pragma: no cover
    sys.exit("numpy is required:  pip install numpy")

# safetensors dtype -> (bytes per element, unsigned view used for bit comparison)
DTYPES = {"F64": (8, np.uint64), "F32": (4, np.uint32), "F16": (2, np.uint16), "BF16": (2, np.uint16),
          "I64": (8, np.uint64), "I32": (4, np.uint32), "I16": (2, np.uint16), "I8": (1, np.uint8),
          "U8": (1, np.uint8), "BOOL": (1, np.uint8), "F8_E4M3": (1, np.uint8), "F8_E5M2": (1, np.uint8)}
LOW_BITS = 4              # differences confined to this many low bits are not what training produces
SPARSE_ROWS = 0.05        # under 5% of rows touched, the rest identical
CHUNK = 1 << 24


def read_index(path):
    """Map tensor name -> (file, dtype, shape, start, end) for a file or a folder of shards."""
    path = Path(path)
    files = sorted(path.glob("*.safetensors")) if path.is_dir() else [path]
    if not files:
        raise ValueError(f"no .safetensors files in {path}")
    index = {}
    for f in files:
        with open(f, "rb") as fh:
            (header_len,) = struct.unpack("<Q", fh.read(8))
            if header_len > 100_000_000:
                raise ValueError(f"{f}: header length {header_len} is not plausible")
            header = json.loads(fh.read(header_len))
        base = 8 + header_len
        for name, meta in header.items():
            if name == "__metadata__":
                continue
            start, end = meta["data_offsets"]
            index[name] = (f, meta["dtype"], tuple(meta["shape"]), base + start, base + end)
    return index


def bits_view(entry):
    f, dtype, shape, start, end = entry
    size, view = DTYPES[dtype]
    return np.memmap(f, dtype=view, mode="r", offset=start, shape=((end - start) // size,))


def compare_tensor(a, b):
    """Bit-level comparison of two tensors with the same dtype and shape."""
    va, vb = bits_view(a), bits_view(b)
    shape = a[2]
    n = va.shape[0]
    changed, highest_bit, rows_changed = 0, 0, None
    row_len = int(np.prod(shape[1:])) if len(shape) >= 2 else 0
    row_hits = np.zeros(shape[0], dtype=bool) if row_len else None
    for lo in range(0, n, CHUNK):
        x = np.bitwise_xor(va[lo:lo + CHUNK], vb[lo:lo + CHUNK])
        nz = np.flatnonzero(x)
        if nz.size == 0:
            continue
        changed += int(nz.size)
        highest_bit = max(highest_bit, int(x[nz].max()).bit_length())
        if row_hits is not None:
            row_hits[np.unique((nz + lo) // row_len)] = True
    if row_hits is not None:
        rows_changed = int(row_hits.sum())
    result = {"elements": n, "changed": changed, "changed_fraction": changed / n if n else 0.0,
              "highest_changed_bit": highest_bit, "rows": shape[0] if row_len else None,
              "rows_changed": rows_changed}
    if changed and row_hits is not None and rows_changed <= 500:
        result["changed_row_numbers"] = [int(i) for i in np.flatnonzero(row_hits)]
    return result


def judge(stats):
    """Turn one tensor's numbers into a finding, or None if it looks like ordinary training."""
    if not stats["changed"]:
        return None
    if stats["highest_changed_bit"] <= LOW_BITS:
        return (f"low-bit edits: {stats['changed']:,} values differ from the base only in their lowest "
                f"{stats['highest_changed_bit']} bit(s). Training does not produce this; hidden data does.")
    rows, touched = stats["rows"], stats["rows_changed"]
    if rows and rows >= 40 and touched / rows <= SPARSE_ROWS:
        listed = stats.get("changed_row_numbers")
        where = f" (rows {', '.join(map(str, listed[:12]))}{'...' if listed and len(listed) > 12 else ''})" if listed else ""
        return (f"sparse row edits: {touched} of {rows:,} rows changed{where}, every other row is bit-identical. "
                "In an embedding these are specific tokens: check what they are.")
    return None


def audit(base_path, suspect_path):
    base, suspect = read_index(base_path), read_index(suspect_path)
    tensors, findings = [], []
    for name in sorted(set(base) | set(suspect)):
        if name not in suspect:
            note = "present in the base, missing from the suspect"
            if name.endswith("lm_head.weight"):
                note += " (often harmless: models with tied embeddings do not store it)"
            findings.append({"tensor": name, "finding": note})
            continue
        if name not in base:
            findings.append({"tensor": name, "finding": "new tensor that the base does not have"})
            continue
        a, b = base[name], suspect[name]
        if a[1] != b[1] or a[2] != b[2]:
            findings.append({"tensor": name, "finding": f"dtype/shape differ: {a[1]}{list(a[2])} vs {b[1]}{list(b[2])}"})
            continue
        if a[1] not in DTYPES:
            findings.append({"tensor": name, "finding": f"dtype {a[1]} is not supported, not compared"})
            continue
        stats = compare_tensor(a, b)
        stats["tensor"] = name
        tensors.append(stats)
        verdict = judge(stats)
        if verdict:
            findings.append({"tensor": name, "finding": verdict})
    changed = [t for t in tensors if t["changed"]]
    if changed and len(changed) <= max(1, len(tensors) // 50) and len(tensors) >= 10:
        flagged = {f["tensor"] for f in findings}
        for t in changed:
            if t["tensor"] not in flagged:
                findings.append({"tensor": t["tensor"], "finding":
                                 f"lone change: one of only {len(changed)} changed tensor(s) out of {len(tensors)}. "
                                 "Fine-tuning normally touches every trained layer."})
    if not changed and not findings:
        summary = "identical: every tensor matches the base bit for bit"
    elif not findings:
        summary = f"ordinary fine-tune: {len(changed)} of {len(tensors)} tensors changed, densely and in all bits"
    else:
        summary = f"needs a look: {len(findings)} finding(s) across {len(changed)} changed tensor(s)"
    return {"base": str(base_path), "suspect": str(suspect_path), "summary": summary,
            "tensors_compared": len(tensors), "tensors_changed": len(changed), "findings": findings, "tensors": tensors}


def main(argv=None):
    ap = argparse.ArgumentParser(description="What changed between a model and the base it claims to come from?")
    ap.add_argument("base")
    ap.add_argument("suspect")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--all", action="store_true", help="also list unchanged tensors")
    a = ap.parse_args(argv)
    try:
        report = audit(a.base, a.suspect)
    except (OSError, ValueError, KeyError, struct.error) as e:
        print(f"could not compare: {e}", file=sys.stderr)
        return 2
    if a.json:
        print(json.dumps(report, indent=2))
    else:
        print(report["summary"])
        for f in report["findings"]:
            print(f"\n  {f['tensor']}\n    {f['finding']}")
        rows = [t for t in report["tensors"] if a.all or t["changed"]]
        if rows:
            print(f"\n  {'tensor':<52}{'changed':>10}{'top bit':>9}{'rows':>14}")
            for t in rows[:200]:
                r = f"{t['rows_changed']}/{t['rows']}" if t["rows"] else "-"
                print(f"  {t['tensor'][:51]:<52}{t['changed_fraction']:>9.1%}{t['highest_changed_bit']:>9}{r:>14}")
    return 1 if report["findings"] else 0


if __name__ == "__main__":
    sys.exit(main())
