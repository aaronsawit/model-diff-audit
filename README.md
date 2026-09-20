# model-diff-audit

What actually changed between this model and the base it claims to come from?

A checkpoint you downloaded says "fine-tuned from X". You can check that against X. This tool compares two
`safetensors` checkpoints tensor by tensor, **at the bit level**, without loading either model or running any code
from them, and tells you whether the differences look like training or like tampering.

```
$ python3 model_diff_audit.py ./stock-base ./downloaded-model
needs a look: 1 finding(s) across 197 changed tensor(s)

  model.embed_tokens.weight
    sparse row edits: 90 of 151,936 rows changed, every other row is bit-identical. In an embedding these
    are specific tokens: check what they are.

  tensor                                  changed  top bit          rows
  model.embed_tokens.weight                  0.1%       16     90/151936
  model.layers.0.mlp.down_proj.weight       69.5%       16     2048/2048
  ...
```

That output is real. It is from a capture-the-flag model in which a message had been hidden inside the weights.
The rest of the model had been genuinely fine-tuned, so at a glance it looked like any other checkpoint. The
embedding gave it away: 151,846 rows untouched, 90 rows edited. Finding those rows by hand took me most of a
day. This finds them in one command.

## What it looks for

Honest training moves almost every number in a tensor, and moves all of its bits. These patterns are different:

| Finding | What it means |
| --- | --- |
| **low-bit edits** | values differ from the base only in their lowest few bits. Optimisers do not produce that. Writing hidden data into a model does |
| **sparse row edits** | a handful of rows changed in an embedding or projection, every other row bit-identical. In an embedding, rows are tokens: planted trigger tokens for a backdoor, or rows used to carry data |
| **lone change** | one or two tensors changed out of hundreds. Fine-tuning normally touches every trained layer |
| missing / new / reshaped tensors | the architecture is not what the base is |

An ordinary fine-tune reports `ordinary fine-tune: N of M tensors changed, densely and in all bits` and exits 0.

## Use

```bash
pip install numpy
python3 model_diff_audit.py BASE SUSPECT            # files, or folders of sharded *.safetensors
python3 model_diff_audit.py BASE SUSPECT --json
python3 model_diff_audit.py BASE SUSPECT --all      # list unchanged tensors too
```

Exit code `0` identical or an ordinary fine-tune, `1` something needs a human look, `2` could not compare.

It parses the safetensors header itself and memory-maps the data, so it never imports the model's code, never
unpickles anything, and handles models larger than RAM. An 8 GB model pair took about four and a half minutes
from a hard disk.

## Limits, stated plainly

- It needs the **real base** to compare against. With no baseline there is nothing to diff.
- **Dense training hides low-bit edits.** In the CTF model the hidden payload sat in the low bits of one MLP
  layer, but that layer had also been fine-tuned, so every bit had moved and the low-bit check could not see it.
  The sparse embedding edits were what gave it away. Low-bit detection works when the carrier is otherwise
  untouched, which is the lazy and common case.
- A finding is a reason to look, not proof of malice. Adding new special tokens legitimately edits a few
  embedding rows. The tool lists the row numbers so you can look the tokens up.
- A clean result is not a safety certificate. A backdoor trained in with ordinary fine-tuning looks like
  ordinary fine-tuning. This catches edits made **to the file**, not behaviour learned in training.
- `safetensors` only. Pickle-based `.bin` / `.pt` files are not read, on purpose: loading them runs code.

## Tests

```bash
pip install numpy
python3 -m unittest discover -s tests -v
```

The tests write tiny safetensors files by hand and cover: identical models, an ordinary fine-tune (must not be
flagged), data hidden in low bits, edited embedding rows, a lone changed layer, missing and reshaped tensors,
half precision, and sharded folders.

MIT licence.
