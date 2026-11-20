# Kev pointer-head reference

`kev-0.8b-pointer.json` contains a small subset of the learned projection
weights and reference scores from
[`jaredpalmer/kev-0.8b`](https://huggingface.co/jaredpalmer/kev-0.8b/tree/bf75a6a8848ea6960ff2ed108d9ed44c2941174f),
revision `bf75a6a8848ea6960ff2ed108d9ed44c2941174f`, distributed under
Apache-2.0. The source `head.pt` SHA-256 is
`f400bd12802b2b105ae45d6b03774a158a3db4fccff42413734ddca2e5c920b6`.

`generate_reference.py` validates the source hash, reads it with PyTorch's
weights-only loader, and calculates the reference using the full original
1024-dimensional weight matrices and sparse synthetic hidden states. The
fixture retains only the three columns required for those inputs. The 256-wide
pointer dimension, trained biases and checkpoint temperature remain unchanged.
This validates the head numerics, not the backbone or inference pipeline.

To reproduce with PyTorch 2.15.0 CPU installed, download that revision's
`head.pt` and run `python tests/fixtures/generate_reference.py path/to/head.pt`.
The scoring formula follows upstream
[`kev/model.py`](https://github.com/jaredpalmer/kev/blob/65b1e763943ace53fd3a4aae00d8d08dbd849772/kev/model.py).
See `UPSTREAM-LICENSE` for the fixture's Apache-2.0 license.

`kev-upstream-rows.json` records upstream `encode` and `rows_of` output for a
two-question request and a deterministic fake tokenizer. To regenerate it,
download `kev/model.py` at revision `65b1e763943ace53fd3a4aae00d8d08dbd849772`
and run `python tests/fixtures/generate_rows_reference.py path/to/model.py`.
The script verifies its SHA-256 and requires PyTorch.
