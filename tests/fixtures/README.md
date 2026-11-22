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

## Kev-0.8B Qwen3.5 and adapter

The selected adapter is [`jaredpalmer/kev-0.8b`](https://huggingface.co/jaredpalmer/kev-0.8b/tree/bf75a6a8848ea6960ff2ed108d9ed44c2941174f),
revision `bf75a6a8848ea6960ff2ed108d9ed44c2941174f`; its Safetensors
SHA-256 is `9b908623acb162118575f4e7a94524f9c139c335be4bfb74d6cfceca01e1885a`.
The adapter configuration does not specify a base revision. For reproducible
tests we chose [`Qwen/Qwen3.5-0.8B-Base`](https://huggingface.co/Qwen/Qwen3.5-0.8B-Base/tree/dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68),
revision `dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68`. Both are Apache-2.0.
Its raw Safetensors JSON header has SHA-256
`c894b6055113f9d5a9c5c68e703662de8ba27df5a9631b47f10b75ac3fe0b083`.

The compact headers retain names, shapes and dtypes, not tensor payloads. The
small LoRA reference contains selected real factor rows and columns and
PyTorch outputs for synthetic inputs. To regenerate, download the adapter
Safetensors, extract the raw JSON header from the base file (the eight-byte
little-endian header length is not part of the extracted JSON), and run:

```text
python tests/fixtures/generate_adapter_reference.py ADAPTER.safetensors BASE_HEADER.json
```

For the additional full-factor comparison, supply a third argument naming
`kev-lora-full-reference.json` and set `MAGNETAR_KEV_LORA_DIR` to a directory
containing that file and the original checkpoint named `kev-adapter.safetensors`.
Then run `cargo test --locked --test qwen35_adapter full_real_adapter_projections_match_pytorch -- --ignored --nocapture`.
The generator checks the checkpoint, header and configuration hashes; the Rust
test rechecks the full adapter hash. It compares f32 LoRA outputs for two real
factor pairs with synthetic hidden states and synthetic base outputs. It does
not test native BF16, a complete Qwen model, or Runtime adapter activation.
