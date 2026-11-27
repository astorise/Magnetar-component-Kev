"""Create an f32-merged Qwen3.5 checkpoint for the pinned Kev-0.8B adapter.

The inputs must be the exact public revisions documented in the README. This
is an offline conversion, not Runtime adapter activation or a Kev decision API.
The output directory must not already exist. Requires PyTorch and Safetensors.
"""
import argparse
import hashlib
import json
import shutil
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

BASE_SHA = "c2b1e5a17d9c1e27685d92ed9b382911ebb99955ecd89052d1721241adfbab6c"
ADAPTER_SHA = "9b908623acb162118575f4e7a94524f9c139c335be4bfb74d6cfceca01e1885a"
CONFIG_SHA = "b90b86f35c8e6925ef74ee04d0e758f0a845c83a42089ad82bbaa948de9b4204"
TOKENIZER_SHA = "06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523"
TOKENIZER_CONFIG_SHA = "8671bed7c852ce9e661be94f179a7b4ffd091c2a65aea0363e5501c20318ee45"


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def verify(path, expected, normalize_newlines=False):
    actual = (hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
              if normalize_newlines else digest(path))
    if actual != expected:
        raise ValueError(f"unexpected SHA-256 for {path.name}: {actual}")


def expected_modules():
    targets = set()
    for layer in range(24):
        attention = (["linear_attn.in_proj_a", "linear_attn.in_proj_b", "linear_attn.in_proj_qkv",
                      "linear_attn.in_proj_z", "linear_attn.out_proj"] if layer % 4 != 3 else
                     ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj"])
        targets.update(f"layers.{layer}.{name}" for name in attention +
                       ["mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"])
    return targets


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base_dir", type=Path, help="pinned Qwen config and model.safetensors")
    parser.add_argument("adapter", type=Path, help="pinned Kev adapter_model.safetensors")
    parser.add_argument("tokenizer_dir", type=Path, help="pinned Kev tokenizer.json and tokenizer_config.json")
    parser.add_argument("output_dir", type=Path, help="new directory for the derived checkpoint")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    if args.output_dir.exists():
        parser.error("output directory already exists; refusing to overwrite it")
    verify(args.base_dir / "model.safetensors", BASE_SHA)
    verify(args.base_dir / "config.json", CONFIG_SHA, normalize_newlines=True)
    verify(args.adapter, ADAPTER_SHA)
    verify(args.tokenizer_dir / "tokenizer.json", TOKENIZER_SHA)
    verify(args.tokenizer_dir / "tokenizer_config.json", TOKENIZER_CONFIG_SHA)
    torch.set_num_threads(args.threads)
    factors = load_file(str(args.adapter))
    modules = expected_modules()
    expected = {f"base_model.model.{module}.lora_{part}.weight"
                for module in modules for part in ("A", "B")}
    if set(factors) != expected:
        raise ValueError("adapter does not contain the exact 186 target factor pairs")
    weights = load_file(str(args.base_dir / "model.safetensors"))
    with torch.no_grad():
        for index, module in enumerate(sorted(modules), start=1):
            target = f"model.language_model.{module}.weight"
            base = weights[target]
            a = factors[f"base_model.model.{module}.lora_A.weight"]
            b = factors[f"base_model.model.{module}.lora_B.weight"]
            if (base.ndim != 2 or base.dtype != torch.bfloat16 or
                    a.dtype != torch.float32 or b.dtype != torch.float32 or
                    tuple(a.shape) != (16, base.shape[1]) or
                    tuple(b.shape) != (base.shape[0], 16)):
                raise ValueError(f"invalid weight geometry or dtype: {module}")
            merged = base.float() + 2.0 * (b @ a)
            if not torch.isfinite(merged).all():
                raise ValueError(f"non-finite merged weights: {module}")
            weights[target] = merged
            if index % 32 == 0 or index == len(modules):
                print(f"Merged {index}/{len(modules)} projections", flush=True)
    # Creation is exclusive: never replace an existing bundle or source file.
    args.output_dir.mkdir(parents=False, exist_ok=False)
    shutil.copyfile(args.base_dir / "config.json", args.output_dir / "config.json")
    for name in ("tokenizer.json", "tokenizer_config.json"):
        shutil.copyfile(args.tokenizer_dir / name, args.output_dir / name)
    output_weights = args.output_dir / "model.safetensors"
    save_file(weights, str(output_weights), metadata={"format": "pt"})
    output_sha = digest(output_weights)
    provenance = {
        "schema": "magnetar.kev-f32-merged-checkpoint.v1",
        "base_revision": "dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68",
        "base_sha256": BASE_SHA,
        "adapter_revision": "bf75a6a8848ea6960ff2ed108d9ed44c2941174f",
        "adapter_sha256": ADAPTER_SHA,
        "merged_weights_sha256": output_sha,
        "merged_projections": len(modules),
        "rank": 16,
        "scale": 2.0,
        "merged_dtype": "float32",
    }
    (args.output_dir / "kev-merge-provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Merged checkpoint SHA-256: {output_sha}", flush=True)


if __name__ == "__main__":
    main()
