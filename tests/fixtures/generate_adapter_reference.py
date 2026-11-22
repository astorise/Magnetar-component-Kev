"""Generate small real-weight LoRA fixtures for the pinned Kev adapter.

Usage: python generate_adapter_reference.py ADAPTER.safetensors BASE_HEADER.json [FULL_REFERENCE.json]
The base header is the raw JSON header from the pinned base Safetensors file.
PyTorch is used independently for the reference calculations.
"""

import hashlib
import json
import struct
import sys
from pathlib import Path

import torch
import torch.nn.functional as functional

ADAPTER_SHA = "9b908623acb162118575f4e7a94524f9c139c335be4bfb74d6cfceca01e1885a"
BASE_HEADER_SHA = "c894b6055113f9d5a9c5c68e703662de8ba27df5a9631b47f10b75ac3fe0b083"
root = Path(__file__).parent
adapter = Path(sys.argv[1]).read_bytes()
base = Path(sys.argv[2]).read_bytes()
for name, digest in [
    ("kev-0.8b-adapter-config.json", "748acb2cda88454cb1ba69d745ba336f3fcb5486eac349e90960c8b8d8d3e854"),
    ("qwen3.5-0.8b-base-config.json", "b90b86f35c8e6925ef74ee04d0e758f0a845c83a42089ad82bbaa948de9b4204"),
]:
    # Git may check these JSON fixtures out with CRLF on Windows; their
    # published bytes use LF and are what the digest identifies.
    raw_config = (root / name).read_bytes().replace(b"\r\n", b"\n")
    if hashlib.sha256(raw_config).hexdigest() != digest:
        raise ValueError(f"unexpected pinned config digest: {name}")
if hashlib.sha256(adapter).hexdigest() != ADAPTER_SHA:
    raise ValueError("unexpected adapter checkpoint digest")
if hashlib.sha256(base).hexdigest() != BASE_HEADER_SHA:
    raise ValueError("unexpected base checkpoint header digest")
header_size = struct.unpack("<Q", adapter[:8])[0]
header = json.loads(adapter[8 : 8 + header_size])
payload = memoryview(adapter)[8 + header_size :]


def compact_header(source):
    return {name: {"dtype": item["dtype"], "shape": item["shape"]}
            for name, item in source.items() if name != "__metadata__"}


def tensor(name):
    item = header[name]
    if item["dtype"] != "F32":
        raise ValueError(f"unexpected dtype for {name}")
    start, end = item["data_offsets"]
    # Clone avoids retaining a non-writable borrowed view.
    return torch.frombuffer(bytearray(payload[start:end]), dtype=torch.float32).clone().reshape(item["shape"])


cases = []
for module, rows in [
    ("layers.0.linear_attn.in_proj_a", [0, 5, 10, 15]),
    ("layers.3.self_attn.q_proj", [0, 7, 255, 4095]),
]:
    prefix = "base_model.model." + module
    indices = [0, 13, 97]
    a = tensor(prefix + ".lora_A.weight")[:, indices]
    b = tensor(prefix + ".lora_B.weight")[rows, :]
    for values in [[0.625, -1.25, 0.3], [-0.2, 0.0, 0.5]]:
        x = torch.tensor(values, dtype=torch.float32)
        base_output = torch.tensor([0.25, -0.5, 1.0, -1.5], dtype=torch.float32)
        with torch.no_grad():
            result = base_output + 2.0 * functional.linear(functional.linear(x, a), b)
        cases.append({"module": module, "rank": 16, "input": values,
                      "base": base_output.tolist(), "a": a.flatten().tolist(),
                      "b": b.flatten().tolist(), "output": result.tolist()})

for filename, value in [
    ("kev-0.8b-adapter-header.json", compact_header(header)),
    ("qwen3.5-0.8b-base-header.json", compact_header(json.loads(base))),
    ("kev-0.8b-lora-reference.json", cases),
]:
    (root / filename).write_text(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                            allow_nan=False) + "\n", encoding="utf-8")

# Optional larger reference: both full factors, dense synthetic hidden states,
# and every output column. Actual factor payloads remain in the checkpoint.
if len(sys.argv) > 3:
    full = []
    for module in ["layers.0.linear_attn.in_proj_a", "layers.3.self_attn.q_proj"]:
        prefix = "base_model.model." + module
        a = tensor(prefix + ".lora_A.weight")
        b = tensor(prefix + ".lora_B.weight")
        x = torch.tensor([((i * 17) % 101 - 50) / 101.0 for i in range(a.shape[1])])
        base_output = torch.tensor([((i * 11) % 83 - 41) / 83.0 for i in range(b.shape[0])])
        with torch.no_grad():
            result = base_output + 2.0 * functional.linear(functional.linear(x, a), b)
        full.append({"module": module, "input": x.tolist(), "base": base_output.tolist(),
                     "output": result.tolist()})
    Path(sys.argv[3]).write_text(json.dumps(full, allow_nan=False), encoding="utf-8")
