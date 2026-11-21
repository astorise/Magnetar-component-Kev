"""Generate the small Kev-0.8B pointer-head fixture from a verified head.pt.

Requires PyTorch 2.15.0 CPU. Usage:
    python tests/fixtures/generate_reference.py path/to/head.pt

The input comes from jaredpalmer/kev-0.8b at revision
bf75a6a8848ea6960ff2ed108d9ed44c2941174f. It is loaded with PyTorch's
weights-only unpickler after checking the published SHA-256.
"""

import hashlib
import json
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as functional

EXPECTED = "f400bd12802b2b105ae45d6b03774a158a3db4fccff42413734ddca2e5c920b6"
path = Path(sys.argv[1])
if hashlib.sha256(path.read_bytes()).hexdigest() != EXPECTED:
    raise ValueError("unexpected head.pt checksum")
checkpoint = torch.load(path, map_location="cpu", weights_only=True)
head = checkpoint["head"]
indices = [0, 13, 97]
dimension = head["q.weight"].shape[1]
pointer_dim = head["q.weight"].shape[0]
temperature = checkpoint["temperature"]
cases = [
    ([0.625, -1.25, 0.3], [[1.1, -0.6, 0.7], [0.0, 0.75, -0.25], [-0.1, 0.3, 1.4]]),
    ([-0.2, 0.0, 0.5], [[-0.9, 0.1, 1.0], [0.7, -0.8, 0.0]]),
]


def expand(values):
    result = torch.zeros(dimension, dtype=torch.float32)
    result[indices] = torch.tensor(values, dtype=torch.float32)
    return result


questions = []
with torch.no_grad():
    for decision, options in cases:
        query = functional.linear(expand(decision), head["q.weight"], head["q.bias"])
        keys = functional.linear(torch.stack([expand(o) for o in options]), head["k.weight"], head["k.bias"])
        logits = (keys @ query) * (1.0 / math.sqrt(pointer_dim)) / temperature
        questions.append({
            "decision": decision,
            "options": options,
            "logits": logits.tolist(),
            "probabilities": torch.softmax(logits, dim=-1).tolist(),
        })

fixture = {
    "input_dim": len(indices),
    "pointer_dim": pointer_dim,
    "temperature": temperature,
    "query_weights": head["q.weight"][:, indices].flatten().tolist(),
    "query_bias": head["q.bias"].tolist(),
    "key_weights": head["k.weight"][:, indices].flatten().tolist(),
    "key_bias": head["k.bias"].tolist(),
    "questions": questions,
}
output = Path(__file__).with_name("kev-0.8b-pointer.json")
output.write_text(json.dumps(fixture, indent=2, allow_nan=False) + "\n", encoding="utf-8")
