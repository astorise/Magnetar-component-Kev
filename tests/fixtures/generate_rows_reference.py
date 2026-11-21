"""Generate the row fixture from a pinned copy of upstream kev/model.py.

Usage: python tests/fixtures/generate_rows_reference.py /path/to/model.py
PyTorch is required because the upstream module imports it. The tokenizer is
a deterministic test double, and the upstream encoder is used unchanged.
"""

import hashlib
import importlib.util
import json
import sys
import types
from pathlib import Path

source = Path(sys.argv[1])
expected = "2ff3b2e67c048e3bc5a0d1a32aafe50804ed315a5564fce06bb4b95697c5271c"
if hashlib.sha256(source.read_bytes()).hexdigest() != expected:
    raise ValueError("unexpected upstream model.py checksum")

# Only the encoder and rows_of are exercised. The module imports transformer
# classes at load time, so stand in for those otherwise-unused dependencies.
transformers = types.ModuleType("transformers")
for name in ("AutoModel", "AutoModelForCausalLM", "AutoTokenizer", "DynamicCache"):
    setattr(transformers, name, type(name, (), {}))
cache_utils = types.ModuleType("transformers.cache_utils")
cache_utils.LinearAttentionCacheLayerMixin = type("LinearAttentionCacheLayerMixin", (), {})
sys.modules["transformers"] = transformers
sys.modules["transformers.cache_utils"] = cache_utils
spec = importlib.util.spec_from_file_location("kev_reference", source)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class FakeTokenizer:
    def convert_tokens_to_ids(self, token):
        return {value: 1000 + index for index, value in enumerate(module.SPECIAL)}[token]

    def __call__(self, text, **kwargs):
        return types.SimpleNamespace(input_ids=[ord(char) for char in text])


record = {
    "state": "AB",
    "questions": [
        {"instr": "C", "options": ["D", "EF"], "label": 0},
        {"instr": "G", "options": ["H"], "label": 0},
    ],
}
encoded = module.encode(FakeTokenizer(), record, strict=True)
prefix, prefix_positions, rows = module.rows_of(encoded)
fixture = {"prefix": prefix, "prefix_positions": prefix_positions, "rows": rows}
Path(__file__).with_name("kev-upstream-rows.json").write_text(
    json.dumps(fixture, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
