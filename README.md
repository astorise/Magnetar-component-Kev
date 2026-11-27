# magnetar-component-kev

Kev decision model component for [Magnetar](https://github.com/astorise/Magnetar).

## Status

The Rust library implements Kev's CPU pointer-head calculation over caller-provided
hidden states and weights, including biases, scaling, temperature and a stable
per-question softmax. It validates tensor dimensions and non-finite values.
It can also assemble tokenized state and question rows with independent
positions and strict token limits, rejecting injected delimiter IDs.
The pointer head can be constructed from already-decoded checkpoint tensors and
can select the decision and option states from an encoded question row. Callers
must authenticate and safely decode the head file, and run each question with
an independent copy of the shared backbone state.
The Kev-0.8B profile identifies the 24 Qwen3.5 text layers and all 186
LoRA targets in the pinned base/adapter checkpoints. A standalone f32 LoRA
reference applies their standard rank-16 factors; real checkpoint tensor
inventories and independent PyTorch comparisons are covered by the tests.
An f32 merge reference can also fold a single pair of decoded LoRA factors into
its checkpoint-order base projection. The caller must authenticate the inputs
and bind the result to the matching model; merging may reassociate f32 math.
These standalone references do not load a
checkpoint, tokenize, run a Qwen backbone, produce a model graph, implement
Component Model exports or execute end-to-end inference. Kev is not yet runnable
in Magnetar.

## Work required

Adapter loading and Runtime activation, structured choices and checkpoint
temperature integration with the existing Qwen backbone remain to be implemented.
The necessary Magnetar runtime work is tracked in
[Magnetar #215](https://github.com/astorise/Magnetar/issues/215).

## Offline adapter conversion

`tools/merge_kev_checkpoint.py` converts the pinned Kev-0.8B LoRA adapter and
Qwen3.5-0.8B-Base checkpoint to a new Safetensors bundle. It verifies the exact
base, adapter, configuration and Kev tokenizer hashes, checks all 186 factor
pairs, and computes `W + 2 * B @ A` in float32. The new bundle includes a
provenance file with its measured weights hash. Other base tensors keep their
original storage dtype; use float32 computation for comparison with this path.
The tool requires PyTorch and Safetensors, substantial RAM and roughly 2.75 GB
of free output space. It refuses to overwrite an existing output directory.

```sh
python tools/merge_kev_checkpoint.py /path/to/qwen-base \
  /path/to/adapter_model.safetensors /path/to/kev-tokenizer \
  /path/to/new-merged-bundle
```

The converter does not authenticate a file merely by its source path: it checks
the pinned hashes embedded in the script. Source files must remain unchanged
throughout conversion. It does not export `head.pt`, load a Runtime adapter,
or implement the structured Kev decision interface. The result is a derived
checkpoint whose weights digest differs from the base's.

A real implementation must integrate the relevant Magnetar component and
runtime contracts and validate end-to-end outputs against an independent
reference. See the [upstream model or project](https://github.com/jaredpalmer/kev).
No full model weights are included; tests include small subsets of the
Apache-2.0 Kev-0.8B head and adapter weights, with provenance in `tests/fixtures/`.

This repository contains a model component, not a hardware Provider. It is intended to be checked out by Magnetar at `components/kev`.
