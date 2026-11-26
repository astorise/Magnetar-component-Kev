# magnetar-component-kev

Kev decision model component for [Magnetar](https://github.com/astorise/Magnetar).

## Status

The Rust library implements Kev's CPU pointer-head calculation over caller-provided
hidden states and weights, including biases, scaling, temperature and a stable
per-question softmax. It validates tensor dimensions and non-finite values.
It can also assemble tokenized state and question rows with independent
positions and strict token limits, rejecting injected delimiter IDs.
The Kev-0.8B profile identifies the 24 Qwen3.5 text layers and all 186
LoRA targets in the pinned base/adapter checkpoints. A standalone f32 LoRA
reference applies their standard rank-16 factors; real checkpoint tensor
inventories and independent PyTorch comparisons are covered by the tests.
These standalone references do not load a
checkpoint, tokenize, run a Qwen backbone, produce a model graph, implement
Component Model exports or execute end-to-end inference. Kev is not yet runnable
in Magnetar.

## Work required

Compatible Qwen backbone, adapter loading and Runtime activation, structured
choices and checkpoint temperature integration remain to be implemented.
The necessary Magnetar runtime work is tracked in
[Magnetar #215](https://github.com/astorise/Magnetar/issues/215).

A real implementation must integrate the relevant Magnetar component and
runtime contracts and validate end-to-end outputs against an independent
reference. See the [upstream model or project](https://github.com/jaredpalmer/kev).
No full model weights are included; tests include small subsets of the
Apache-2.0 Kev-0.8B head and adapter weights, with provenance in `tests/fixtures/`.

This repository contains a model component, not a hardware Provider. It is intended to be checked out by Magnetar at `components/kev`.
