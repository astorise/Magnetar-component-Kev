# magnetar-component-kev

Kev decision model component for [Magnetar](https://github.com/astorise/Magnetar).

## Status

The Rust library implements Kev's CPU pointer-head calculation over caller-provided
hidden states and weights, including biases, scaling, temperature and a stable
per-question softmax. It validates tensor dimensions and non-finite values.
It can also assemble tokenized state and question rows with independent
positions and strict token limits, rejecting injected delimiter IDs.
This is a reference implementation of the head only: it does not load a
checkpoint, tokenize, run a Qwen backbone, produce a model graph, implement
Component Model exports or execute end-to-end inference. Kev is not yet runnable
in Magnetar.

## Work required

Compatible Qwen backbone, checkpoint-specific LoRA or full weights, model
loading, structured choices and checkpoint temperature integration remain to
be implemented. The necessary Magnetar runtime work is tracked in
[Magnetar #215](https://github.com/astorise/Magnetar/issues/215).

A real implementation must integrate the relevant Magnetar component and
runtime contracts and validate end-to-end outputs against an independent
reference. See the [upstream model or project](https://github.com/jaredpalmer/kev).
No full model weights are included; tests include a small subset of the
Apache-2.0 Kev-0.8B head weights, with provenance in `tests/fixtures/`.

This repository contains a model component, not a hardware Provider. It is intended to be checked out by Magnetar at `components/kev`.
