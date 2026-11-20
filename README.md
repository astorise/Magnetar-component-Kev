# magnetar-component-kev

Placeholder for Kev decision models in [Magnetar](https://github.com/astorise/Magnetar).

## Status

This is a scaffold only. It has no model graph, loader, weights, inference implementation, Component Model exports or supported checkpoint. Creating this repository does not make the model runnable in Magnetar.

## Work required

Compatible Qwen backbone, checkpoint-specific LoRA or full weights, pointer-head scoring, structured choices and calibration.

A real implementation must identify a specific checkpoint and its license, integrate the relevant Magnetar component and runtime contracts, and validate outputs against an independent reference. See the [upstream model or project](https://github.com/jaredpalmer/kev). Model weights are not included.

This repository contains a model component, not a hardware Provider. It is intended to be checked out by Magnetar at `components/kev`.
