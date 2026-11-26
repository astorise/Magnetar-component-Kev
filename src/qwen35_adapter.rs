//! Exact Kev-0.8B/Qwen3.5 adapter inventory and a CPU LoRA reference.
//!
//! This profile does not authenticate files or activate an adapter in the
//! Runtime. Callers must verify provenance and use a validated Safetensors
//! parser before comparing an inventory or decoding its tensor payloads.

use std::{collections::BTreeMap, error::Error, fmt};

const BASE_CONFIG: &[u8] = include_bytes!("../tests/fixtures/qwen3.5-0.8b-base-config.json");
const ADAPTER_CONFIG: &[u8] = include_bytes!("../tests/fixtures/kev-0.8b-adapter-config.json");
const RANK: usize = 16;
const HIDDEN: usize = 1024;
const INTERMEDIATE: usize = 3584;
const MAX_ELEMENTS: usize = 67_108_864;

/// The two distinct Qwen3.5 text-layer families in the selected base model.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LayerKind {
    LinearAttention,
    FullAttention,
}

/// One checkpoint tensor's storage properties, as read by a trusted parser.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TensorInfo {
    pub dtype: String,
    pub shape: Vec<usize>,
}

/// The two adapter factors and the base projection they modify.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LoraTarget {
    pub layer: usize,
    /// Name in the pinned base Safetensors checkpoint, stored `[out, in]`.
    pub base_checkpoint_name: String,
    pub a_checkpoint_name: String,
    pub b_checkpoint_name: String,
    pub input_dim: usize,
    pub output_dim: usize,
    pub rank: usize,
}

/// The exact selected model and adapter, not a generic Qwen3.5 parser.
#[derive(Clone, Debug)]
pub struct Kev08bProfile {
    layers: Vec<LayerKind>,
    targets: Vec<LoraTarget>,
}

impl Kev08bProfile {
    /// Accept only the two pinned JSON configurations, compared as JSON values.
    /// The upstream adapter does not itself pin its base revision: the chosen
    /// base revision is documented with the fixtures. This check is not a
    /// cryptographic authenticity check.
    pub fn from_configs(base: &[u8], adapter: &[u8]) -> Result<Self, ProfileError> {
        fn matches_config(actual: &[u8], expected: &[u8]) -> bool {
            if actual.len() > 1_048_576 {
                return false;
            }
            let parsed = serde_json::from_slice::<serde_json::Value>(actual);
            let pinned = serde_json::from_slice::<serde_json::Value>(expected);
            matches!((parsed, pinned), (Ok(a), Ok(b)) if a == b)
        }
        if !matches_config(base, BASE_CONFIG) || !matches_config(adapter, ADAPTER_CONFIG) {
            return Err(ProfileError::UnsupportedConfig);
        }

        let mut layers = Vec::with_capacity(24);
        let mut targets = Vec::with_capacity(186);
        for layer in 0..24 {
            let kind = if layer % 4 == 3 {
                LayerKind::FullAttention
            } else {
                LayerKind::LinearAttention
            };
            layers.push(kind);
            let mut add = |module: &str, input_dim: usize, output_dim: usize| {
                let suffix = format!("layers.{layer}.{module}");
                targets.push(LoraTarget {
                    layer,
                    base_checkpoint_name: format!("model.language_model.{suffix}.weight"),
                    a_checkpoint_name: format!("base_model.model.{suffix}.lora_A.weight"),
                    b_checkpoint_name: format!("base_model.model.{suffix}.lora_B.weight"),
                    input_dim,
                    output_dim,
                    rank: RANK,
                });
            };
            match kind {
                LayerKind::LinearAttention => {
                    add("linear_attn.in_proj_a", HIDDEN, 16);
                    add("linear_attn.in_proj_b", HIDDEN, 16);
                    add("linear_attn.in_proj_qkv", HIDDEN, 6144);
                    add("linear_attn.in_proj_z", HIDDEN, 2048);
                    add("linear_attn.out_proj", 2048, HIDDEN);
                }
                LayerKind::FullAttention => {
                    // Q includes both the query and its output gate.
                    add("self_attn.q_proj", HIDDEN, 4096);
                    add("self_attn.k_proj", HIDDEN, 512);
                    add("self_attn.v_proj", HIDDEN, 512);
                    add("self_attn.o_proj", 2048, HIDDEN);
                }
            }
            add("mlp.gate_proj", HIDDEN, INTERMEDIATE);
            add("mlp.up_proj", HIDDEN, INTERMEDIATE);
            add("mlp.down_proj", INTERMEDIATE, HIDDEN);
        }
        Ok(Self { layers, targets })
    }

    pub fn layers(&self) -> &[LayerKind] {
        &self.layers
    }

    pub fn targets(&self) -> &[LoraTarget] {
        &self.targets
    }

    /// The adapter uses alpha=32 and rank=16, without rank-stabilized scaling.
    pub fn scale(&self) -> f32 {
        2.0
    }

    /// Reject missing, extra, incorrectly typed or incorrectly shaped factors.
    /// This only compares a previously validated inventory, not payload bytes.
    pub fn validate_inventory(
        &self,
        inventory: &BTreeMap<String, TensorInfo>,
    ) -> Result<(), ProfileError> {
        if inventory.len() != self.targets.len() * 2 {
            return Err(ProfileError::InventoryMismatch);
        }
        for target in &self.targets {
            for (name, shape) in [
                (&target.a_checkpoint_name, [target.rank, target.input_dim]),
                (&target.b_checkpoint_name, [target.output_dim, target.rank]),
            ] {
                if !inventory
                    .get(name)
                    .is_some_and(|info| info.dtype == "F32" && info.shape == shape)
                {
                    return Err(ProfileError::InventoryMismatch);
                }
            }
        }
        Ok(())
    }

    /// Check every target against the pinned base checkpoint. Other base
    /// tensors are allowed because this is only the adapter's target subset.
    pub fn validate_base_inventory(
        &self,
        inventory: &BTreeMap<String, TensorInfo>,
    ) -> Result<(), ProfileError> {
        for target in &self.targets {
            if !inventory
                .get(&target.base_checkpoint_name)
                .is_some_and(|info| {
                    info.dtype == "BF16" && info.shape == [target.output_dim, target.input_dim]
                })
            {
                return Err(ProfileError::InventoryMismatch);
            }
        }
        Ok(())
    }
}

/// Apply one standard LoRA overlay in f32: `base + scale * B * (A * x)`.
/// A and B are checkpoint-order row-major `[rank, input]` and `[output, rank]`.
/// The returned vector is new; invalid input never mutates the base output.
pub fn apply_lora_f32(
    input: &[f32],
    base_output: &[f32],
    a: &[f32],
    b: &[f32],
    rank: usize,
    scale: f32,
) -> Result<Vec<f32>, ProfileError> {
    let input_dim = input.len();
    let output_dim = base_output.len();
    let size = |left: usize, right: usize| {
        left.checked_mul(right)
            .filter(|&count| count > 0 && count <= MAX_ELEMENTS)
    };
    if size(rank, input_dim) != Some(a.len())
        || size(output_dim, rank) != Some(b.len())
        || !scale.is_finite()
    {
        return Err(ProfileError::InvalidDimensions);
    }
    if input
        .iter()
        .chain(base_output)
        .chain(a)
        .chain(b)
        .any(|value| !value.is_finite())
    {
        return Err(ProfileError::NonFiniteValue);
    }
    let intermediate: Vec<f32> = a
        .chunks_exact(input_dim)
        .map(|row| row.iter().zip(input).map(|(weight, x)| weight * x).sum())
        .collect();
    let output: Vec<f32> = b
        .chunks_exact(rank)
        .zip(base_output)
        .map(|(row, base)| {
            let delta: f32 = row
                .iter()
                .zip(&intermediate)
                .map(|(weight, value)| weight * value)
                .sum();
            base + scale * delta
        })
        .collect();
    if output.iter().any(|value| !value.is_finite()) {
        return Err(ProfileError::NonFiniteValue);
    }
    Ok(output)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ProfileError {
    UnsupportedConfig,
    InventoryMismatch,
    InvalidDimensions,
    NonFiniteValue,
}

impl fmt::Display for ProfileError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::UnsupportedConfig => "unsupported Kev/Qwen3.5 configuration",
            Self::InventoryMismatch => {
                "Kev LoRA tensor inventory does not match the pinned profile"
            }
            Self::InvalidDimensions => "invalid LoRA dimensions or scale",
            Self::NonFiniteValue => "non-finite LoRA input, weight or output",
        })
    }
}

impl Error for ProfileError {}
