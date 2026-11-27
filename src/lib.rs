//! Kev pointer-head reference implementation.
//!
//! Given the final hidden state of a decision token and one hidden state per
//! option's closing token, this computes the two learned linear projections,
//! scaled dot products, temperature adjustment and per-question softmax.
//! It does not load checkpoints, run Qwen, tokenize text or implement a
//! Magnetar Component Model export.

use std::collections::BTreeMap;
use std::error::Error;
use std::fmt;

pub mod encoding;
pub mod qwen35_adapter;

/// Row-major affine transform, with weights shaped `[output_dim, input_dim]`.
#[derive(Debug, Clone)]
pub struct LinearProjection {
    input_dim: usize,
    output_dim: usize,
    weights: Vec<f32>,
    bias: Vec<f32>,
}

impl LinearProjection {
    /// Construct a projection, rejecting inconsistent or non-finite weights.
    pub fn new(
        input_dim: usize,
        output_dim: usize,
        weights: Vec<f32>,
        bias: Vec<f32>,
    ) -> Result<Self, HeadError> {
        if input_dim == 0 || output_dim == 0 {
            return Err(HeadError::InvalidDimensions);
        }
        let size = input_dim
            .checked_mul(output_dim)
            .ok_or(HeadError::InvalidDimensions)?;
        if weights.len() != size || bias.len() != output_dim {
            return Err(HeadError::InvalidDimensions);
        }
        if weights.iter().chain(&bias).any(|value| !value.is_finite()) {
            return Err(HeadError::NonFiniteValue);
        }
        Ok(Self {
            input_dim,
            output_dim,
            weights,
            bias,
        })
    }

    fn apply(&self, input: &[f32]) -> Result<Vec<f32>, HeadError> {
        if input.len() != self.input_dim {
            return Err(HeadError::InvalidDimensions);
        }
        if input.iter().any(|value| !value.is_finite()) {
            return Err(HeadError::NonFiniteValue);
        }
        self.weights
            .chunks_exact(self.input_dim)
            .zip(&self.bias)
            .map(|(row, bias)| {
                let result = row
                    .iter()
                    .zip(input)
                    .fold(*bias, |sum, (weight, value)| sum + weight * value);
                if result.is_finite() {
                    Ok(result)
                } else {
                    Err(HeadError::NonFiniteValue)
                }
            })
            .collect()
    }
}

/// Kev's query and key projections plus inference-time temperature.
#[derive(Debug, Clone)]
pub struct PointerHead {
    query: LinearProjection,
    key: LinearProjection,
    temperature: f32,
}

/// Calibrated logits and probabilities, in the caller's option order.
#[derive(Debug, Clone, PartialEq)]
pub struct DecisionScores {
    pub logits: Vec<f32>,
    pub probabilities: Vec<f32>,
}

/// One already-decoded tensor from the checkpoint's `head` state dictionary.
/// Authentication and safe decoding of the original file are caller-owned.
#[derive(Debug, Clone, PartialEq)]
pub struct CheckpointTensor {
    pub shape: Vec<usize>,
    pub values: Vec<f32>,
}

impl PointerHead {
    /// Construct the pointer head from `q.weight`, `q.bias`, `k.weight` and
    /// `k.bias` in checkpoint order. The caller supplies the checkpoint's
    /// temperature separately and must authenticate and safely decode it.
    pub fn from_checkpoint_tensors(
        mut tensors: BTreeMap<String, CheckpointTensor>,
        temperature: f32,
    ) -> Result<Self, HeadError> {
        if tensors.len() != 4 {
            return Err(HeadError::InvalidTensorInventory);
        }
        let q_weight = tensors
            .remove("q.weight")
            .ok_or(HeadError::InvalidTensorInventory)?;
        let q_bias = tensors
            .remove("q.bias")
            .ok_or(HeadError::InvalidTensorInventory)?;
        let k_weight = tensors
            .remove("k.weight")
            .ok_or(HeadError::InvalidTensorInventory)?;
        let k_bias = tensors
            .remove("k.bias")
            .ok_or(HeadError::InvalidTensorInventory)?;
        let [pointer, hidden] = q_weight.shape.as_slice() else {
            return Err(HeadError::InvalidTensorInventory);
        };
        if *pointer == 0
            || *hidden == 0
            || k_weight.shape != q_weight.shape
            || q_bias.shape != [*pointer]
            || k_bias.shape != [*pointer]
        {
            return Err(HeadError::InvalidTensorInventory);
        }
        let query = LinearProjection::new(*hidden, *pointer, q_weight.values, q_bias.values)?;
        let key = LinearProjection::new(*hidden, *pointer, k_weight.values, k_bias.values)?;
        Self::new(query, key, temperature)
    }

    /// Both projections must map the same hidden dimension to the same
    /// nonzero pointer dimension; temperature must be finite and positive.
    pub fn new(
        query: LinearProjection,
        key: LinearProjection,
        temperature: f32,
    ) -> Result<Self, HeadError> {
        if query.input_dim != key.input_dim || query.output_dim != key.output_dim {
            return Err(HeadError::InvalidDimensions);
        }
        if !temperature.is_finite() || temperature <= 0.0 {
            return Err(HeadError::InvalidTemperature);
        }
        Ok(Self {
            query,
            key,
            temperature,
        })
    }

    /// Score one question. `decision` is its `<decide>` hidden state;
    /// `options` contains the `</opt>` hidden state of each option, in order.
    ///
    /// The caller is responsible for producing correctly isolated question
    /// states with a compatible backbone. No confidence calibration is
    /// claimed beyond applying the caller-supplied checkpoint temperature.
    pub fn score(&self, decision: &[f32], options: &[&[f32]]) -> Result<DecisionScores, HeadError> {
        if options.is_empty() {
            return Err(HeadError::NoOptions);
        }
        let query = self.query.apply(decision)?;
        let scale = (self.query.output_dim as f32).sqrt().recip();
        let mut logits = Vec::with_capacity(options.len());
        for option in options {
            let key = self.key.apply(option)?;
            let dot = key.iter().zip(&query).map(|(k, q)| k * q).sum::<f32>();
            let logit = dot * scale / self.temperature;
            if !logit.is_finite() {
                return Err(HeadError::NonFiniteValue);
            }
            logits.push(logit);
        }
        // Max subtraction keeps the exponentials bounded for large logits.
        let maximum = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut probabilities: Vec<f32> = logits.iter().map(|z| (z - maximum).exp()).collect();
        let normalizer: f32 = probabilities.iter().sum();
        for probability in &mut probabilities {
            *probability /= normalizer;
        }
        Ok(DecisionScores {
            logits,
            probabilities,
        })
    }

    /// Score a single independently executed question row. `hidden_states`
    /// is the row's contiguous `[row.token_ids.len(), hidden_size]` output,
    /// excluding the shared prefix. Option and decision indices are relative
    /// to that row, as produced by [`encoding::encode_rows`]. The caller must
    /// supply an encoded row: structural checks cannot authenticate token IDs
    /// or establish that the backbone evaluated the row independently.
    pub fn score_row(
        &self,
        row: &encoding::QuestionRow,
        hidden_states: &[f32],
    ) -> Result<DecisionScores, HeadError> {
        let tokens = row.token_ids.len();
        if tokens == 0
            || row.position_ids.len() != tokens
            || row.position_ids.first().copied().unwrap_or(0) == 0
            || row
                .position_ids
                .windows(2)
                .any(|pair| pair[0].checked_add(1) != Some(pair[1]))
            || row.decision_index != tokens - 1
            || !(1..=255).contains(&row.option_end_indices.len())
            || row
                .option_end_indices
                .iter()
                .any(|&index| index < 2 || index >= row.decision_index)
            || row
                .option_end_indices
                .windows(2)
                .any(|pair| pair[0].checked_add(2).is_none_or(|min| pair[1] < min))
        {
            return Err(HeadError::InvalidRow);
        }
        let hidden = self.query.input_dim;
        if tokens.checked_mul(hidden) != Some(hidden_states.len()) {
            return Err(HeadError::InvalidDimensions);
        }
        if hidden_states.iter().any(|value| !value.is_finite()) {
            return Err(HeadError::NonFiniteValue);
        }
        let state = |index| &hidden_states[index * hidden..(index + 1) * hidden];
        let options: Vec<_> = row
            .option_end_indices
            .iter()
            .map(|&index| state(index))
            .collect();
        self.score(state(row.decision_index), &options)
    }
}

/// Invalid weights, hidden states, or scoring parameters.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HeadError {
    InvalidDimensions,
    InvalidTemperature,
    NonFiniteValue,
    NoOptions,
    InvalidTensorInventory,
    InvalidRow,
}

impl fmt::Display for HeadError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let message = match self {
            Self::InvalidDimensions => "inconsistent or empty tensor dimensions",
            Self::InvalidTemperature => "temperature must be finite and positive",
            Self::NonFiniteValue => "non-finite input or intermediate value",
            Self::NoOptions => "at least one option is required",
            Self::InvalidTensorInventory => "pointer-head tensor inventory is invalid",
            Self::InvalidRow => "question row indices or positions are invalid",
        };
        formatter.write_str(message)
    }
}

impl Error for HeadError {}
